"""OFC Pineapple - WebSocket Message Handlers"""
import time
import asyncio

from ai.engine.game_engine import evaluate_board_with_joker_constraint
from ai.engine.scoring import (
    calculate_scores, check_session_end,
)

from .room import GameRoom, GameState, db_logger, solve_fantasyland
from .ai_player import AI_PLAYER_ID, init_ai, ai_select_action_sync


async def ai_auto_play(room: GameRoom):
    """Auto-play AI's turn. Runs as background task."""
    if not room.is_ai_game or not room.game:
        return

    async with room.ai_lock:
        ai_seat = room.ai_seat
        human_seat = 1 - ai_seat

        # Skip if AI already placed this turn or board is complete
        if room.game.placed_this_turn[ai_seat]:
            return
        if not room.game.dealt_cards.get(ai_seat):
            return

        # Capture pre-action board state
        board_before = {k: list(v) for k, v in room.game.boards[ai_seat].items()}
        opp_board_before = {k: list(v) for k, v in room.game.boards[human_seat].items()}

        # Compute AI action in background thread (non-blocking)
        print(f"[AI] Computing action for turn {room.game.turn}, seat {ai_seat}...")
        t0 = time.time()
        result = await asyncio.to_thread(ai_select_action_sync, room.game, ai_seat)
        elapsed = time.time() - t0
        print(f"[AI] Action computed in {elapsed:.1f}s: {result}")

        if not result:
            print("[AI] No action returned, skipping")
            return

        placements = result["placements"]
        discard = result["discard"]

        # Apply AI's placement
        if not room.game.apply_placement(ai_seat, placements, discard):
            print("[AI] Failed to apply placement")
            return

        # Log AI turn
        room.log_turn(ai_seat, placements, discard,
                      board_before=board_before,
                      opp_board_before=opp_board_before)

        # Send AI's board to human
        await room.send_to_seat(human_seat, {
            "type": "opponent_placed",
            "opponent_board": room.game.boards[ai_seat]
        })

        print(f"[AI] After AI place: placed_this_turn={room.game.placed_this_turn}")

        # Check if turn complete (human may have already placed)
        if room.game.is_turn_complete():
            print(f"[AI] Turn complete! turn={room.game.turn}")
            if room.game.is_hand_complete():
                # Hand complete - score it
                result_scores = calculate_scores(room.game)

                try:
                    db_logger.log_hand_end(
                        room.game.hand_id, room.game.chips,
                        result_scores["raw_score"], result_scores["actual_score"], result_scores)
                except Exception as e:
                    print(f"[WARN] DB log hand end: {e}")

                await room.broadcast({
                    "type": "hand_end",
                    "result": result_scores
                })

                end_check = check_session_end(room.game)
                if end_check:
                    await room.broadcast({"type": "session_end", **end_check})
                else:
                    room.pending_next_hand = True
                    room.next_hand_votes = set()
            else:
                # Deal next turn
                room.game.deal_turn()
                for s in [0, 1]:
                    opp = 1 - s
                    opp_board = room.game.boards[opp] if not room.game.is_fantasyland[opp] else {"top": [], "middle": [], "bottom": []}
                    await room.send_to_seat(s, {
                        "type": "deal",
                        "turn": room.game.turn,
                        "cards": room.game.dealt_cards[s],
                        "opponent_board": opp_board
                    })

    # Start next AI turn outside the lock (recursive)
    if room.is_ai_game and room.game and not room.game.placed_this_turn[room.ai_seat]:
        if room.game.dealt_cards.get(room.ai_seat):
            asyncio.create_task(ai_auto_play(room))


async def handle_message(room: GameRoom, player_id: str, seat: int, data: dict):
    msg_type = data.get("type")

    if msg_type == "game_start":
        room.start_votes.add(player_id)
        # AI auto-votes
        if room.is_ai_game:
            room.start_votes.add(AI_PLAYER_ID)

        if len(room.start_votes) >= 2:
            room.start_votes.clear()
            room.game = GameState(room.room_id, room.players)
            hand_info = room.game.start_hand()

            # Log session and hand start to DB
            try:
                db_logger.log_session_start(
                    room.game.session_id, room.room_id,
                    room.players[0], room.players[1] if len(room.players) > 1 else "")
                db_logger.log_hand_start(
                    room.game.hand_id, room.game.session_id,
                    room.game.hands_played, room.game.btn, room.game.chips)
            except Exception as e:
                print(f"[WARN] DB log session/hand start: {e}")

            await room.broadcast({
                "type": "session_start",
                "session_id": room.game.session_id,
                "chips": room.game.chips
            })

            # Send hand start to each player with their cards
            for s in [0, 1]:
                await room.send_to_seat(s, {
                    "type": "deal",
                    "turn": 0,
                    "cards": room.game.dealt_cards[s],
                    "your_seat": s,
                    "btn": room.game.btn,
                    "opponent_board": room.game.boards[1-s]
                })

            # AI auto-play initial turn (non-blocking)
            if room.is_ai_game and room.game:
                asyncio.create_task(ai_auto_play(room))
        else:
            await room.broadcast({
                "type": "waiting_for_start",
                "votes": len(room.start_votes)
            })

    elif msg_type == "place":
        if not room.game:
            return

        placements = data.get("placements", [])
        discard = data.get("discard")

        # Capture pre-action board state (deep copy)
        board_before = {
            k: list(v) for k, v in room.game.boards[seat].items()
        }
        opp_board_before = {
            k: list(v) for k, v in room.game.boards[1-seat].items()
        }

        # Apply placement
        if room.game.apply_placement(seat, placements, discard):
            # Log with pre-action board state
            room.log_turn(seat, placements, discard,
                          board_before=board_before,
                          opp_board_before=opp_board_before)

            # Notify opponent (hide FL board)
            opponent_seat = 1 - seat
            if room.game.is_fantasyland[seat]:
                pass
            else:
                new_board = room.game.boards[seat]
                await room.send_to_seat(opponent_seat, {
                    "type": "opponent_placed",
                    "opponent_board": new_board
                })

            # Check if turn complete
            print(f"[DEBUG] Seat {seat} placed. placed_this_turn: {room.game.placed_this_turn}")
            if room.game.is_turn_complete():
                print(f"[DEBUG] Turn complete! turn={room.game.turn}, checking hand complete...")
                if room.game.is_hand_complete():
                    # Hand complete - score it
                    result = calculate_scores(room.game)

                    try:
                        db_logger.log_hand_end(
                            room.game.hand_id, room.game.chips,
                            result["raw_score"], result["actual_score"], result)
                    except Exception as e:
                        print(f"[WARN] DB log hand end: {e}")

                    await room.broadcast({
                        "type": "hand_end",
                        "result": result
                    })

                    end_check = check_session_end(room.game)
                    if end_check:
                        await room.broadcast({
                            "type": "session_end",
                            **end_check
                        })
                    else:
                        room.pending_next_hand = True
                        room.next_hand_votes = set()
                else:
                    # Deal next turn
                    room.game.deal_turn()
                    for s in [0, 1]:
                        opp = 1 - s
                        opp_board = room.game.boards[opp] if not room.game.is_fantasyland[opp] else {"top": [], "middle": [], "bottom": []}
                        await room.send_to_seat(s, {
                            "type": "deal",
                            "turn": room.game.turn,
                            "cards": room.game.dealt_cards[s],
                            "opponent_board": opp_board
                        })

                    # AI auto-play next turn (non-blocking)
                    if room.is_ai_game and room.game:
                        asyncio.create_task(ai_auto_play(room))
        else:
            await room.send_to_seat(seat, {"type": "error", "message": "Invalid placement"})

    elif msg_type == "next_hand":
        if not room.game or not room.pending_next_hand:
            return

        room.next_hand_votes.add(seat)
        # AI auto-votes
        if room.is_ai_game:
            room.next_hand_votes.add(room.ai_seat)

        if len(room.next_hand_votes) >= 2:
            room.pending_next_hand = False
            room.next_hand_votes = set()

            # Start next hand
            room.game.next_btn()
            hand_info = room.game.start_hand()

            # Log hand start to DB
            try:
                db_logger.log_hand_start(
                    room.game.hand_id, room.game.session_id,
                    room.game.hands_played, room.game.btn, room.game.chips)
            except Exception as e:
                print(f"[WARN] DB log hand start: {e}")

            # Auto-solve FL hands
            for s in [0, 1]:
                if room.game.is_fantasyland[s]:
                    # Pass opponent's board for vs_normal scoring (lines+scoop+royalty)
                    opp_seat = 1 - s
                    opp_board = room.game.boards[opp_seat] if not room.game.is_fantasyland[opp_seat] else None
                    fl_result = solve_fantasyland(room.game.dealt_cards[s], opponent_board=opp_board)
                    if fl_result:
                        print(f"[DEBUG] FL Solver result for seat {s}: {fl_result}")
                        # Validate solver output - check for bust
                        evaluated = evaluate_board_with_joker_constraint(
                            fl_result["top"],
                            fl_result["middle"],
                            fl_result["bottom"],
                        )
                        if evaluated["busted"]:
                            values = evaluated["values"]
                            print(
                                f"[WARN] FL Solver returned BUST for seat {s}! "
                                f"top={values['top']} mid={values['middle']} "
                                f"bot={values['bottom']}"
                            )
                            print(f"[WARN] Rejecting solver result, will use fallback")
                            fl_result = None

                    if fl_result:
                        room.game.boards[s] = {
                            "top": fl_result["top"],
                            "middle": fl_result["middle"],
                            "bottom": fl_result["bottom"]
                        }
                        room.game.placed_this_turn[s] = True
                        room.log_turn(s, [[c, "auto"] for c in fl_result["top"] + fl_result["middle"] + fl_result["bottom"]], None)
                    else:
                        print(f"[WARN] FL Solver failed for seat {s}, cards: {room.game.dealt_cards[s]}")

            # Check if both players are already done (both FL)
            if room.game.placed_this_turn[0] and room.game.placed_this_turn[1]:
                for s in [0, 1]:
                    await room.send_to_seat(s, {
                        "type": "fl_solved",
                        "board": room.game.boards[s],
                        "message": "FL auto-placed"
                    })
                # Score immediately
                result = calculate_scores(room.game)

                try:
                    db_logger.log_hand_end(
                        room.game.hand_id, room.game.chips,
                        result["raw_score"], result["actual_score"], result)
                except Exception as e:
                    print(f"[WARN] DB log hand end: {e}")

                await room.broadcast({
                    "type": "hand_end",
                    "result": result
                })
                end_check = check_session_end(room.game)
                if end_check:
                    await room.broadcast({"type": "session_end", **end_check})
                else:
                    room.pending_next_hand = True
                    room.next_hand_votes = set()
            else:
                for s in [0, 1]:
                    is_fl = room.game.is_fantasyland[s]
                    fl_cards = room.game.fl_card_count[s] if is_fl else 0

                    if is_fl and room.game.placed_this_turn[s]:
                        await room.send_to_seat(s, {
                            "type": "fl_solved",
                            "board": room.game.boards[s],
                            "message": "FL auto-placed"
                        })
                    else:
                        await room.send_to_seat(s, {
                            "type": "deal",
                            "turn": 0,
                            "cards": room.game.dealt_cards[s],
                            "your_seat": s,
                            "btn": room.game.btn,
                            "opponent_board": {"top": [], "middle": [], "bottom": []},
                            "is_fantasyland": is_fl,
                            "fl_card_count": fl_cards
                        })

                # AI auto-play initial turn of new hand (non-blocking)
                if room.is_ai_game and room.game:
                    asyncio.create_task(ai_auto_play(room))
