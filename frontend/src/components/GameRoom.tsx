import React, { useState, useEffect } from 'react';
import { Board } from './Board';
import { CardHand } from './CardHand';
import { useWebSocket } from '../hooks/useWebSocket';
import './GameRoom.css';

interface BoardState {
    top: string[];
    middle: string[];
    bottom: string[];
}

export const GameRoom: React.FC = () => {
    const { connected, lastMessage, sendMessage, connect } = useWebSocket();

    const [roomId, setRoomId] = useState('');
    const [joined, setJoined] = useState(false);
    const [seat, setSeat] = useState<number | null>(null);
    const [chips, setChips] = useState([200, 200]);
    const [turn, setTurn] = useState(0);
    const [hand, setHand] = useState<string[]>([]);
    const [myBoard, setMyBoard] = useState<BoardState>({ top: [], middle: [], bottom: [] });
    const [opponentBoard, setOpponentBoard] = useState<BoardState>({ top: [], middle: [], bottom: [] });
    const [selectedCard, setSelectedCard] = useState<string | null>(null);
    const [pendingPlacements, setPendingPlacements] = useState<[string, string][]>([]);
    const [pendingDiscard, setPendingDiscard] = useState<string | null>(null);
    const [status, setStatus] = useState('Waiting...');
    const [gameStarted, setGameStarted] = useState(false);
    const [isBtn, setIsBtn] = useState(false);
    const [waitingForOpponent, setWaitingForOpponent] = useState(false);
    const [handResult, setHandResult] = useState<any>(null);
    const [isAIGame, setIsAIGame] = useState(false);

    useEffect(() => {
        if (!lastMessage) return;

        console.log('Message:', lastMessage);

        switch (lastMessage.type) {
            case 'connected':
                setSeat(lastMessage.seat as number);
                setStatus(`Connected as Player ${(lastMessage.seat as number) + 1}`);
                break;

            case 'ready_to_start':
                setStatus('Both players connected! Click Start Game');
                break;

            case 'waiting_for_start':
                setStatus('Waiting for other player to start...');
                break;

            case 'session_start':
                setChips(lastMessage.chips as number[]);
                setGameStarted(true);
                setStatus('Game started!');
                break;

            case 'deal':
                setHand(lastMessage.cards as string[]);
                setTurn(lastMessage.turn as number);
                setPendingPlacements([]);
                setPendingDiscard(null);
                setWaitingForOpponent(false);
                setGameStarted(true);
                // Reset board on new hand (turn 0)
                if ((lastMessage.turn as number) === 0) {
                    setMyBoard({ top: [], middle: [], bottom: [] });
                }
                if (lastMessage.opponent_board) {
                    setOpponentBoard(lastMessage.opponent_board as BoardState);
                }
                if (lastMessage.btn !== undefined && lastMessage.your_seat !== undefined) {
                    setIsBtn((lastMessage.btn as number) === (lastMessage.your_seat as number));
                }

                // Check FL mode
                const isFL = lastMessage.is_fantasyland as boolean;
                const flCards = lastMessage.fl_card_count as number;
                if (isFL && flCards > 0) {
                    setStatus(`🎰 FANTASYLAND! (${flCards} cards) - Place 13 cards, discard ${flCards - 13}`);
                } else {
                    const cardsText = (lastMessage.turn as number) === 0 ? '5 cards' : '3 cards';
                    setStatus(`Turn ${lastMessage.turn}: Place your ${cardsText}`);
                }
                break;

            case 'opponent_placed':
                setOpponentBoard(lastMessage.opponent_board as BoardState);
                break;

            case 'hand_end':
                const result = lastMessage.result as any;
                setChips(result.chips);
                if (seat !== null) {
                    setMyBoard(result.boards[seat]);
                    setOpponentBoard(result.boards[1 - seat]);
                }
                setHandResult(result);
                setHand([]);
                setWaitingForOpponent(false);
                setStatus('ハンド終了');
                break;

            case 'fl_solved':
                // FL was auto-solved by Rust solver
                const solvedBoard = lastMessage.board as BoardState;
                setMyBoard(solvedBoard);
                setStatus(lastMessage.message as string || '🎰 FL自動配置完了！');
                setWaitingForOpponent(true);
                setHand([]);
                break;

            case 'session_end':
                const winner = lastMessage.winner as number;
                const isWinner = winner === seat;
                setStatus(isWinner ? '🎉 You Win!' : '😔 You Lose');
                setGameStarted(false);
                break;

            case 'error':
                setStatus(`Error: ${lastMessage.message}`);
                break;
        }
    }, [lastMessage, seat, turn]);

    const handleJoinRoom = () => {
        if (roomId.trim()) {
            connect(roomId.trim());
            setJoined(true);
        }
    };

    const handleCreateRoom = () => {
        const newRoomId = Math.random().toString(36).substring(2, 8);
        setRoomId(newRoomId);
        connect(newRoomId);
        setJoined(true);
    };

    const handleCreateAIRoom = async () => {
        try {
            const apiBase = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
                ? 'http://localhost:8080'
                : `${window.location.protocol}//${window.location.host}`;
            const res = await fetch(`${apiBase}/api/rooms?vs_ai=true`, { method: 'POST' });
            const data = await res.json();
            if (data.error) {
                setStatus(`Error: ${data.error}`);
                return;
            }
            setRoomId(data.room_id);
            setIsAIGame(true);
            connect(data.room_id);
            setJoined(true);
        } catch (e) {
            setStatus('Failed to create AI room');
            console.error(e);
        }
    };

    const handleStartGame = () => {
        sendMessage({ type: 'game_start' });
        setStatus('Waiting for both players...');
    };

    const handleDropCard = (position: string, droppedCard?: string) => {
        const card = droppedCard || selectedCard;
        if (!card || waitingForOpponent) return;

        const maxCards = position === 'top' ? 3 : 5;
        const currentCount = position === 'top' ? myBoard.top.length :
            position === 'middle' ? myBoard.middle.length : myBoard.bottom.length;

        if (currentCount >= maxCards) {
            setStatus(`${position} row is full!`);
            return;
        }

        // Update local board
        const newBoard = { ...myBoard };
        if (position === 'top') newBoard.top = [...newBoard.top, card];
        else if (position === 'middle') newBoard.middle = [...newBoard.middle, card];
        else newBoard.bottom = [...newBoard.bottom, card];

        setMyBoard(newBoard);
        setPendingPlacements(prev => [...prev, [card, position]]);
        setHand(prev => prev.filter(c => c !== card));
        setSelectedCard(null);
    };

    const handleDiscard = () => {
        if (!selectedCard || waitingForOpponent) return;
        setPendingDiscard(selectedCard);
        setHand(prev => prev.filter(c => c !== selectedCard));
        setSelectedCard(null);
    };

    const handleConfirm = () => {
        const expectedPlacements = turn === 0 ? 5 : 2;

        if (pendingPlacements.length !== expectedPlacements) {
            setStatus(`${expectedPlacements}枚配置してください (${pendingPlacements.length}/${expectedPlacements})`);
            return;
        }

        // Auto-discard: remaining card in hand becomes discard
        let discard = pendingDiscard;
        if (turn > 0 && !discard && hand.length === 1) {
            discard = hand[0];
        }

        console.log('[DEBUG] Sending place:', { placements: pendingPlacements, discard });
        sendMessage({
            type: 'place',
            placements: pendingPlacements,
            discard: discard || undefined
        });

        setWaitingForOpponent(true);
        setStatus('相手を待っています...');
    };

    const handleUndo = () => {
        if (waitingForOpponent) return;

        if (pendingPlacements.length > 0) {
            const [card, position] = pendingPlacements[pendingPlacements.length - 1];
            setHand(prev => [...prev, card]);

            const newBoard = { ...myBoard };
            if (position === 'top') newBoard.top = newBoard.top.slice(0, -1);
            else if (position === 'middle') newBoard.middle = newBoard.middle.slice(0, -1);
            else newBoard.bottom = newBoard.bottom.slice(0, -1);

            setMyBoard(newBoard);
            setPendingPlacements(prev => prev.slice(0, -1));
        } else if (pendingDiscard) {
            setHand(prev => [...prev, pendingDiscard]);
            setPendingDiscard(null);
        }
    };

    const handleNextHand = () => {
        sendMessage({ type: 'next_hand' });
        setHandResult(null);
        setStatus('次のハンドを待っています...');
    };

    // Helper to render result overlay
    const renderResultOverlay = () => {
        if (!handResult || seat === null) return null;
        const mySeat = seat;
        const oppSeat = 1 - seat;
        const lines = ['top', 'middle', 'bottom'] as const;
        const lineLabelsJa = { top: 'トップ', middle: 'ミドル', bottom: 'ボトム' };
        const myScore = handResult.raw_score[mySeat];
        const myFL = handResult.fl_entry?.[mySeat];
        const myFLCards = handResult.fl_card_count?.[mySeat] || 0;
        const oppFL = handResult.fl_entry?.[oppSeat];

        return (
            <div className="result-overlay">
                <div className="result-panel">
                    <h2 className="result-title">
                        {handResult.busted[mySeat] ? '💥 BUST' : myScore > 0 ? '🏆 勝ち' : myScore < 0 ? '😔 負け' : '🤝 引き分け'}
                    </h2>
                    <div className="result-score-big">
                        {myScore > 0 ? '+' : ''}{myScore} pts
                    </div>

                    <table className="result-table">
                        <thead>
                            <tr>
                                <th></th>
                                <th>あなた</th>
                                <th>勝敗</th>
                                <th>相手</th>
                            </tr>
                        </thead>
                        <tbody>
                            {lines.map((line, i) => {
                                const lr = handResult.line_results[i];
                                // Flip line_results for player perspective (results are P0 perspective)
                                const myLR = mySeat === 0 ? lr : -lr;
                                const myIcon = myLR === 1 ? '✅' : myLR === -1 ? '❌' : '➖';
                                return (
                                    <tr key={line}>
                                        <td className="line-label">{lineLabelsJa[line]}</td>
                                        <td className="hand-name">
                                            {handResult.busted[mySeat] ? '💥' : handResult.hand_names?.[mySeat]?.[line] || '---'}
                                            {!handResult.busted[mySeat] && handResult.royalties[mySeat][line] > 0 && (
                                                <span className="royalty-badge">+{handResult.royalties[mySeat][line]}</span>
                                            )}
                                        </td>
                                        <td className="line-result">{myIcon}</td>
                                        <td className="hand-name">
                                            {handResult.busted[oppSeat] ? '💥' : handResult.hand_names?.[oppSeat]?.[line] || '---'}
                                            {!handResult.busted[oppSeat] && handResult.royalties[oppSeat][line] > 0 && (
                                                <span className="royalty-badge">+{handResult.royalties[oppSeat][line]}</span>
                                            )}
                                        </td>
                                    </tr>
                                );
                            })}
                        </tbody>
                    </table>

                    {handResult.scoop && (
                        <div className="scoop-badge">🎯 SCOOP! +3</div>
                    )}

                    <div className="result-chips">
                        💰 Chips: {handResult.chips[mySeat]} vs {handResult.chips[oppSeat]}
                    </div>

                    {(myFL || oppFL) && (
                        <div className="fl-badge">
                            {myFL && <span>🎰 あなた → FL ({myFLCards}枚)</span>}
                            {oppFL && <span>🎰 相手 → FL</span>}
                        </div>
                    )}

                    <button onClick={handleNextHand} className="btn-primary btn-next-hand">
                        次のハンドへ ▶
                    </button>
                </div>
            </div>
        );
    };

    // Lobby
    if (!joined) {
        return (
            <div className="lobby">
                <h1>🃏 OFC Pineapple</h1>
                <div className="lobby-actions">
                    <button onClick={handleCreateAIRoom} className="btn-primary btn-ai">
                        🤖 vs AI
                    </button>
                    <div className="lobby-divider">── or ──</div>
                    <button onClick={handleCreateRoom} className="btn-secondary">
                        Create PvP Room
                    </button>
                    <div className="join-section">
                        <input
                            type="text"
                            placeholder="Enter Room ID"
                            value={roomId}
                            onChange={(e) => setRoomId(e.target.value)}
                            onKeyDown={(e) => e.key === 'Enter' && handleJoinRoom()}
                        />
                        <button onClick={handleJoinRoom} className="btn-secondary">
                            Join
                        </button>
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div className="game-room">
            <header className="game-header">
                <div className="room-info">
                    <span>Room: <strong>{roomId}</strong></span>
                    {seat !== null && <span className="seat-badge">{isBtn ? 'BTN' : 'BB'}</span>}
                </div>
                <div className="chips-display">
                    <span className={seat === 0 ? 'you' : ''}>P1: {chips[0]}</span>
                    <span className={seat === 1 ? 'you' : ''}>P2: {chips[1]}</span>
                </div>
                <div className="status">{status}</div>
            </header>

            <main className="game-area">
                <div className="opponent-section">
                    <h3>{isAIGame ? '🤖 AI' : 'Opponent'} {!isBtn && '(BTN)'}</h3>
                    <Board {...opponentBoard} isOpponent={true} />
                </div>

                <div className="my-section">
                    <h3>Your Board {isBtn && '(BTN)'}</h3>
                    <Board {...myBoard} onDropCard={handleDropCard} />
                </div>

                {hand.length > 0 && (
                    <CardHand
                        cards={hand}
                        selectedCard={selectedCard}
                        onCardSelect={setSelectedCard}
                    />
                )}

                <div className="controls">
                    {!gameStarted && connected && (
                        <button onClick={handleStartGame} className="btn-primary">
                            Start Game
                        </button>
                    )}
                    {gameStarted && (hand.length > 0 || pendingPlacements.length > 0) && !waitingForOpponent && (
                        <>
                            {turn > 0 && (
                                <button
                                    onClick={handleDiscard}
                                    disabled={!selectedCard}
                                    className="btn-discard"
                                >
                                    Discard {pendingDiscard && '✓'}
                                </button>
                            )}
                            <button onClick={handleUndo} className="btn-secondary">
                                Undo
                            </button>
                            <button onClick={handleConfirm} className="btn-primary">
                                Confirm ({pendingPlacements.length}/{turn === 0 ? 5 : 2})
                            </button>
                        </>
                    )}
                </div>

                {pendingDiscard && (
                    <div className="discard-pile">
                        <span className="discard-label">🗑 捨て札:</span>
                        <span className="discard-card">{pendingDiscard}</span>
                    </div>
                )}

                <div className="turn-info">
                    Turn {turn}/8 | 配置: {pendingPlacements.length} 枚
                    {waitingForOpponent && (isAIGame ? ' | 🤖 AI思考中...' : ' | ⏳ 相手を待っています...')}
                </div>
            </main>
            {renderResultOverlay()}
        </div>
    );
};

export default GameRoom;
