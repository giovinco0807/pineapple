"""Flask application for OFC Pineapple."""
from flask import Flask, render_template, jsonify, request, session
from game import GameState, GamePhase, Card, Row
from ai import RandomAI
import os

app = Flask(__name__, static_folder='static', template_folder='static')
app.secret_key = os.urandom(24)

# Store game states (in production, use a proper session store)
games = {}
ai_player = RandomAI()


def get_game() -> GameState:
    """Get or create game state for current session."""
    game_id = session.get('game_id')
    if game_id and game_id in games:
        return games[game_id]
    
    # Create new game
    game = GameState()
    game_id = os.urandom(8).hex()
    session['game_id'] = game_id
    games[game_id] = game
    return game


@app.route('/')
def index():
    """Serve the main game page."""
    return render_template('index.html')


@app.route('/api/new_game', methods=['POST'])
def new_game():
    """Start a new game."""
    game = get_game()
    game.start_new_round()
    
    # Process AI turn and advance game
    _advance_game(game)
    
    return jsonify(game.get_state_for_player('player'))


@app.route('/api/state', methods=['GET'])
def get_state():
    """Get current game state."""
    game = get_game()
    return jsonify(game.get_state_for_player('player'))


@app.route('/api/place_cards', methods=['POST'])
def place_cards():
    """Place cards on the board."""
    game = get_game()
    data = request.json
    
    placements = []
    for p in data.get('placements', []):
        card = Card.from_dict(p['card'])
        row = Row(p['row'])
        placements.append((card, row))
    
    success = game.place_cards('player', placements)
    if not success:
        return jsonify({'error': 'Invalid placement'}), 400
    
    # Check for discard in pineapple phase
    if 'discard' in data and data['discard']:
        discard_card = Card.from_dict(data['discard'])
        game.discard_card('player', discard_card)
    
    # Handle Fantasyland discards
    if 'fl_discards' in data and data['fl_discards']:
        player_state = game.players['player']
        for d in data['fl_discards']:
            discard_card = Card.from_dict(d)
            if discard_card in player_state.hand:
                player_state.hand.remove(discard_card)
    
    # Process game flow
    _advance_game(game)
    
    return jsonify(game.get_state_for_player('player'))


@app.route('/api/next_round', methods=['POST'])
def next_round():
    """Start the next round."""
    game = get_game()
    if game.phase != GamePhase.ROUND_END:
        return jsonify({'error': 'Cannot start next round yet'}), 400
    
    game.start_new_round()
    _advance_game(game)
    
    return jsonify(game.get_state_for_player('player'))


@app.route('/api/end_session', methods=['POST'])
def end_session():
    """End the current session."""
    game = get_game()
    if not game.can_end_session():
        return jsonify({'error': 'Cannot end session yet'}), 400
    
    game.end_session()
    return jsonify(game.get_state_for_player('player'))


def _advance_game(game: GameState):
    """Advance game state, processing AI turns and phase transitions."""
    # Loop until we need player input or round ends
    for _ in range(20):  # Safety limit
        # Check if phase complete
        phase_changed = game.check_phase_complete()
        
        # Handle scoring phase
        if game.phase == GamePhase.SCORING:
            game.calculate_scores()
            game.finalize_round()
            return
        
        # Handle round end
        if game.phase == GamePhase.ROUND_END:
            return
        
        # Process AI turn if AI has cards
        ai_state = game.players['ai']
        if ai_state.hand:
            _process_ai_turn(game)
        elif not phase_changed:
            # No more changes, wait for player
            return


def _process_ai_turn(game: GameState):
    """Process AI's turn."""
    ai_state = game.players['ai']
    
    if not ai_state.hand:
        return
    
    if ai_state.in_fantasyland:
        # Fantasyland: place all cards at once
        placements = ai_player.place_fantasyland_cards(ai_state.hand, ai_state.board)
        game.place_cards('ai', placements)
    elif game.phase == GamePhase.INITIAL_DEAL:
        # Initial deal: place all 5 cards
        placements = ai_player.place_initial_cards(ai_state.hand, ai_state.board)
        game.place_cards('ai', placements)
    elif game.phase == GamePhase.PINEAPPLE:
        # Pineapple: place 2, discard 1
        placements, discard = ai_player.place_pineapple_cards(ai_state.hand, ai_state.board)
        game.place_cards('ai', placements)
        game.discard_card('ai', discard)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
