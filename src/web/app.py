from flask import Flask, render_template, request, jsonify, session
import json
import glob
import numpy as np
import random
import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.game import SimpleJong, Player, Tile, TileType, Discard, Tsumo, Suit, Pon, Chi, PassCall, CalledSet

# Create Flask app with correct template and static folder paths
template_dir = os.path.join(os.path.dirname(__file__), 'templates')
static_dir = os.path.join(os.path.dirname(__file__), 'static')
app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
app.secret_key = 'mahjong_secret_key_2024'

# In-memory debug session storage
debug_sessions = {}

def _get_session_id():
    if 'sid' not in session:
        session['sid'] = str(random.randint(10_000, 99_999))
    return session['sid']

class HumanPlayer(Player):
    """Human player that can be controlled via web interface"""
    
    def __init__(self, player_id: int):
        super().__init__(player_id)
        self.pending_action = None
    
    def play(self, game_state):
        # Human players will have their actions set via the web interface
        if self.pending_action:
            action = self.pending_action
            self.pending_action = None
            return action
        else:
            # Default to discarding a random tile if no action is set
            return Discard(random.choice(self.hand))
    
    def set_action(self, action):
        """Set the action for the human player"""
        self.pending_action = action

class AIPlayer(Player):
    """AI player that uses random strategy"""
    
    def play(self, game_state):
        if not self.hand:
            return Tsumo()
        
        # Check if we can win
        if self.can_win():
            return Tsumo()
        
        # Check if we can call (pon/chi) before discarding
        if game_state.can_call:
            possible_calls = self.can_call(game_state)
            all_calls = []
            
            # Collect all possible calls with priority (pon > chi)
            for tiles in possible_calls['pon']:
                all_calls.append(('pon', tiles))
            for tiles in possible_calls['chi']:
                all_calls.append(('chi', tiles))
            
            # 30% chance to make a call if possible
            if all_calls and random.random() < 0.3:
                call_type, tiles = random.choice(all_calls)
                if call_type == 'pon':
                    return Pon(tiles)
                else:
                    return Chi(tiles)
        
        # Discard a random tile
        return Discard(random.choice(self.hand))

class GameManager:
    """Manages the game state and handles web interface interactions"""
    
    def __init__(self):
        self.game = None
        self.players = None
        self.game_id = None
        self.newly_drawn_tile = None  # Deprecated: use game.last_drawn_tile
        self.win_type = None # Track win type (Ron or Tsumo)
        self.game_mode = None  # 'play' or 'watch'
    
    def start_new_game(self, mode='play'):
        """Start a new game with specified mode"""
        self.game_mode = mode
        
        if mode == 'play':
            # Play mode: 1 human player and 3 AI players
            self.players = [
                HumanPlayer(0),  # Human player
                AIPlayer(1),     # AI player 1
                AIPlayer(2),     # AI player 2
                AIPlayer(3)      # AI player 3
            ]
        elif mode == 'watch':
            # Watch mode: 4 AI players
            self.players = [
                AIPlayer(0),     # AI player 0
                AIPlayer(1),     # AI player 1
                AIPlayer(2),     # AI player 2
                AIPlayer(3)      # AI player 3
            ]
        else:
            raise ValueError(f"Invalid game mode: {mode}")
        
        self.game = SimpleJong(self.players)
        self.game_id = random.randint(1000, 9999)
        self.newly_drawn_tile = None
        self.win_type = None
        return self.game_id
    
    def start_watch_game(self):
        """Start a new watch game (4 AI players)"""
        return self.start_new_game(mode='watch')
    
    def start_play_game(self):
        """Start a new play game (1 human + 3 AI players)"""
        return self.start_new_game(mode='play')
    
    def _sort_hand(self, tiles):
        """Sort tiles by suit first (pinzu before souzu), then by number"""
        def tile_sort_key(tile):
            # Convert tile object to sort key: (suit_priority, tile_number)
            suit_priority = 0 if tile.suit == Suit.PINZU else 1
            return (suit_priority, tile.tile_type.value)
        
        return sorted(tiles, key=tile_sort_key)
    
    def get_game_state(self):
        """Get the current game state for the web interface"""
        if not self.game:
            return None
        
        # If it's the human player's turn and they haven't drawn a tile yet this turn,
        # draw one automatically (only in play mode)
        if (self.game_mode == 'play' and 
            self.game.current_player_idx == 0 and 
            len(self.players[0].hand) == 11 and 
            self.game.tiles):
            success, message = self.draw_tile_for_human()
            if not success:
                # If drawing fails, return error state
                return {
                    'error': message,
                    'game_over': True
                }
        
        # Get human player's hand - sorted by suit then number (only in play mode)
        human_hand = []
        if self.game_mode == 'play':
            human_hand = [str(tile) for tile in self._sort_hand(self.players[0].hand)]
        
        # Get discarded tiles computed from per-player discards
        discarded_tiles = []
        # Per-player discards from engine
        try:
            player_discards = [[str(t) for t in self.game.player_discards[i]] for i in range(4)]
            for i in range(4):
                discarded_tiles.extend(player_discards[i])
        except Exception:
            player_discards = [[], [], [], []]
            discarded_tiles = []
        
        # Get other players' hand sizes or winning hand
        other_hands = []
        if self.game.is_game_over() and self.game.get_winner() is not None:
            winner_id = self.game.get_winner()
            for i, player in enumerate(self.players):
                if self.game_mode == 'play' and i == 0:
                    continue  # Skip human player in play mode
                
                if i == winner_id:
                    # Show the sorted winning hand
                    other_hands.append([str(tile) for tile in self._sort_hand(player.hand)])
                else:
                    # Show hand size for other AI players
                    other_hands.append(len(player.hand))
        else:
            # If game is not over, show hand sizes
            for i, player in enumerate(self.players):
                if self.game_mode == 'play' and i == 0:
                    continue  # Skip human player in play mode
                other_hands.append(len(player.hand))
        
        # Get called sets for all players
        called_sets = {}
        for i, player in enumerate(self.players):
            called_sets[i] = []
            for called_set in player.called_sets:
                called_sets[i].append({
                    'tiles': [str(tile) for tile in called_set.tiles],
                    'call_type': called_set.call_type,
                    'called_tile': str(called_set.called_tile),
                    'source_position': called_set.source_position
                })
        
        # Get possible actions for the human player (only in play mode)
        possible_actions = {'pon': [], 'chi': [], 'ron': [], 'tsumo': []}
        if not self.game.is_game_over() and self.game_mode == 'play':
            game_state_for_human = self.game.get_turn_snapshot(0)
            possible_actions = self.players[0].get_possible_actions(game_state_for_human)
            
            # Only show tsumo on human's turn, not on other players' discards
            if self.game.current_player_idx != 0:
                possible_actions['tsumo'] = []

        return {
            'game_id': self.game_id,
            'game_mode': self.game_mode,
            'current_player': self.game.current_player_idx,
            'human_hand': human_hand,
            'discarded_tiles': discarded_tiles,
            'player_discards': player_discards,
            'other_hands': other_hands,
            'remaining_tiles': self.game.get_remaining_tiles(),
            'game_over': self.game.is_game_over(),
            'winner': self.game.get_winner(),
            'win_type': self.win_type,
            'is_human_turn': self.game_mode == 'play' and self.game.current_player_idx == 0,
            'newly_drawn_tile': str(self.game.last_drawn_tile) if getattr(self.game, 'last_drawn_tile', None) else None,
            'called_sets': called_sets,
            'possible_actions': possible_actions,
            'last_discarded_tile': str(self.game.last_discarded_tile) if self.game.last_discarded_tile else None
        }
    
    def draw_tile_for_human(self):
        """Draw a tile for the human player if it's their turn"""
        if not self.game or self.game.current_player_idx != 0:
            return False, "Not human player's turn"
        
        # Check if there are tiles to draw
        if not self.game.tiles:
            return False, "No tiles remaining"
        
        # Draw a tile
        new_tile = self.game.tiles.pop()
        self.players[0].add_tile(new_tile)
        # Track newly drawn tile in engine
        try:
            self.game.last_drawn_tile = new_tile
            self.game.last_drawn_player = 0
        except Exception:
            pass
        return True, f"Drew tile {new_tile}"

    def human_discard(self, tile_str):
        """Handle human player discarding a tile"""
        if not self.game or self.game.current_player_idx != 0:
            return False, "Not human player's turn"
        
        # Convert tile string back to Tile object
        try:
            suit_char = tile_str[-1]
            tile_value = int(tile_str[:-1])

            suit = Suit(suit_char)
            tile_type = TileType(tile_value)
            tile = Tile(suit, tile_type)
        except (ValueError, KeyError):
            return False, "Invalid tile"
        
        # Check if player has this tile
        if tile not in self.players[0].hand:
            return False, "Tile not in hand"
        
        # Remove the tile from hand and add to per-player discarded tiles
        self.players[0].remove_tile(tile)
        # Track discard in engine per-player discards
        try:
            self.game.player_discards[0].append(tile)
        except Exception:
            pass
        self.game.last_discarded_tile = tile
        self.game.last_discard_player = 0
        
        # Clear the newly drawn tile since we're discarding
        try:
            self.game.last_drawn_tile = None
            self.game.last_drawn_player = None
        except Exception:
            pass
        
        # Check if another player can win with the discarded tile (Ron)
        for i, player in enumerate(self.players):
            if i != 0:  # Check all other players
                if player.can_ron(tile):
                    self.game.winner = i
                    self.game.game_over = True
                    self.win_type = 'Ron'
                    # Add the winning tile to the AI's hand to show it
                    player.add_tile(tile)
                    return True, f"Game over! AI Player {i} won with Ron!"
        
        # Don't move to next player yet - let the frontend check for calls first
        # The turn will advance when the frontend calls /api/play_ai_turn or /api/action
        return True, "Tile discarded"
    
    def human_tsumo(self):
        """Handle human player declaring tsumo"""
        if not self.game or self.game.current_player_idx != 0:
            return False, "Not human player's turn"
        
        # Check if player can actually win
        if not self.players[0].can_win():
            return False, "Cannot declare tsumo - no winning hand"
        
        # Player wins!
        self.game.winner = 0
        self.game.game_over = True
        self.win_type = 'Tsumo'
        return True, "Tsumo! You won!"
    
    def play_ai_turn(self):
        """Play the current AI player's turn"""
        if not self.game or self.game.is_game_over():
            return False
        
        # If there's a discarded tile from any player, all other AI players should get a chance to act on it
        if self.game.last_discarded_tile and self.game.last_discard_player is not None:
            # Check all other AI players for possible actions, starting from the next player after the discarder
            discarder_id = self.game.last_discard_player
            for player_offset in range(1, 4):  # Check the next 3 players in turn order
                ai_player_idx = (discarder_id - player_offset) % 4  # Counterclockwise order
                if ai_player_idx == discarder_id:
                    continue  # Skip the player who discarded
                
                ai_player = self.players[ai_player_idx]
                game_state_for_ai = self.game.get_turn_snapshot(ai_player_idx)
                possible_actions = ai_player.get_possible_actions(game_state_for_ai)

                # AI decision logic: Ron > Pon > Chi (30% chance to call)
                if possible_actions.get('ron'):
                    self.game.winner = ai_player_idx
                    self.game.game_over = True
                    self.win_type = 'Ron'
                    ai_player.add_tile(self.game.last_discarded_tile)
                    return True

                # 30% chance to make a call
                if random.random() < 0.3:
                    if possible_actions.get('pon'):
                        # Convert tile strings back to Tile objects
                        pon_tiles = [Tile(Suit(t[-1]), TileType(int(t[:-1]))) for t in possible_actions['pon'][0]]
                        self.game.make_call(ai_player_idx, 'pon', pon_tiles)
                        # Clear last discard since it was called
                        self.game.last_discarded_tile = None
                        self.game.last_discard_player = None
                        # The calling player becomes the current player and must discard
                        self.game.current_player_idx = ai_player_idx
                        return True
                    elif possible_actions.get('chi'):
                        # Convert tile strings back to Tile objects
                        chi_tiles = [Tile(Suit(t[-1]), TileType(int(t[:-1]))) for t in possible_actions['chi'][0]]
                        self.game.make_call(ai_player_idx, 'chi', chi_tiles)
                        # Clear last discard since it was called
                        self.game.last_discarded_tile = None
                        self.game.last_discard_player = None
                        # The calling player becomes the current player and must discard
                        self.game.current_player_idx = ai_player_idx
                        return True
            
            # No AI player called, so move to next player after the discarder
            self.game.current_player_idx = (discarder_id - 1) % 4  # Counterclockwise order
            self.game.last_discarded_tile = None
            self.game.last_discard_player = None

        # Standard AI turn: draw and discard
        current_player_idx = self.game.current_player_idx
        current_player = self.players[current_player_idx]
        
        if self.game.tiles:
            new_tile = self.game.tiles.pop()
            current_player.add_tile(new_tile)
        else:
            self.game.game_over = True
            return False

        # Check for tsumo
        if current_player.can_tsumo():
            self.game.winner = current_player_idx
            self.game.game_over = True
            self.win_type = 'Tsumo'
            return True

        # AI discards a random tile
        discarded_tile = random.choice(current_player.hand)
        current_player.remove_tile(discarded_tile)
        try:
            self.game.player_discards[current_player_idx].append(discarded_tile)
        except Exception:
            pass
        self.game.last_discarded_tile = discarded_tile
        self.game.last_discard_player = current_player_idx

        # Check if another player can ron on this discard
        for i, p in enumerate(self.players):
            if i != current_player_idx and p.can_ron(discarded_tile):
                if self.game_mode == 'play' and i == 0:
                    # Human can ron, so we stop and wait for human action
                    return True
                elif self.game_mode == 'watch':
                    # AI vs AI ron - handle immediately
                    self.game.winner = i
                    self.game.game_over = True
                    self.win_type = 'Ron'
                    p.add_tile(discarded_tile)
                    return True
        
        # Move to next player (counterclockwise: 3→2→1→0→3)
        self.game.current_player_idx = (self.game.current_player_idx - 1) % 4
        
        return True

# Global game manager
game_manager = GameManager()

@app.route('/')
def index():
    """Main game page"""
    return render_template('index.html')

@app.route('/debug')
def debug_viewer():
    """Debug viewer page for stepping through generated training data (.npz)."""
    _get_session_id()
    return render_template('debug.html')

@app.route('/api/debug/list_files')
def debug_list_files():
    """List available .npz files under training_data/**/data."""
    base_dir = os.path.join(os.path.dirname(__file__), '..', '..')
    base_dir = os.path.abspath(base_dir)
    npz_paths = glob.glob(os.path.join(base_dir, 'training_data', '**', 'data', '*.npz'), recursive=True)
    # Return paths relative to project root
    project_root = base_dir
    files = [os.path.relpath(p, project_root) for p in npz_paths]
    return jsonify({'files': sorted(files)})

@app.route('/api/debug/load_file', methods=['POST'])
def debug_load_file():
    """Load a selected .npz file into memory for this session and build indices."""
    data = request.get_json(force=True)
    rel_path = data.get('path')
    if not rel_path:
        return jsonify({'error': 'path is required'}), 400
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    abs_path = os.path.abspath(os.path.join(project_root, rel_path))
    if not abs_path.startswith(project_root) or not os.path.exists(abs_path):
        return jsonify({'error': 'invalid path'}), 400

    try:
        npz = np.load(abs_path, allow_pickle=True)
        features = npz['features']
        policies = npz['policies']
        values = npz['values']
        game_ids = npz['game_ids'] if 'game_ids' in npz.files else np.zeros((features.shape[0],), dtype=np.int32)
        step_ids = npz['step_ids'] if 'step_ids' in npz.files else np.arange(features.shape[0], dtype=np.int32)
        states = npz['states'] if 'states' in npz.files else np.array([], dtype=object)

        # Build mapping from game_id -> ordered list of indices by step_id
        indices_by_game = {}
        for idx, gid in enumerate(game_ids.tolist()):
            indices_by_game.setdefault(gid, []).append((int(step_ids[idx]), idx))
        # Sort steps
        for gid in indices_by_game:
            indices_by_game[gid].sort(key=lambda x: x[0])
            indices_by_game[gid] = [i for _, i in indices_by_game[gid]]

        sid = _get_session_id()
        debug_sessions[sid] = {
            'path': abs_path,
            'features_shape': tuple(features.shape),
            'policies': policies,
            'values': values,
            'states': states,
            'game_ids': game_ids,
            'step_ids': step_ids,
            'indices_by_game': indices_by_game,
            'unique_games': sorted(indices_by_game.keys()),
        }
        total_games = len(debug_sessions[sid]['unique_games'])
        return jsonify({'ok': True, 'total_games': total_games, 'features_shape': debug_sessions[sid]['features_shape']})
    except Exception as e:
        return jsonify({'error': f'failed to load file: {e}'}), 500

@app.route('/api/debug/get_state')
def debug_get_state():
    """Return state, policy, and value for a given game and step."""
    sid = _get_session_id()
    if sid not in debug_sessions:
        return jsonify({'error': 'no file loaded'}), 400
    sess = debug_sessions[sid]
    try:
        game_index = int(request.args.get('game', 0))
        step_index = int(request.args.get('step', 0))
    except Exception:
        return jsonify({'error': 'invalid indices'}), 400

    unique_games = sess['unique_games']
    if not unique_games:
        return jsonify({'error': 'no games available'}), 400
    game_index = max(0, min(game_index, len(unique_games) - 1))
    gid = unique_games[game_index]
    step_indices = sess['indices_by_game'].get(gid, [])
    if not step_indices:
        return jsonify({'error': 'no steps for game'}), 400
    step_index = max(0, min(step_index, len(step_indices) - 1))
    sample_idx = step_indices[step_index]

    # Extract stored policy/value and state
    policy = sess['policies'][sample_idx].tolist()
    value = float(sess['values'][sample_idx])
    state_json = None
    state_obj = {}
    if sess['states'].size > 0:
        try:
            raw = sess['states'][sample_idx]
            # Normalize possible numpy scalar types to Python primitives
            try:
                import numpy as _np  # local import to avoid polluting module namespace
                np_bytes_types = (bytes, bytearray, getattr(_np, 'bytes_', bytes))
                np_str_types = (str, getattr(_np, 'str_', str))
            except Exception:
                np_bytes_types = (bytes, bytearray)
                np_str_types = (str,)

            if isinstance(raw, np_bytes_types):
                state_json = raw.decode('utf-8')
            elif isinstance(raw, np_str_types):
                # Ensure we have a native Python str
                state_json = str(raw)
            else:
                # Some objects may wrap the bytes in an object scalar
                try:
                    item = raw.item() if hasattr(raw, 'item') else raw
                    if isinstance(item, np_bytes_types):
                        state_json = item.decode('utf-8')
                    else:
                        state_json = str(item)
                except Exception:
                    state_json = str(raw)

            # Handle stray representations like b'...'
            if isinstance(state_json, str) and len(state_json) > 2 and state_json.startswith("b'") and state_json.endswith("'"):
                state_json = state_json[2:-1]

            state_obj = json.loads(state_json)
        except Exception:
            state_json = None
            state_obj = {}

    # Normalize state for frontend renderer
    normalized = {
        'game_id': int(gid),
        'current_player': state_obj.get('player_id', 0),
        'human_hand': state_obj.get('player_hand', []),
        'player_discards': state_obj.get('player_discards', {0: [], 1: [], 2: [], 3: []}),
        'remaining_tiles': state_obj.get('remaining_tiles', 0),
        'last_discarded_tile': state_obj.get('last_discarded_tile'),
        'called_sets': state_obj.get('called_sets', {}),
        'game_mode': 'watch',
        'game_over': False,
        'winner': None,
        'win_type': None,
    }
    # Ensure player_discards is a list-of-lists for 4 players
    if isinstance(normalized['player_discards'], dict):
        pdis = normalized['player_discards']
        normalized['player_discards'] = [pdis.get(i, []) for i in range(4)]

    return jsonify({
        'ok': True,
        'game_index': game_index,
        'step_index': step_index,
        'total_steps': len(step_indices),
        'total_games': len(unique_games),
        'state': normalized,
        'policy': policy,
        'value': value,
    })

@app.route('/api/new_game', methods=['POST'])
def new_game():
    """Start a new game (defaults to play mode for backward compatibility)"""
    game_id = game_manager.start_new_game(mode='play')
    return jsonify({'game_id': game_id, 'message': 'New game started', 'mode': 'play'})

@app.route('/api/play_game', methods=['POST'])
def play_game():
    """Start a new play game (1 human + 3 AI players)"""
    game_id = game_manager.start_play_game()
    return jsonify({'game_id': game_id, 'message': 'New play game started', 'mode': 'play'})

@app.route('/api/watch_game', methods=['POST'])
def watch_game():
    """Start a new watch game (4 AI players)"""
    game_id = game_manager.start_watch_game()
    return jsonify({'game_id': game_id, 'message': 'New watch game started', 'mode': 'watch'})

@app.route('/api/game_state')
def get_game_state():
    """Get current game state"""
    state = game_manager.get_game_state()
    if state is None:
        return jsonify({'error': 'No active game'}), 400
    return jsonify(state)

@app.route('/api/discard', methods=['POST'])
def discard_tile():
    """Handle human player discarding a tile"""
    data = request.get_json()
    tile_str = data.get('tile')
    
    if not tile_str:
        return jsonify({'error': 'No tile specified'}), 400
    
    success, message = game_manager.human_discard(tile_str)
    
    return jsonify({'success': success, 'message': message, 'ai_turn_needed': success})

@app.route('/api/play_ai_turn', methods=['POST'])
def play_ai_turn():
    """Play one AI turn"""
    if not game_manager.game or game_manager.game.is_game_over():
        return jsonify({'success': False, 'game_over': True})
    
    # If it's human's turn and there's no last discarded tile to respond to, don't play AI turn (only in play mode)
    if (game_manager.game_mode == 'play' and 
        game_manager.game.current_player_idx == 0 and 
        not game_manager.game.last_discarded_tile):
        return jsonify({'success': False, 'human_turn': True})
    
    # Play AI turn (this handles both responding to human discards and regular AI turns)
    success = game_manager.play_ai_turn()
    
    return jsonify({
        'success': success, 
        'human_turn': game_manager.game_mode == 'play' and game_manager.game.current_player_idx == 0, 
        'game_over': game_manager.game.is_game_over(),
        'continue_ai': game_manager.game.current_player_idx != 0 and not game_manager.game.is_game_over()
    })

@app.route('/api/tsumo', methods=['POST'])
def declare_tsumo():
    """Handle human player declaring tsumo"""
    success, message = game_manager.human_tsumo()
    return jsonify({'success': success, 'message': message})

@app.route('/api/action', methods=['POST'])
def player_action():
    """Handle player actions like pon, chi, ron"""
    data = request.get_json()
    action_type = data.get('action_type')
    
    if not game_manager.game:
        return jsonify({'error': 'No active game'}), 400

    human_player = game_manager.players[0]
    
    if action_type == 'pass':
        # Clear the last discard and move to next AI player
        last_discard_player = game_manager.game.last_discard_player
        game_manager.game.last_discarded_tile = None
        game_manager.game.last_discard_player = None
        # Move to next player after the one who discarded (counterclockwise)
        if last_discard_player is not None:
            game_manager.game.current_player_idx = (last_discard_player - 1) % 4
        return jsonify({'success': True, 'message': 'Passed', 'ai_turn_needed': True})

    if action_type == 'ron':
        game_manager.game.winner = 0
        game_manager.game.game_over = True
        game_manager.win_type = 'Ron'
        human_player.add_tile(game_manager.game.last_discarded_tile)
        return jsonify({'success': True, 'message': 'Ron! You won!'})

    tiles_str = data.get('tiles')
    if not tiles_str or not tiles_str[0]:
        return jsonify({'error': 'Tiles not provided for action'}), 400
        
    # Convert tile strings back to Tile objects
    tiles = [Tile(Suit(t[-1]), TileType(int(t[:-1]))) for t in tiles_str[0]]
    
    # Make the call
    game_manager.game.make_call(0, action_type, tiles)
    
    # Clear the last discard since it was called
    game_manager.game.last_discarded_tile = None
    game_manager.game.last_discard_player = None
    
    # Human player now needs to discard
    game_manager.game.current_player_idx = 0
    
    return jsonify({'success': True, 'message': f'{action_type.capitalize()} successful', 'ai_turn_needed': False})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
