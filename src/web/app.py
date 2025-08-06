from flask import Flask, render_template, request, jsonify, session
import random
import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.game import SimpleJong, Player, Tile, TileType, Discard, Tsumo

# Create Flask app with correct template and static folder paths
template_dir = os.path.join(os.path.dirname(__file__), 'templates')
static_dir = os.path.join(os.path.dirname(__file__), 'static')
app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
app.secret_key = 'mahjong_secret_key_2024'

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
        
        # Discard a random tile
        return Discard(random.choice(self.hand))

class GameManager:
    """Manages the game state and handles web interface interactions"""
    
    def __init__(self):
        self.game = None
        self.players = None
        self.game_id = None
        self.newly_drawn_tile = None  # Track the newly drawn tile
        self.player_discards = [[], [], [], []]  # Track discards by player
        self.win_type = None # Track win type (Ron or Tsumo)
    
    def start_new_game(self):
        """Start a new game with 1 human player and 3 AI players"""
        self.players = [
            HumanPlayer(0),  # Human player
            AIPlayer(1),     # AI player 1
            AIPlayer(2),     # AI player 2
            AIPlayer(3)      # AI player 3
        ]
        self.game = SimpleJong(self.players)
        self.game_id = random.randint(1000, 9999)
        self.newly_drawn_tile = None
        self.player_discards = [[], [], [], []]
        self.win_type = None
        return self.game_id
    
    def get_game_state(self):
        """Get the current game state for the web interface"""
        if not self.game:
            return None
        
        # If it's the human player's turn and they haven't drawn a tile yet this turn,
        # draw one automatically
        if (self.game.current_player_idx == 0 and 
            len(self.players[0].hand) == 11 and 
            self.game.tiles):
            success, message = self.draw_tile_for_human()
            if not success:
                # If drawing fails, return error state
                return {
                    'error': message,
                    'game_over': True
                }
        
        # Get human player's hand
        human_hand = sorted([str(tile) for tile in self.players[0].hand])
        
        # Get discarded tiles by player
        discarded_tiles = [str(tile) for tile in self.game.discarded_tiles]
        
        # Get other players' hand sizes or winning hand
        other_hands = []
        if self.game.is_game_over() and self.game.get_winner() is not None and self.game.get_winner() != 0:
            winner_id = self.game.get_winner()
            for i, player in enumerate(self.players[1:], start=1):
                if i == winner_id:
                    # Show the sorted winning hand
                    other_hands.append(sorted([str(tile) for tile in player.hand]))
                else:
                    # Show hand size for other AI players
                    other_hands.append(len(player.hand))
        else:
            # If game is not over, show hand sizes
            other_hands = [len(player.hand) for player in self.players[1:]]

        return {
            'game_id': self.game_id,
            'current_player': self.game.current_player_idx,
            'human_hand': human_hand,
            'discarded_tiles': discarded_tiles,
            'player_discards': [discards for discards in self.player_discards],
            'other_hands': other_hands,
            'remaining_tiles': self.game.get_remaining_tiles(),
            'game_over': self.game.is_game_over(),
            'winner': self.game.get_winner(),
            'win_type': self.win_type,
            'is_human_turn': self.game.current_player_idx == 0,
            'newly_drawn_tile': self.newly_drawn_tile
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
        self.newly_drawn_tile = str(new_tile)  # Track the newly drawn tile
        return True, f"Drew tile {new_tile}"

    def human_discard(self, tile_str):
        """Handle human player discarding a tile"""
        if not self.game or self.game.current_player_idx != 0:
            return False, "Not human player's turn"
        
        # Convert tile string back to Tile object
        try:
            tile_value = int(tile_str.replace('p', ''))
            tile_type = TileType(tile_value)
            tile = Tile(tile_type)
        except (ValueError, KeyError):
            return False, "Invalid tile"
        
        # Check if player has this tile
        if tile not in self.players[0].hand:
            return False, "Tile not in hand"
        
        # Remove the tile from hand and add to discarded tiles
        self.players[0].remove_tile(tile)
        self.game.discarded_tiles.append(tile)
        self.player_discards[0].append(str(tile)) # Track the discard
        
        # Clear the newly drawn tile since we're discarding
        self.newly_drawn_tile = None
        
        # Check if another player can win with the discarded tile (Ron)
        for i, player in enumerate(self.players):
            if i != 0:  # Check all other players
                # Temporarily add tile to check if they can win
                player.add_tile(tile)
                can_win = player.can_win()
                player.remove_tile(tile)  # Remove it immediately after checking
                
                if can_win:
                    self.game.winner = i
                    self.game.game_over = True
                    self.win_type = 'Ron'
                    return True, f"Game over! AI Player {i} won with Ron!"
        
        # Move to next player (counterclockwise: 0→3→2→1→0)
        self.game.current_player_idx = (self.game.current_player_idx - 1) % 4
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
        
        current_player = self.players[self.game.current_player_idx]
        
        # Draw a tile
        if self.game.tiles:
            new_tile = self.game.tiles.pop()
            current_player.add_tile(new_tile)
        
        # Get game state and let player decide
        game_state = self.game.get_game_state(self.game.current_player_idx)
        action = current_player.play(game_state)
        
        # Handle the action
        if isinstance(action, Tsumo):
            self.game.winner = self.game.current_player_idx
            self.game.game_over = True
            self.win_type = 'Tsumo'
            return True
        elif isinstance(action, Discard):
            current_player.remove_tile(action.tile)
            self.game.discarded_tiles.append(action.tile)
            self.player_discards[self.game.current_player_idx].append(str(action.tile)) # Track the discard
            
            # Check if another player can win with the discarded tile (Ron)
            for i, player in enumerate(self.players):
                if i != self.game.current_player_idx:
                    # Temporarily add tile to check if they can win
                    player.add_tile(action.tile)
                    can_win = player.can_win()
                    player.remove_tile(action.tile)  # Remove it immediately after checking
                    
                    if can_win:
                        self.game.winner = i
                        self.game.game_over = True
                        self.win_type = 'Ron'
                        return True
        
        # Move to next player (counterclockwise: 0→3→2→1→0)
        self.game.current_player_idx = (self.game.current_player_idx - 1) % 4
        return True

# Global game manager
game_manager = GameManager()

@app.route('/')
def index():
    """Main game page"""
    return render_template('index.html')

@app.route('/api/new_game', methods=['POST'])
def new_game():
    """Start a new game"""
    game_id = game_manager.start_new_game()
    return jsonify({'game_id': game_id, 'message': 'New game started'})

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
    if success:
        # Play AI turns until it's human's turn again or game is over
        while (not game_manager.game.is_game_over() and 
               game_manager.game.current_player_idx != 0):
            game_manager.play_ai_turn()
    
    return jsonify({'success': success, 'message': message})

@app.route('/api/tsumo', methods=['POST'])
def declare_tsumo():
    """Handle human player declaring tsumo"""
    success, message = game_manager.human_tsumo()
    if success:
        # Game is over if tsumo is declared
        game_manager.game.game_over = True
        game_manager.game.winner = 0
        game_manager.win_type = 'Tsumo'
    
    return jsonify({'success': success, 'message': message})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 