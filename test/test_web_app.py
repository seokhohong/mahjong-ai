#!/usr/bin/env python3
"""
Unit tests for the web application API endpoints.
Tests the Flask routes and GameManager functionality.
"""

import unittest
import json
import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from web.app import app, GameManager, HumanPlayer, AIPlayer
from core.game import Tile, TileType, Suit


class TestWebAppAPI(unittest.TestCase):
    """Test the Flask web application API endpoints"""
    
    def setUp(self):
        """Set up test client and test mode"""
        self.app = app.test_client()
        self.app.testing = True
        
        # Reset the global game manager for each test
        from web.app import game_manager
        game_manager.game = None
        game_manager.players = None
        game_manager.game_id = None
        game_manager.newly_drawn_tile = None
        game_manager.player_discards = [[], [], [], []]
        game_manager.win_type = None
    
    def test_index_page_loads(self):
        """Test that the index page loads successfully"""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Mahjong AI', response.data)
    
    def test_new_game_creation(self):
        """Test that a new game can be created"""
        response = self.app.post('/api/new_game', 
                                content_type='application/json')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('game_id', data)
        self.assertIn('message', data)
        self.assertEqual(data['message'], 'New game started')
        self.assertIsInstance(data['game_id'], int)
    
    def test_game_state_after_new_game(self):
        """Test that game state is correct after creating a new game"""
        # Create a new game first
        self.app.post('/api/new_game', content_type='application/json')
        
        # Get game state
        response = self.app.get('/api/game_state')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        
        # Verify basic structure
        self.assertIn('game_id', data)
        self.assertIn('human_hand', data)
        self.assertIn('other_hands', data)
        self.assertIn('current_player', data)
        self.assertIn('is_human_turn', data)
        self.assertIn('game_over', data)
        self.assertIn('possible_actions', data)
        
        # Verify game is initialized correctly
        self.assertFalse(data['game_over'])
        self.assertEqual(data['current_player'], 0)  # Human starts
        self.assertTrue(data['is_human_turn'])
        
        # Verify hand sizes - human has 12 tiles (11 dealt + 1 auto-drawn for turn)
        self.assertEqual(len(data['human_hand']), 12)  # 12 tiles (11 + drawn)
        self.assertEqual(len(data['other_hands']), 3)  # 3 AI players
        for hand_size in data['other_hands']:
            self.assertEqual(hand_size, 11)  # Each AI has 11 tiles
        
        # Verify newly drawn tile for human turn
        self.assertIsNotNone(data['newly_drawn_tile'])
        
        # Verify action structure
        expected_actions = ['tsumo', 'ron', 'pon', 'chi']
        for action in expected_actions:
            self.assertIn(action, data['possible_actions'])
    
    def test_game_state_without_game(self):
        """Test that game state returns error when no game exists"""
        response = self.app.get('/api/game_state')
        self.assertEqual(response.status_code, 400)
        
        data = json.loads(response.data)
        self.assertIn('error', data)
        self.assertEqual(data['error'], 'No active game')
    
    def test_human_discard(self):
        """Test that human player can discard a tile"""
        # Create a new game
        self.app.post('/api/new_game', content_type='application/json')
        
        # Get initial state to see what tiles are available
        state_response = self.app.get('/api/game_state')
        initial_state = json.loads(state_response.data)
        
        # Pick a tile from the hand to discard
        tile_to_discard = initial_state['human_hand'][0]
        
        # Discard the tile
        response = self.app.post('/api/discard',
                               data=json.dumps({'tile': tile_to_discard}),
                               content_type='application/json')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        self.assertIn('message', data)
        self.assertTrue(data['ai_turn_needed'])
    
    def test_invalid_tile_discard(self):
        """Test that discarding an invalid tile fails"""
        # Create a new game
        self.app.post('/api/new_game', content_type='application/json')
        
        # Get current hand to find a tile definitely not in it
        state_response = self.app.get('/api/game_state')
        state = json.loads(state_response.data)
        
        # Find a tile that's definitely not in hand
        all_possible_tiles = [f"{i}{suit}" for i in range(1, 10) for suit in ['p', 's']]
        tiles_in_hand = set(state['human_hand'])
        if state['newly_drawn_tile']:
            tiles_in_hand.add(state['newly_drawn_tile'])
        
        tile_not_in_hand = None
        for tile in all_possible_tiles:
            if tile not in tiles_in_hand:
                tile_not_in_hand = tile
                break
        
        # Try to discard a tile not in hand
        response = self.app.post('/api/discard',
                               data=json.dumps({'tile': tile_not_in_hand}),
                               content_type='application/json')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertFalse(data['success'])
        self.assertIn('Tile not in hand', data['message'])
    
    def test_tsumo_without_winning_hand(self):
        """Test that tsumo fails when hand is not winning"""
        # Create a new game
        self.app.post('/api/new_game', content_type='application/json')
        
        # Try to declare tsumo (should fail since random hand unlikely to be winning)
        response = self.app.post('/api/tsumo', content_type='application/json')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertFalse(data['success'])
        self.assertIn('Cannot declare tsumo', data['message'])
    
    def test_action_without_game(self):
        """Test that actions fail when no game exists"""
        # Try to discard without a game
        response = self.app.post('/api/discard',
                               data=json.dumps({'tile': '1p'}),
                               content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertFalse(data['success'])
        
        # Try to tsumo without a game
        response = self.app.post('/api/tsumo', content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertFalse(data['success'])
    
    def test_ai_turn_progression(self):
        """Test that AI turns progress correctly"""
        # Create a new game
        self.app.post('/api/new_game', content_type='application/json')
        
        # Get initial state
        state_response = self.app.get('/api/game_state')
        initial_state = json.loads(state_response.data)
        
        # Discard a tile to trigger AI turn
        tile_to_discard = initial_state['human_hand'][0]
        self.app.post('/api/discard',
                     data=json.dumps({'tile': tile_to_discard}),
                     content_type='application/json')
        
        # Play AI turn
        response = self.app.post('/api/play_ai_turn', content_type='application/json')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertTrue(data['success'])
        
        # Verify game state changed
        new_state_response = self.app.get('/api/game_state')
        new_state = json.loads(new_state_response.data)
        
        # Current player should have changed (unless AI won)
        if not new_state['game_over']:
            self.assertNotEqual(initial_state['current_player'], new_state['current_player'])
    
    def test_human_hand_always_populated(self):
        """Test that human hand is always populated after new game"""
        # Create a new game
        self.app.post('/api/new_game', content_type='application/json')
        
        # Get game state
        response = self.app.get('/api/game_state')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        
        # Human hand should never be empty after new game
        self.assertGreater(len(data['human_hand']), 0)
        self.assertGreaterEqual(len(data['human_hand']), 11)  # At least 11 tiles
        
        # All tiles should be valid tile strings
        for tile_str in data['human_hand']:
            self.assertRegex(tile_str, r'^[1-9][ps]$')  # Valid tile format like "1p", "9s"


class TestGameManager(unittest.TestCase):
    """Test the GameManager class directly"""
    
    def setUp(self):
        """Set up a fresh GameManager for each test"""
        self.game_manager = GameManager()
    
    def test_game_manager_initialization(self):
        """Test that GameManager initializes correctly"""
        self.assertIsNone(self.game_manager.game)
        self.assertIsNone(self.game_manager.players)
        self.assertIsNone(self.game_manager.game_id)
        self.assertIsNone(self.game_manager.newly_drawn_tile)
        self.assertEqual(self.game_manager.player_discards, [[], [], [], []])
        self.assertIsNone(self.game_manager.win_type)
    
    def test_start_new_game(self):
        """Test that starting a new game works correctly"""
        game_id = self.game_manager.start_new_game()
        
        # Verify game was created
        self.assertIsNotNone(self.game_manager.game)
        self.assertIsNotNone(self.game_manager.players)
        self.assertIsInstance(game_id, int)
        self.assertEqual(self.game_manager.game_id, game_id)
        
        # Verify players were created correctly
        self.assertEqual(len(self.game_manager.players), 4)
        self.assertIsInstance(self.game_manager.players[0], HumanPlayer)
        for i in range(1, 4):
            self.assertIsInstance(self.game_manager.players[i], AIPlayer)
        
        # Verify each player has initial hand
        for player in self.game_manager.players:
            self.assertEqual(len(player.hand), 11)  # 11 tiles dealt initially
    
    def test_get_game_state_structure(self):
        """Test that get_game_state returns correct structure"""
        self.game_manager.start_new_game()
        state = self.game_manager.get_game_state()
        
        # Verify all required fields are present
        required_fields = [
            'game_id', 'current_player', 'human_hand', 'discarded_tiles',
            'player_discards', 'other_hands', 'remaining_tiles', 'game_over',
            'winner', 'win_type', 'is_human_turn', 'newly_drawn_tile',
            'called_sets', 'possible_actions', 'last_discarded_tile'
        ]
        
        for field in required_fields:
            self.assertIn(field, state)
    
    def test_hand_dealing_correctness(self):
        """Test that hands are dealt correctly"""
        self.game_manager.start_new_game()
        
        # Verify total tiles are correct
        total_tiles_in_hands = sum(len(player.hand) for player in self.game_manager.players)
        tiles_in_deck = len(self.game_manager.game.tiles)
        
        # Total should be 72 tiles (4 copies of 18 types)
        self.assertEqual(total_tiles_in_hands + tiles_in_deck, 72)
        
        # Each player should have 11 tiles initially
        for player in self.game_manager.players:
            self.assertEqual(len(player.hand), 11)
        
        # Remaining tiles should be 72 - (4 * 11) = 28
        self.assertEqual(tiles_in_deck, 28)
    
    def test_tile_sorting(self):
        """Test that tile sorting works correctly"""
        # Create test tiles
        tiles = [
            Tile(Suit.SOUZU, TileType.FIVE),
            Tile(Suit.PINZU, TileType.THREE),
            Tile(Suit.SOUZU, TileType.ONE),
            Tile(Suit.PINZU, TileType.ONE),
        ]
        
        sorted_tiles = self.game_manager._sort_hand(tiles)
        
        # Should be sorted: pinzu before souzu, then by number
        expected = [
            Tile(Suit.PINZU, TileType.ONE),
            Tile(Suit.PINZU, TileType.THREE),
            Tile(Suit.SOUZU, TileType.ONE),
            Tile(Suit.SOUZU, TileType.FIVE),
        ]
        
        self.assertEqual(sorted_tiles, expected)
    
    def test_human_discard_validation(self):
        """Test that human discard validation works"""
        self.game_manager.start_new_game()
        
        # Try to discard when not human's turn
        self.game_manager.game.current_player_idx = 1  # Set to AI turn
        success, message = self.game_manager.human_discard('1p')
        self.assertFalse(success)
        self.assertIn("Not human player's turn", message)
        
        # Try to discard invalid tile
        self.game_manager.game.current_player_idx = 0  # Set to human turn
        success, message = self.game_manager.human_discard('invalid')
        self.assertFalse(success)
        self.assertIn("Invalid tile", message)
        
        # Try to discard tile not in hand
        # Find a tile that's definitely not in the human's hand
        human_hand = self.game_manager.players[0].hand
        all_possible_tiles = [f"{i}{suit}" for i in range(1, 10) for suit in ['p', 's']]
        hand_tile_strings = [str(tile) for tile in human_hand]
        
        tile_not_in_hand = None
        for tile_str in all_possible_tiles:
            if tile_str not in hand_tile_strings:
                tile_not_in_hand = tile_str
                break
        
        success, message = self.game_manager.human_discard(tile_not_in_hand)
        self.assertFalse(success)
        self.assertIn("Tile not in hand", message)
    
    def test_newly_drawn_tile_tracking(self):
        """Test that newly drawn tiles are tracked correctly"""
        self.game_manager.start_new_game()
        
        # Initially, human should have drawn a tile automatically
        state = self.game_manager.get_game_state()
        self.assertIsNotNone(state['newly_drawn_tile'])
        
        # Human should have 12 tiles total (11 + 1 drawn)
        total_human_tiles = len(state['human_hand'])
        if state['newly_drawn_tile'] in state['human_hand']:
            # Newly drawn tile is included in hand count
            self.assertEqual(total_human_tiles, 12)
        else:
            # Newly drawn tile is separate from hand
            self.assertEqual(total_human_tiles, 11)


if __name__ == '__main__':
    # Run the tests
    unittest.main(verbosity=2)