#!/usr/bin/env python3
"""
Unit tests for SimpleJong game using Python's unittest framework
"""

import unittest
import random
from simple_jong import SimpleJong, Player, Tile, TileType


class TestSimpleJong(unittest.TestCase):
    """Test cases for SimpleJong game"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        self.players = [Player(i) for i in range(4)]
        self.game = SimpleJong(self.players)
    
    def test_game_initialization(self):
        """Test that the game initializes correctly"""
        # Check that we have exactly 4 players
        self.assertEqual(len(self.game.players), 4)
        
        # Check that each player has 8 tiles initially
        for player in self.game.players:
            self.assertEqual(len(player.hand), 8)
        
        # Check that there are remaining tiles (54 total - 32 dealt = 22 remaining)
        self.assertEqual(self.game.get_remaining_tiles(), 22)
        
        # Check that game is not over initially
        self.assertFalse(self.game.is_game_over())
        self.assertIsNone(self.game.get_winner())
    
    def test_winning_hand_three_sets_same_tiles(self):
        """Test winning hand detection with three sets of same tiles (333, 444, 555)"""
        player = Player(0)
        test_hand = [
            Tile(TileType.THREE), Tile(TileType.THREE), Tile(TileType.THREE),
            Tile(TileType.FOUR), Tile(TileType.FOUR), Tile(TileType.FOUR),
            Tile(TileType.FIVE), Tile(TileType.FIVE), Tile(TileType.FIVE)
        ]
        player.hand = test_hand
        
        self.assertTrue(player.can_win(), 
                       f"Hand {[str(tile) for tile in test_hand]} should be a winning hand")
    
    def test_winning_hand_three_sequences(self):
        """Test winning hand detection with three sequences (123, 234, 345)"""
        player = Player(1)
        test_hand = [
            Tile(TileType.ONE), Tile(TileType.TWO), Tile(TileType.THREE),
            Tile(TileType.TWO), Tile(TileType.THREE), Tile(TileType.FOUR),
            Tile(TileType.THREE), Tile(TileType.FOUR), Tile(TileType.FIVE)
        ]
        player.hand = test_hand
        
        self.assertTrue(player.can_win(), 
                       f"Hand {[str(tile) for tile in test_hand]} should be a winning hand")
    
    def test_winning_hand_mixed_sets(self):
        """Test winning hand detection with mixed sets (333, 234, 567)"""
        player = Player(2)
        test_hand = [
            Tile(TileType.THREE), Tile(TileType.THREE), Tile(TileType.THREE),
            Tile(TileType.TWO), Tile(TileType.THREE), Tile(TileType.FOUR),
            Tile(TileType.FIVE), Tile(TileType.SIX), Tile(TileType.SEVEN)
        ]
        player.hand = test_hand
        
        self.assertTrue(player.can_win(), 
                       f"Hand {[str(tile) for tile in test_hand]} should be a winning hand")
    
    def test_non_winning_hand(self):
        """Test that non-winning hands are correctly identified"""
        player = Player(3)
        test_hand = [
            Tile(TileType.THREE), Tile(TileType.THREE), Tile(TileType.THREE),
            Tile(TileType.TWO), Tile(TileType.THREE), Tile(TileType.FOUR),
            Tile(TileType.ONE), Tile(TileType.TWO), Tile(TileType.FIVE)
        ]
        player.hand = test_hand
        
        self.assertFalse(player.can_win(), 
                        f"Hand {[str(tile) for tile in test_hand]} should not be a winning hand")
    
    def test_winning_hand_insufficient_tiles(self):
        """Test that hands with insufficient tiles are not winning"""
        player = Player(0)
        test_hand = [
            Tile(TileType.ONE), Tile(TileType.TWO), Tile(TileType.THREE),
            Tile(TileType.FOUR), Tile(TileType.FIVE), Tile(TileType.SIX),
            Tile(TileType.SEVEN), Tile(TileType.EIGHT)  # Only 8 tiles
        ]
        player.hand = test_hand
        
        self.assertFalse(player.can_win(), 
                        f"Hand with {len(test_hand)} tiles should not be a winning hand")
    
    def test_winning_hand_too_many_tiles(self):
        """Test that hands with too many tiles are not winning"""
        player = Player(0)
        test_hand = [
            Tile(TileType.ONE), Tile(TileType.TWO), Tile(TileType.THREE),
            Tile(TileType.FOUR), Tile(TileType.FIVE), Tile(TileType.SIX),
            Tile(TileType.SEVEN), Tile(TileType.EIGHT), Tile(TileType.NINE),
            Tile(TileType.ONE)  # 10 tiles
        ]
        player.hand = test_hand
        
        self.assertFalse(player.can_win(), 
                        f"Hand with {len(test_hand)} tiles should not be a winning hand")
    
    def test_game_state_creation(self):
        """Test that game state is created correctly for each player"""
        game_state = self.game.get_game_state(0)
        
        self.assertEqual(game_state.player_id, 0)
        self.assertEqual(len(game_state.player_hand), 8)
        self.assertEqual(game_state.remaining_tiles, 22)
        self.assertEqual(len(game_state.visible_tiles), 0)
        self.assertEqual(len(game_state.other_players_discarded), 3)
    
    def test_player_turn(self):
        """Test that players can take turns and discard tiles"""
        player = self.players[0]
        initial_hand_size = len(player.hand)
        
        game_state = self.game.get_game_state(0)
        discarded_tile = player.play(game_state)
        
        # Player should discard a tile
        self.assertIsNotNone(discarded_tile)
        self.assertIsInstance(discarded_tile, Tile)
        
        # Hand size should decrease by 1 after discarding
        player.remove_tile(discarded_tile)
        self.assertEqual(len(player.hand), initial_hand_size - 1)
    
    def test_tile_equality_and_hash(self):
        """Test that tiles can be compared and hashed correctly"""
        tile1 = Tile(TileType.ONE)
        tile2 = Tile(TileType.ONE)
        tile3 = Tile(TileType.TWO)
        
        # Test equality
        self.assertEqual(tile1, tile2)
        self.assertNotEqual(tile1, tile3)
        
        # Test hash
        self.assertEqual(hash(tile1), hash(tile2))
        self.assertNotEqual(hash(tile1), hash(tile3))
    
    def test_tile_string_representation(self):
        """Test that tiles have correct string representation"""
        tile = Tile(TileType.FIVE)
        self.assertEqual(str(tile), "5p")
        
        tile = Tile(TileType.ONE)
        self.assertEqual(str(tile), "1p")
    
    def test_game_round_play(self):
        """Test that a game round can be played (may not have a winner)"""
        winner = self.game.play_round()
        
        # Game should be over after playing a round
        self.assertTrue(self.game.is_game_over())
        
        # Winner should be None or a valid player ID
        if winner is not None:
            self.assertIn(winner, [0, 1, 2, 3])
            # If there's a winner, they should have a winning hand
            self.assertTrue(self.players[winner].can_win())
    
    def test_invalid_player_count(self):
        """Test that SimpleJong raises an error with invalid player count"""
        with self.assertRaises(ValueError):
            SimpleJong([Player(0), Player(1)])  # Only 2 players
        
        with self.assertRaises(ValueError):
            SimpleJong([Player(0), Player(1), Player(2), Player(3), Player(4)])  # 5 players


class TestPlayerMethods(unittest.TestCase):
    """Test cases for Player class methods"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        self.player = Player(0)
    
    def test_player_initialization(self):
        """Test that player initializes correctly"""
        self.assertEqual(self.player.player_id, 0)
        self.assertEqual(len(self.player.hand), 0)
    
    def test_add_and_remove_tiles(self):
        """Test adding and removing tiles from player's hand"""
        tile = Tile(TileType.ONE)
        
        # Add tile
        self.player.add_tile(tile)
        self.assertEqual(len(self.player.hand), 1)
        self.assertIn(tile, self.player.hand)
        
        # Remove tile
        self.player.remove_tile(tile)
        self.assertEqual(len(self.player.hand), 0)
        self.assertNotIn(tile, self.player.hand)
    
    def test_remove_nonexistent_tile(self):
        """Test removing a tile that doesn't exist in hand"""
        tile = Tile(TileType.ONE)
        other_tile = Tile(TileType.TWO)
        
        self.player.add_tile(tile)
        initial_hand_size = len(self.player.hand)
        
        # Try to remove a tile that's not in the hand
        self.player.remove_tile(other_tile)
        
        # Hand size should remain the same
        self.assertEqual(len(self.player.hand), initial_hand_size)
        self.assertIn(tile, self.player.hand)


if __name__ == "__main__":
    # Run the tests
    unittest.main(verbosity=2) 