#!/usr/bin/env python3
"""
Unit tests for SimpleJong game using Python's unittest framework
"""

import unittest
import random
import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.game import SimpleJong, Player, Tile, TileType, Discard, Tsumo, Ron, GameState


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
        action = player.play(game_state)
        
        # Player should return an action
        self.assertIsNotNone(action)
        
        # If it's a discard action, check the tile
        if isinstance(action, Discard):
            discarded_tile = action.tile
            self.assertIsInstance(discarded_tile, Tile)
            
            # Hand size should decrease by 1 after discarding
            player.remove_tile(discarded_tile)
            self.assertEqual(len(player.hand), initial_hand_size - 1)
        elif isinstance(action, Tsumo):
            # If it's a tsumo action, that's also valid
            pass
        else:
            self.fail(f"Unexpected action type: {type(action)}")
    
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

    def test_ron_functionality(self):
        """Test that players can declare Ron when another player discards a tile that completes their hand"""
        # Create a player with a hand that needs one specific tile to win
        player = Player(0)
        # Hand: 333, 444, 55 (needs 5 to complete 555)
        test_hand = [
            Tile(TileType.THREE), Tile(TileType.THREE), Tile(TileType.THREE),
            Tile(TileType.FOUR), Tile(TileType.FOUR), Tile(TileType.FOUR),
            Tile(TileType.FIVE), Tile(TileType.FIVE)
        ]
        player.hand = test_hand
        
        # Test that player can declare Ron with a 5 tile
        five_tile = Tile(TileType.FIVE)
        self.assertTrue(player.can_ron(five_tile), 
                       f"Player should be able to declare Ron with {five_tile}")
        
        # Test that player cannot declare Ron with other tiles
        one_tile = Tile(TileType.ONE)
        self.assertFalse(player.can_ron(one_tile), 
                        f"Player should not be able to declare Ron with {one_tile}")
        
        # Test that player can declare Ron when the tile is in visible_tiles
        game_state = GameState(
            player_hand=test_hand,
            visible_tiles=[five_tile],  # The needed tile is discarded
            remaining_tiles=10,
            player_id=0,
            other_players_discarded={}
        )
        
        action = player.play(game_state)
        self.assertIsInstance(action, Ron, 
                            f"Player should declare Ron when {five_tile} is discarded")

    def test_ron_vs_tsumo_priority(self):
        """Test that Tsumo takes priority over Ron when both are possible"""
        player = Player(0)
        # Hand: 333, 444, 555 (already a winning hand)
        test_hand = [
            Tile(TileType.THREE), Tile(TileType.THREE), Tile(TileType.THREE),
            Tile(TileType.FOUR), Tile(TileType.FOUR), Tile(TileType.FOUR),
            Tile(TileType.FIVE), Tile(TileType.FIVE), Tile(TileType.FIVE)
        ]
        player.hand = test_hand
        
        # Even if there's a discarded tile that could complete the hand,
        # player should declare Tsumo since they already have a winning hand
        game_state = GameState(
            player_hand=test_hand,
            visible_tiles=[Tile(TileType.ONE)],  # Some discarded tile
            remaining_tiles=10,
            player_id=0,
            other_players_discarded={}
        )
        
        action = player.play(game_state)
        self.assertIsInstance(action, Tsumo, 
                            "Player should declare Tsumo when they already have a winning hand")

    def test_ron_game_scenario(self):
        """Test a complete game scenario where Ron is declared"""
        # Create players
        players = [Player(i) for i in range(4)]
        
        # Create game (this will deal initial tiles)
        game = SimpleJong(players)
        
        # Set up a specific scenario where player 1 can declare Ron
        # Player 1: 333, 444, 55 (needs 5 to complete 555)
        players[1].hand = [
            Tile(TileType.THREE), Tile(TileType.THREE), Tile(TileType.THREE),
            Tile(TileType.FOUR), Tile(TileType.FOUR), Tile(TileType.FOUR),
            Tile(TileType.FIVE), Tile(TileType.FIVE)
        ]
        
        # Other players have random hands (just ensure they have 8 tiles)
        for i in [0, 2, 3]:
            players[i].hand = [Tile(TileType.ONE) for _ in range(8)]
        
        # Manually set the current player to 0
        game.current_player_idx = 0
        
        # Simulate player 0 discarding a 5 tile
        five_tile = Tile(TileType.FIVE)
        players[0].remove_tile(five_tile)
        game.discarded_tiles.append(five_tile)
        
        # Check if player 1 can declare Ron (should have 8 tiles + 1 discarded = 9 tiles)
        self.assertTrue(players[1].can_ron(five_tile), 
                       "Player 1 should be able to declare Ron with the discarded 5 tile")
        
        # Verify that the hand with the added tile is a winning hand
        players[1].add_tile(five_tile)
        self.assertTrue(players[1].can_win(), 
                       "Player 1 should have a winning hand after adding the 5 tile")
        players[1].remove_tile(five_tile)  # Remove it back
        
        # Simulate the Ron declaration
        game.winner = 1
        game.game_over = True
        players[1].add_tile(five_tile)  # Add the claimed tile to winner's hand
        
        self.assertEqual(game.get_winner(), 1, "Player 1 should be the winner")
        self.assertTrue(game.is_game_over(), "Game should be over after Ron declaration")


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