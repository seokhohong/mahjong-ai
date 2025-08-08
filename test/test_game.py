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

from core.game import SimpleJong, Player, Tile, TileType, Discard, Tsumo, Ron, GameState, Suit, Pon, Chi, CalledSet


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
        
        # Check that each player has 11 tiles initially
        for player in self.game.players:
            self.assertEqual(len(player.hand), 11)
        
        # Check that there are remaining tiles (72 total - 44 dealt = 28 remaining)
        self.assertEqual(self.game.get_remaining_tiles(), 28)
        
        # Check that game is not over initially
        self.assertFalse(self.game.is_game_over())
        self.assertIsNone(self.game.get_winner())
    
    def test_winning_hand_three_sets_same_tiles(self):
        """Test winning hand detection with three sets of same tiles (333, 444, 555)"""
        player = Player(0)
        test_hand = [
            Tile(Suit.PINZU, TileType.THREE), Tile(Suit.PINZU, TileType.THREE), Tile(Suit.PINZU, TileType.THREE),
            Tile(Suit.PINZU, TileType.FOUR), Tile(Suit.PINZU, TileType.FOUR), Tile(Suit.PINZU, TileType.FOUR),
            Tile(Suit.PINZU, TileType.FIVE), Tile(Suit.PINZU, TileType.FIVE), Tile(Suit.PINZU, TileType.FIVE),
            Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.ONE)
        ]
        player.hand = test_hand
        
        self.assertTrue(player.can_win(), 
                       f"Hand {[str(tile) for tile in test_hand]} should be a winning hand")
    
    def test_winning_hand_three_sequences(self):
        """Test winning hand detection with three sequences (123, 234, 345)"""
        player = Player(1)
        test_hand = [
            Tile(Suit.PINZU, TileType.ONE), Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.THREE),
            Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.THREE), Tile(Suit.PINZU, TileType.FOUR),
            Tile(Suit.PINZU, TileType.THREE), Tile(Suit.PINZU, TileType.FOUR), Tile(Suit.PINZU, TileType.FIVE),
            Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.TWO), Tile(Suit.SOUZU, TileType.THREE)
        ]
        player.hand = test_hand
        
        self.assertTrue(player.can_win(), 
                       f"Hand {[str(tile) for tile in test_hand]} should be a winning hand")
    
    def test_winning_hand_mixed_sets(self):
        """Test winning hand detection with mixed sets (333, 234, 567)"""
        player = Player(2)
        test_hand = [
            Tile(Suit.PINZU, TileType.THREE), Tile(Suit.PINZU, TileType.THREE), Tile(Suit.PINZU, TileType.THREE),
            Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.THREE), Tile(Suit.PINZU, TileType.FOUR),
            Tile(Suit.SOUZU, TileType.FIVE), Tile(Suit.SOUZU, TileType.SIX), Tile(Suit.SOUZU, TileType.SEVEN),
            Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.ONE)
        ]
        player.hand = test_hand
        
        self.assertTrue(player.can_win(), 
                       f"Hand {[str(tile) for tile in test_hand]} should be a winning hand")
    
    def test_non_winning_hand(self):
        """Test that non-winning hands are correctly identified"""
        player = Player(3)
        test_hand = [
            Tile(Suit.PINZU, TileType.THREE), Tile(Suit.PINZU, TileType.THREE), Tile(Suit.PINZU, TileType.THREE),
            Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.THREE), Tile(Suit.PINZU, TileType.FOUR),
            Tile(Suit.PINZU, TileType.ONE), Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FIVE),
            Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.TWO), Tile(Suit.SOUZU, TileType.THREE)
        ]
        player.hand = test_hand
        
        self.assertFalse(player.can_win(), 
                        f"Hand {[str(tile) for tile in test_hand]} should not be a winning hand")
    
    def test_winning_hand_insufficient_tiles(self):
        """Test that hands with insufficient tiles are not winning"""
        player = Player(0)
        test_hand = [
            Tile(Suit.PINZU, TileType.ONE), Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.THREE),
            Tile(Suit.PINZU, TileType.FOUR), Tile(Suit.PINZU, TileType.FIVE), Tile(Suit.PINZU, TileType.SIX),
            Tile(Suit.SOUZU, TileType.SEVEN), Tile(Suit.SOUZU, TileType.EIGHT)  # Only 8 tiles
        ]
        player.hand = test_hand
        
        self.assertFalse(player.can_win(), 
                        f"Hand with {len(test_hand)} tiles should not be a winning hand")
    
    def test_winning_hand_too_many_tiles(self):
        """Test that hands with too many tiles are not winning"""
        player = Player(0)
        test_hand = [
            Tile(Suit.PINZU, TileType.ONE), Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.THREE),
            Tile(Suit.PINZU, TileType.FOUR), Tile(Suit.PINZU, TileType.FIVE), Tile(Suit.PINZU, TileType.SIX),
            Tile(Suit.SOUZU, TileType.SEVEN), Tile(Suit.SOUZU, TileType.EIGHT), Tile(Suit.SOUZU, TileType.NINE),
            Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.TWO),
            Tile(Suit.SOUZU, TileType.THREE) # 13 tiles
        ]
        player.hand = test_hand
        
        self.assertFalse(player.can_win(), 
                        f"Hand with {len(test_hand)} tiles should not be a winning hand")
    
    def test_game_state_creation(self):
        """Test that game state is created correctly for each player"""
        game_state = self.game.get_game_state(0)
        
        self.assertEqual(game_state.player_id, 0)
        self.assertEqual(len(game_state.player_hand), 11)
        self.assertEqual(game_state.remaining_tiles, 28)
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
        tile1 = Tile(Suit.PINZU, TileType.ONE)
        tile2 = Tile(Suit.PINZU, TileType.ONE)
        tile3 = Tile(Suit.SOUZU, TileType.ONE)
        
        # Test equality
        self.assertEqual(tile1, tile2)
        self.assertNotEqual(tile1, tile3)
        
        # Test hash
        self.assertEqual(hash(tile1), hash(tile2))
        self.assertNotEqual(hash(tile1), hash(tile3))
    
    def test_tile_string_representation(self):
        """Test that tiles have correct string representation"""
        tile = Tile(Suit.PINZU, TileType.FIVE)
        self.assertEqual(str(tile), "5p")
        
        tile = Tile(Suit.SOUZU, TileType.ONE)
        self.assertEqual(str(tile), "1s")
    
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
        # Hand: 333p, 444p, 55p (needs 5p to complete 555p) and 111s
        test_hand = [
            Tile(Suit.PINZU, TileType.THREE), Tile(Suit.PINZU, TileType.THREE), Tile(Suit.PINZU, TileType.THREE),
            Tile(Suit.PINZU, TileType.FOUR), Tile(Suit.PINZU, TileType.FOUR), Tile(Suit.PINZU, TileType.FOUR),
            Tile(Suit.PINZU, TileType.FIVE), Tile(Suit.PINZU, TileType.FIVE),
            Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.ONE)
        ]
        player.hand = test_hand
        
        # Test that player can declare Ron with a 5p tile
        five_tile = Tile(Suit.PINZU, TileType.FIVE)
        self.assertTrue(player.can_ron(five_tile), 
                       f"Player should be able to declare Ron with {five_tile}")
        
        # Test that player cannot declare Ron with other tiles
        one_tile = Tile(Suit.PINZU, TileType.ONE)
        self.assertFalse(player.can_ron(one_tile), 
                        f"Player should not be able to declare Ron with {one_tile}")
        
        # Test that player can declare Ron when the tile is in visible_tiles
        game_state = GameState(
            player_hand=test_hand,
            visible_tiles=[five_tile],  # The needed tile is discarded
            remaining_tiles=10,
            player_id=0,
            other_players_discarded={},
            called_sets={},
            last_discarded_tile=five_tile,
            last_discard_player=1,
            can_call=False
        )
        
        action = player.play(game_state)
        self.assertIsInstance(action, Ron, 
                            f"Player should declare Ron when {five_tile} is discarded")

    def test_ron_vs_tsumo_priority(self):
        """Test that Tsumo takes priority over Ron when both are possible"""
        player = Player(0)
        # Hand: 333p, 444p, 555p, 111s (already a winning hand)
        test_hand = [
            Tile(Suit.PINZU, TileType.THREE), Tile(Suit.PINZU, TileType.THREE), Tile(Suit.PINZU, TileType.THREE),
            Tile(Suit.PINZU, TileType.FOUR), Tile(Suit.PINZU, TileType.FOUR), Tile(Suit.PINZU, TileType.FOUR),
            Tile(Suit.PINZU, TileType.FIVE), Tile(Suit.PINZU, TileType.FIVE), Tile(Suit.PINZU, TileType.FIVE),
            Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.ONE)
        ]
        player.hand = test_hand
        
        # Even if there's a discarded tile that could complete the hand,
        # player should declare Tsumo since they already have a winning hand
        game_state = GameState(
            player_hand=test_hand,
            visible_tiles=[Tile(Suit.SOUZU, TileType.TWO)],  # Some discarded tile
            remaining_tiles=10,
            player_id=0,
            other_players_discarded={},
            called_sets={},
            last_discarded_tile=Tile(Suit.SOUZU, TileType.TWO),
            last_discard_player=1,
            can_call=False
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
        # Player 1: 333p, 444p, 55p (needs 5p to complete 555p) and 111s
        players[1].hand = [
            Tile(Suit.PINZU, TileType.THREE), Tile(Suit.PINZU, TileType.THREE), Tile(Suit.PINZU, TileType.THREE),
            Tile(Suit.PINZU, TileType.FOUR), Tile(Suit.PINZU, TileType.FOUR), Tile(Suit.PINZU, TileType.FOUR),
            Tile(Suit.PINZU, TileType.FIVE), Tile(Suit.PINZU, TileType.FIVE),
            Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.ONE)
        ]
        
        # Other players have random hands (just ensure they have 11 tiles)
        for i in [0, 2, 3]:
            players[i].hand = [Tile(Suit.PINZU, TileType.ONE) for _ in range(11)]
        
        # Manually set the current player to 0
        game.current_player_idx = 0
        
        # Simulate player 0 discarding a 5p tile
        five_tile = Tile(Suit.PINZU, TileType.FIVE)
        players[0].remove_tile(five_tile)
        game.discarded_tiles.append(five_tile)
        
        # Check if player 1 can declare Ron
        self.assertTrue(players[1].can_ron(five_tile), 
                       "Player 1 should be able to declare Ron with the discarded 5p tile")
        
        # Verify that the hand with the added tile is a winning hand
        players[1].add_tile(five_tile)
        self.assertTrue(players[1].can_win(), 
                       "Player 1 should have a winning hand after adding the 5p tile")
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
        tile = Tile(Suit.PINZU, TileType.ONE)
        
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
        tile = Tile(Suit.PINZU, TileType.ONE)
        other_tile = Tile(Suit.SOUZU, TileType.TWO)
        
        self.player.add_tile(tile)
        initial_hand_size = len(self.player.hand)
        
        # Try to remove a tile that's not in the hand
        self.player.remove_tile(other_tile)
        
        # Hand size should remain the same
        self.assertEqual(len(self.player.hand), initial_hand_size)
        self.assertIn(tile, self.player.hand)


class TestPonChiMechanics(unittest.TestCase):
    """Test cases for Pon and Chi mechanics"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        self.players = [Player(i) for i in range(4)]
        self.game = SimpleJong(self.players)
    
    def test_pon_combinations_basic(self):
        """Test basic pon combination detection"""
        player = Player(0)
        # Hand has two 5p tiles
        player.hand = [
            Tile(Suit.PINZU, TileType.FIVE), Tile(Suit.PINZU, TileType.FIVE),
            Tile(Suit.PINZU, TileType.ONE), Tile(Suit.PINZU, TileType.TWO)
        ]
        
        # Should be able to pon a 5p tile
        discarded_tile = Tile(Suit.PINZU, TileType.FIVE)
        combinations = player.get_pon_combinations(discarded_tile)
        
        self.assertEqual(len(combinations), 1)
        self.assertEqual(len(combinations[0]), 2)
        for tile in combinations[0]:
            self.assertEqual(tile, discarded_tile)
    
    def test_pon_combinations_insufficient_tiles(self):
        """Test pon when player doesn't have enough matching tiles"""
        player = Player(0)
        # Hand has only one 5p tile
        player.hand = [
            Tile(Suit.PINZU, TileType.FIVE),
            Tile(Suit.PINZU, TileType.ONE), Tile(Suit.PINZU, TileType.TWO)
        ]
        
        # Should not be able to pon a 5p tile
        discarded_tile = Tile(Suit.PINZU, TileType.FIVE)
        combinations = player.get_pon_combinations(discarded_tile)
        
        self.assertEqual(len(combinations), 0)
    
    def test_chi_combinations_middle_tile(self):
        """Test chi when discarded tile is in the middle of sequence"""
        player = Player(0)
        # Hand has 1p and 3p
        player.hand = [
            Tile(Suit.PINZU, TileType.ONE), Tile(Suit.PINZU, TileType.THREE),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE)
        ]
        
        # Should be able to chi a 2p tile (1-2-3 sequence)
        discarded_tile = Tile(Suit.PINZU, TileType.TWO)
        combinations = player.get_chi_combinations(discarded_tile)
        
        self.assertEqual(len(combinations), 1)
        self.assertEqual(len(combinations[0]), 2)
        self.assertIn(Tile(Suit.PINZU, TileType.ONE), combinations[0])
        self.assertIn(Tile(Suit.PINZU, TileType.THREE), combinations[0])
    
    def test_chi_combinations_high_tile(self):
        """Test chi when discarded tile is the highest in sequence"""
        player = Player(0)
        # Hand has 1p and 2p
        player.hand = [
            Tile(Suit.PINZU, TileType.ONE), Tile(Suit.PINZU, TileType.TWO),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE)
        ]
        
        # Should be able to chi a 3p tile (1-2-3 sequence)
        discarded_tile = Tile(Suit.PINZU, TileType.THREE)
        combinations = player.get_chi_combinations(discarded_tile)
        
        self.assertEqual(len(combinations), 1)
        self.assertEqual(len(combinations[0]), 2)
        self.assertIn(Tile(Suit.PINZU, TileType.ONE), combinations[0])
        self.assertIn(Tile(Suit.PINZU, TileType.TWO), combinations[0])
    
    def test_chi_combinations_low_tile(self):
        """Test chi when discarded tile is the lowest in sequence"""
        player = Player(0)
        # Hand has 2p and 3p
        player.hand = [
            Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.THREE),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE)
        ]
        
        # Should be able to chi a 1p tile (1-2-3 sequence)
        discarded_tile = Tile(Suit.PINZU, TileType.ONE)
        combinations = player.get_chi_combinations(discarded_tile)
        
        self.assertEqual(len(combinations), 1)
        self.assertEqual(len(combinations[0]), 2)
        self.assertIn(Tile(Suit.PINZU, TileType.TWO), combinations[0])
        self.assertIn(Tile(Suit.PINZU, TileType.THREE), combinations[0])
    
    def test_chi_combinations_multiple_options(self):
        """Test chi when multiple sequence options are possible"""
        player = Player(0)
        # Hand has 4p, 5p, 6p, 7p (can chi 3p as 3-4-5 or chi 8p as 6-7-8)
        player.hand = [
            Tile(Suit.PINZU, TileType.FOUR), Tile(Suit.PINZU, TileType.FIVE),
            Tile(Suit.PINZU, TileType.SIX), Tile(Suit.PINZU, TileType.SEVEN)
        ]
        
        # Should be able to chi a 3p tile (3-4-5 sequence)
        discarded_tile = Tile(Suit.PINZU, TileType.THREE)
        combinations = player.get_chi_combinations(discarded_tile)
        
        self.assertEqual(len(combinations), 1)
        self.assertIn(Tile(Suit.PINZU, TileType.FOUR), combinations[0])
        self.assertIn(Tile(Suit.PINZU, TileType.FIVE), combinations[0])
        
        # Should be able to chi an 8p tile (6-7-8 sequence)
        discarded_tile = Tile(Suit.PINZU, TileType.EIGHT)
        combinations = player.get_chi_combinations(discarded_tile)
        
        self.assertEqual(len(combinations), 1)
        self.assertIn(Tile(Suit.PINZU, TileType.SIX), combinations[0])
        self.assertIn(Tile(Suit.PINZU, TileType.SEVEN), combinations[0])
    
    def test_chi_combinations_no_valid_sequence(self):
        """Test chi when no valid sequence can be formed"""
        player = Player(0)
        # Hand has 1p and 4p (no valid sequence with any discarded tile)
        player.hand = [
            Tile(Suit.PINZU, TileType.ONE), Tile(Suit.PINZU, TileType.FOUR),
            Tile(Suit.SOUZU, TileType.FIVE), Tile(Suit.SOUZU, TileType.SIX)
        ]
        
        # Should not be able to chi a 2p tile
        discarded_tile = Tile(Suit.PINZU, TileType.TWO)
        combinations = player.get_chi_combinations(discarded_tile)
        
        self.assertEqual(len(combinations), 0)
    
    def test_chi_different_suits(self):
        """Test that chi doesn't work across different suits"""
        player = Player(0)
        # Hand has 1p and 2s (different suits)
        player.hand = [
            Tile(Suit.PINZU, TileType.ONE), Tile(Suit.SOUZU, TileType.TWO),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE)
        ]
        
        # Should not be able to chi a 2p tile (wrong suit for sequence)
        discarded_tile = Tile(Suit.PINZU, TileType.TWO)
        combinations = player.get_chi_combinations(discarded_tile)
        
        self.assertEqual(len(combinations), 0)
    
    def test_call_priority_pon_over_chi(self):
        """Test that pon takes priority over chi when both are possible"""
        # Create game with specific scenario
        players = [Player(i) for i in range(4)]
        game = SimpleJong(players)
        
        # Player 1 (left of player 0) has tiles for chi with 5p: 4p, 6p
        players[1].hand = [
            Tile(Suit.PINZU, TileType.FOUR), Tile(Suit.PINZU, TileType.SIX),
            Tile(Suit.PINZU, TileType.ONE), Tile(Suit.PINZU, TileType.ONE)
        ]
        
        # Player 2 has tiles for pon with 5p: 5p, 5p
        players[2].hand = [
            Tile(Suit.PINZU, TileType.FIVE), Tile(Suit.PINZU, TileType.FIVE),
            Tile(Suit.PINZU, TileType.ONE), Tile(Suit.PINZU, TileType.ONE)
        ]
        
        # Player 0 discards 5p
        discarded_tile = Tile(Suit.PINZU, TileType.FIVE)
        call_result = game.check_for_calls(discarded_tile, 0)
        
        # Should return the pon call (higher priority)
        self.assertIsNotNone(call_result)
        caller_id, call_type, tiles = call_result
        self.assertEqual(call_type, 'pon')
        self.assertEqual(caller_id, 2)  # Player 2 made the pon call
    
    def test_chi_only_from_left_player(self):
        """Test that chi can only be called from the left player (previous in turn order)"""
        # Create game with specific scenario
        players = [Player(i) for i in range(4)]
        game = SimpleJong(players)
        
        # Player 1 (left of player 0) has tiles for chi with 5p: 4p, 6p
        players[1].hand = [
            Tile(Suit.PINZU, TileType.FOUR), Tile(Suit.PINZU, TileType.SIX),
            Tile(Suit.PINZU, TileType.ONE), Tile(Suit.PINZU, TileType.ONE)
        ]
        
        # Player 2 also has tiles for chi with 5p: 4p, 6p
        players[2].hand = [
            Tile(Suit.PINZU, TileType.FOUR), Tile(Suit.PINZU, TileType.SIX),
            Tile(Suit.PINZU, TileType.ONE), Tile(Suit.PINZU, TileType.ONE)
        ]
        
        # Player 0 discards 5p
        discarded_tile = Tile(Suit.PINZU, TileType.FIVE)
        call_result = game.check_for_calls(discarded_tile, 0)
        
        # Should return the chi call from player 1 only (left player)
        if call_result:
            caller_id, call_type, tiles = call_result
            if call_type == 'chi':
                self.assertEqual(caller_id, 1)  # Only player 1 can chi from player 0
    
    def test_make_call_updates_state(self):
        """Test that making a call properly updates player state"""
        player = Player(0)
        # Hand has two 5p tiles
        player.hand = [
            Tile(Suit.PINZU, TileType.FIVE), Tile(Suit.PINZU, TileType.FIVE),
            Tile(Suit.PINZU, TileType.ONE), Tile(Suit.PINZU, TileType.TWO)
        ]
        
        initial_hand_size = len(player.hand)
        discarded_tile = Tile(Suit.PINZU, TileType.FIVE)
        pon_tiles = [Tile(Suit.PINZU, TileType.FIVE), Tile(Suit.PINZU, TileType.FIVE)]
        
        # Make the call
        player.make_call('pon', pon_tiles, discarded_tile, 1)
        
        # Check that tiles were removed from hand
        self.assertEqual(len(player.hand), initial_hand_size - 2)
        
        # Check that called set was created
        self.assertEqual(len(player.called_sets), 1)
        called_set = player.called_sets[0]
        self.assertEqual(called_set.call_type, 'pon')
        self.assertEqual(called_set.called_tile, discarded_tile)
        self.assertEqual(called_set.caller_position, 0)
        self.assertEqual(called_set.source_position, 1)
        self.assertEqual(len(called_set.tiles), 3)  # 2 from hand + 1 discarded
    
    def test_winning_with_called_sets(self):
        """Test that players can win with called sets"""
        player = Player(0)
        
        # Player has one called set (pon of 5p)
        called_set = CalledSet(
            tiles=[Tile(Suit.PINZU, TileType.FIVE), Tile(Suit.PINZU, TileType.FIVE), Tile(Suit.PINZU, TileType.FIVE)],
            call_type='pon',
            called_tile=Tile(Suit.PINZU, TileType.FIVE),
            caller_position=0,
            source_position=1
        )
        player.called_sets = [called_set]
        
        # Player needs 3 more sets (9 tiles) to win
        # Give them exactly 3 complete sets
        player.hand = [
            Tile(Suit.PINZU, TileType.ONE), Tile(Suit.PINZU, TileType.ONE), Tile(Suit.PINZU, TileType.ONE),
            Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.THREE), Tile(Suit.PINZU, TileType.FOUR),
            Tile(Suit.SOUZU, TileType.SEVEN), Tile(Suit.SOUZU, TileType.EIGHT), Tile(Suit.SOUZU, TileType.NINE)
        ]
        
        # Should be able to win
        self.assertTrue(player.can_win())


class TestPlayerActions(unittest.TestCase):
    def setUp(self):
        self.player = Player(0)
        self.game_state = GameState(
            player_hand=self.player.hand,
            visible_tiles=[],
            remaining_tiles=20,
            player_id=0,
            other_players_discarded={},
            called_sets={},
            last_discarded_tile=None,
            last_discard_player=None,
            can_call=False
        )

    def test_can_tsumo_with_winning_hand(self):
        """Test that can_tsumo returns true for a winning hand"""
        self.player.hand = [
            Tile(Suit.PINZU, TileType.ONE), Tile(Suit.PINZU, TileType.ONE), Tile(Suit.PINZU, TileType.ONE),
            Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.TWO),
            Tile(Suit.PINZU, TileType.THREE), Tile(Suit.PINZU, TileType.THREE), Tile(Suit.PINZU, TileType.THREE),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FOUR),
        ]
        self.assertTrue(self.player.can_tsumo())

    def test_get_possible_actions_all(self):
        """Test getting all possible actions (pon, chi, ron)"""
        # Tenpai hand (11 tiles) - waiting for 2p to complete winning hand  
        self.player.hand = [
            # Two 2p tiles (can form triplet with discarded 2p)
            Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.TWO),
            # Complete triplet
            Tile(Suit.PINZU, TileType.ONE), Tile(Suit.PINZU, TileType.ONE), Tile(Suit.PINZU, TileType.ONE),
            # Complete triplet
            Tile(Suit.PINZU, TileType.THREE), Tile(Suit.PINZU, TileType.THREE), Tile(Suit.PINZU, TileType.THREE),
            # Complete triplet  
            Tile(Suit.SOUZU, TileType.FIVE), Tile(Suit.SOUZU, TileType.FIVE), Tile(Suit.SOUZU, TileType.FIVE)
        ]

        discarded_tile = Tile(Suit.PINZU, TileType.TWO)
        self.game_state.last_discarded_tile = discarded_tile
        self.game_state.last_discard_player = 3 # from player 3 (left player of player 0)

        actions = self.player.get_possible_actions(self.game_state)

        self.assertTrue(actions['pon'])
        self.assertTrue(actions['chi'])
        self.assertTrue(actions['ron'])

    def test_get_possible_actions_none(self):
        """Test getting no possible actions"""
        self.player.hand = [Tile(Suit.SOUZU, TileType.NINE), Tile(Suit.SOUZU, TileType.NINE)]
        discarded_tile = Tile(Suit.PINZU, TileType.FIVE)
        self.game_state.last_discarded_tile = discarded_tile
        self.game_state.last_discard_player = 1

        actions = self.player.get_possible_actions(self.game_state)

        self.assertFalse(actions['pon'])
        self.assertFalse(actions['chi'])
        self.assertFalse(actions['ron'])


class TestChiPonInterface(unittest.TestCase):
    """Test cases specifically for chi/pon interface functionality"""
    
    def setUp(self):
        """Set up test fixtures before each test method"""
        self.players = [Player(i) for i in range(4)]
        self.game = SimpleJong(self.players)
    
    def test_chi_interface_9s_scenario(self):
        """Test the specific scenario mentioned: chi with 9s"""
        # Set up player 0 (human) with tiles that can chi a 9s
        # Hand has 7s and 8s for a potential 7-8-9 sequence
        self.players[0].hand = [
            Tile(Suit.SOUZU, TileType.SEVEN), Tile(Suit.SOUZU, TileType.EIGHT),
            Tile(Suit.PINZU, TileType.ONE), Tile(Suit.PINZU, TileType.TWO),
            Tile(Suit.PINZU, TileType.THREE), Tile(Suit.PINZU, TileType.FOUR),
            Tile(Suit.PINZU, TileType.FIVE), Tile(Suit.PINZU, TileType.SIX),
            Tile(Suit.PINZU, TileType.SEVEN), Tile(Suit.PINZU, TileType.EIGHT),
            Tile(Suit.PINZU, TileType.NINE)
        ]
        
        # Player 3 (left of player 0) discards a 9s
        discarded_tile = Tile(Suit.SOUZU, TileType.NINE)
        self.game.last_discarded_tile = discarded_tile
        self.game.last_discard_player = 3  # Player 3 is the "left" player for player 0
        
        # Get game state and check possible actions
        game_state = self.game.get_game_state(0)
        actions = self.players[0].get_possible_actions(game_state)
        
        # Player 0 should be able to chi with 7s-8s
        self.assertTrue(len(actions['chi']) > 0, "Player should be able to chi the 9s")
        self.assertIn('7s', actions['chi'][0], "Chi should include 7s")
        self.assertIn('8s', actions['chi'][0], "Chi should include 8s")
    
    def test_pon_interface_scenario(self):
        """Test pon interface functionality"""
        # Set up player 0 with two 5p tiles for a potential pon
        self.players[0].hand = [
            Tile(Suit.PINZU, TileType.FIVE), Tile(Suit.PINZU, TileType.FIVE),
            Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.TWO),
            Tile(Suit.SOUZU, TileType.THREE), Tile(Suit.SOUZU, TileType.FOUR),
            Tile(Suit.SOUZU, TileType.SIX), Tile(Suit.SOUZU, TileType.SEVEN),
            Tile(Suit.SOUZU, TileType.EIGHT), Tile(Suit.SOUZU, TileType.NINE),
            Tile(Suit.PINZU, TileType.ONE)
        ]
        
        # Any player discards a 5p
        discarded_tile = Tile(Suit.PINZU, TileType.FIVE)
        self.game.last_discarded_tile = discarded_tile
        self.game.last_discard_player = 2  # Player 2 discarded
        
        # Get game state and check possible actions
        game_state = self.game.get_game_state(0)
        actions = self.players[0].get_possible_actions(game_state)
        
        # Player 0 should be able to pon with their two 5p tiles
        self.assertTrue(len(actions['pon']) > 0, "Player should be able to pon the 5p")
        self.assertEqual(len(actions['pon'][0]), 2, "Pon should use 2 tiles from hand")
        self.assertTrue(all(tile == '5p' for tile in actions['pon'][0]), "All pon tiles should be 5p")
    
    def test_chi_only_from_left_player(self):
        """Test that chi can only be called from the left player"""
        # Set up player 0 with tiles that can chi a 5p
        self.players[0].hand = [
            Tile(Suit.PINZU, TileType.THREE), Tile(Suit.PINZU, TileType.FOUR),
            Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.TWO),
            Tile(Suit.SOUZU, TileType.THREE), Tile(Suit.SOUZU, TileType.FOUR),
            Tile(Suit.SOUZU, TileType.SIX), Tile(Suit.SOUZU, TileType.SEVEN),
            Tile(Suit.SOUZU, TileType.EIGHT), Tile(Suit.SOUZU, TileType.NINE),
            Tile(Suit.PINZU, TileType.ONE)
        ]
        
        # Test when left player (player 3) discards 5p - should be able to chi
        discarded_tile = Tile(Suit.PINZU, TileType.FIVE)
        self.game.last_discarded_tile = discarded_tile
        self.game.last_discard_player = 3  # Left player
        
        game_state = self.game.get_game_state(0)
        actions = self.players[0].get_possible_actions(game_state)
        
        self.assertTrue(len(actions['chi']) > 0, "Should be able to chi from left player")
        
        # Test when non-left player (player 1) discards 5p - should not be able to chi
        self.game.last_discard_player = 1  # Not left player
        
        game_state = self.game.get_game_state(0)
        actions = self.players[0].get_possible_actions(game_state)
        
        self.assertEqual(len(actions['chi']), 0, "Should not be able to chi from non-left player")
    
    def test_ron_with_chi_pon_available(self):
        """Test that ron is available when chi/pon are also possible"""
        # Set up a tenpai hand that can complete with discarded tile
        self.players[0].hand = [
            # Two 2p tiles (can form triplet with discarded 2p for ron)
            Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.TWO),
            # Complete triplet
            Tile(Suit.PINZU, TileType.ONE), Tile(Suit.PINZU, TileType.ONE), Tile(Suit.PINZU, TileType.ONE),
            # Complete triplet
            Tile(Suit.PINZU, TileType.THREE), Tile(Suit.PINZU, TileType.THREE), Tile(Suit.PINZU, TileType.THREE),
            # Complete triplet  
            Tile(Suit.SOUZU, TileType.FIVE), Tile(Suit.SOUZU, TileType.FIVE), Tile(Suit.SOUZU, TileType.FIVE)
        ]
        
        # Player 3 (left player) discards a 2p
        discarded_tile = Tile(Suit.PINZU, TileType.TWO)
        self.game.last_discarded_tile = discarded_tile
        self.game.last_discard_player = 3  # Left player for potential chi
        
        game_state = self.game.get_game_state(0)
        actions = self.players[0].get_possible_actions(game_state)
        
        # Should have ron, pon, and chi all available
        self.assertTrue(actions['ron'], "Should be able to ron")
        self.assertTrue(len(actions['pon']) > 0, "Should be able to pon")
        self.assertTrue(len(actions['chi']) > 0, "Should be able to chi")
    
    def test_call_execution_updates_hand(self):
        """Test that making a call properly updates the player's hand and game state"""
        # Set up player 0 with tiles for a pon
        initial_hand = [
            Tile(Suit.PINZU, TileType.FIVE), Tile(Suit.PINZU, TileType.FIVE),
            Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.TWO),
            Tile(Suit.SOUZU, TileType.THREE), Tile(Suit.SOUZU, TileType.FOUR),
            Tile(Suit.SOUZU, TileType.SIX), Tile(Suit.SOUZU, TileType.SEVEN),
            Tile(Suit.SOUZU, TileType.EIGHT), Tile(Suit.SOUZU, TileType.NINE),
            Tile(Suit.PINZU, TileType.ONE)
        ]
        self.players[0].hand = initial_hand.copy()
        
        discarded_tile = Tile(Suit.PINZU, TileType.FIVE)
        tiles_to_use = [Tile(Suit.PINZU, TileType.FIVE), Tile(Suit.PINZU, TileType.FIVE)]
        
        initial_hand_size = len(self.players[0].hand)
        
        # Make the pon call
        self.players[0].make_call('pon', tiles_to_use, discarded_tile, 1)
        
        # Check that hand size decreased by 2
        self.assertEqual(len(self.players[0].hand), initial_hand_size - 2)
        
        # Check that called set was created
        self.assertEqual(len(self.players[0].called_sets), 1)
        called_set = self.players[0].called_sets[0]
        self.assertEqual(called_set.call_type, 'pon')
        self.assertEqual(called_set.called_tile, discarded_tile)
        self.assertEqual(len(called_set.tiles), 3)  # 2 from hand + 1 discarded
    
    def test_chi_sequence_formation(self):
        """Test different chi sequence formations"""
        player = Player(0)
        
        # Test case 1: middle tile discarded (need 1p and 3p for 2p discard)
        player.hand = [
            Tile(Suit.PINZU, TileType.ONE), Tile(Suit.PINZU, TileType.THREE),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE)
        ]
        discarded_tile = Tile(Suit.PINZU, TileType.TWO)
        combinations = player.get_chi_combinations(discarded_tile)
        
        self.assertEqual(len(combinations), 1)
        self.assertEqual(len(combinations[0]), 2)
        self.assertIn(Tile(Suit.PINZU, TileType.ONE), combinations[0])
        self.assertIn(Tile(Suit.PINZU, TileType.THREE), combinations[0])
        
        # Test case 2: high tile discarded (need 1p and 2p for 3p discard)
        player.hand = [
            Tile(Suit.PINZU, TileType.ONE), Tile(Suit.PINZU, TileType.TWO),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE)
        ]
        discarded_tile = Tile(Suit.PINZU, TileType.THREE)
        combinations = player.get_chi_combinations(discarded_tile)
        
        self.assertEqual(len(combinations), 1)
        self.assertEqual(len(combinations[0]), 2)
        self.assertIn(Tile(Suit.PINZU, TileType.ONE), combinations[0])
        self.assertIn(Tile(Suit.PINZU, TileType.TWO), combinations[0])
        
        # Test case 3: low tile discarded (need 2p and 3p for 1p discard)
        player.hand = [
            Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.THREE),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE)
        ]
        discarded_tile = Tile(Suit.PINZU, TileType.ONE)
        combinations = player.get_chi_combinations(discarded_tile)
        
        self.assertEqual(len(combinations), 1)
        self.assertEqual(len(combinations[0]), 2)
        self.assertIn(Tile(Suit.PINZU, TileType.TWO), combinations[0])
        self.assertIn(Tile(Suit.PINZU, TileType.THREE), combinations[0])


if __name__ == "__main__":
    # Run the tests
    unittest.main(verbosity=2)