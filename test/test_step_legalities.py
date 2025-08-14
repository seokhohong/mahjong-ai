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

from core.game import SimpleJong, Player, Tile, TileType, Discard, Tsumo, Ron, GamePerspective, Suit, Pon, Chi, CalledSet, Action, Reaction, PassCall


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
        for i in range(SimpleJong.NUM_PLAYERS):
            self.assertEqual(len(self.game.hand(i)), 11)
    
    def test_game_state_creation(self):
        """Test that game state is created correctly for each player"""
        game_state = self.game.get_game_perspective(0)
        
        self.assertEqual(game_state.player_id, 0)
        self.assertEqual(len(game_state.player_hand), 11)
        # Remaining tiles should match the engine's internal wall count
        self.assertEqual(game_state.remaining_tiles, len(self.game.tiles))
        self.assertEqual(len(game_state.other_players_discarded), 3)
    
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
        self.game.play_round()
        
        # Game should be over after playing a round
        self.assertTrue(self.game.is_game_over())
        
        # Winners, if any, can be checked after game_over
        winners = self.game.get_winners() if self.game.is_game_over() else []
        for w in winners:
            self.assertIn(w, [0, 1, 2, 3])

    def test_state_copy_roundtrip(self):
        """Copying the game via SimpleJong.copy() preserves state and is independent on mutation."""
        # Prepare a deterministic tiny state
        g = SimpleJong([Player(0), Player(1), Player(2), Player(3)])
        g.tiles = []
        g.current_player_idx = 2
        g._player_hands[0] = [Tile(Suit.PINZU, TileType.ONE)] * 11
        g._player_hands[1] = [Tile(Suit.SOUZU, TileType.TWO)] * 11
        g._player_hands[2] = [Tile(Suit.PINZU, TileType.THREE)] * 11
        g._player_hands[3] = [Tile(Suit.SOUZU, TileType.FOUR)] * 11
        g.player_discards = {i: [] for i in range(4)}
        g.last_discarded_tile = Tile(Suit.PINZU, TileType.FIVE)
        g.last_discard_player = 1
        g.last_drawn_tile = None
        g.last_drawn_player = None

        c = g.copy()
        # Verify basic fields equal
        self.assertEqual(c.current_player_idx, g.current_player_idx)
        self.assertEqual(c.last_discarded_tile, g.last_discarded_tile)
        self.assertEqual(c.last_discard_player, g.last_discard_player)
        for i in range(4):
            self.assertEqual(c.hand(i), g.hand(i))
            self.assertEqual(len(c.called_sets(i)), len(g.called_sets(i)))

        # Mutate original; copy should not change
        g._player_hands[2].pop()
        g.last_discarded_tile = None
        self.assertNotEqual(len(c.hand(2)), len(g.hand(2)))
        self.assertIsNotNone(c.last_discarded_tile)

    def test_action_tsumo_detection(self):
        """Player 0 holds 12 tiles that form four melds; action perspective reports tsumo available."""
        # Compose 4 sequences: 123p, 456p, 789p, 123s (12 tiles)
        tiles = [
            Tile(Suit.PINZU, TileType.ONE), Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.THREE),
            Tile(Suit.PINZU, TileType.FOUR), Tile(Suit.PINZU, TileType.FIVE), Tile(Suit.PINZU, TileType.SIX),
            Tile(Suit.PINZU, TileType.SEVEN), Tile(Suit.PINZU, TileType.EIGHT), Tile(Suit.PINZU, TileType.NINE),
            Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.TWO), Tile(Suit.SOUZU, TileType.THREE),
        ]
        self.game._player_hands[0] = tiles.copy()
        self.game.current_player_idx = 0
        # Indicate last draw belongs to player 0 to flag action state/newly drawn
        self.game.last_drawn_tile = tiles[-1]
        self.game.last_drawn_player = 0

        gp = self.game.get_game_perspective(0)
        self.assertIs(gp.state, Action)
        self.assertIsNotNone(gp.newly_drawn_tile)
        self.assertTrue(gp.can_tsumo())

    def test_reaction_chi_detection_for_left_player(self):
        """With last discard 3p from player 0, player 1 (left) can chi if holding 2p and 4p."""
        self.game.last_discarded_tile = Tile(Suit.PINZU, TileType.THREE)
        self.game.last_discard_player = 0
        # Player 1 holds 2p and 4p enabling chi
        self.game._player_hands[1] = [Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR)]

        rs = self.game.get_game_perspective(1)
        self.assertIs(rs.state, Reaction)
        # Use engine helper to compute calls
        options = self.game.get_call_options(rs)
        self.assertGreaterEqual(len(options['chi']), 1)
        # Ensure non-left player (player 2) cannot chi even with 2p and 4p
        self.game._player_hands[2] = [Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR)]
        rs2 = self.game.get_game_perspective(2)
        options2 = self.game.get_call_options(rs2)
        self.assertEqual(len(options2['chi']), 0)

    def test_reaction_pon_detection(self):
        """Any player may pon if holding two of the discarded tile."""
        self.game.last_discarded_tile = Tile(Suit.SOUZU, TileType.FIVE)
        self.game.last_discard_player = 0
        # Player 2 holds two 5s enabling pon
        self.game._player_hands[2] = [Tile(Suit.SOUZU, TileType.FIVE), Tile(Suit.SOUZU, TileType.FIVE)]
        rs = self.game.get_game_perspective(2)
        options = self.game.get_call_options(rs)
        self.assertGreaterEqual(len(options['pon']), 1)

    def test_reaction_ron_detection(self):
        """Player 1 can ron on 3p if the discard completes four melds."""
        self.game.last_discarded_tile = Tile(Suit.PINZU, TileType.THREE)
        self.game.last_discard_player = 0
        # Hand of 11 tiles: 123s, 456s, 789s, and 2p,4p so that 3p completes 2-3-4p
        self.game._player_hands[1] = [
            Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.TWO), Tile(Suit.SOUZU, TileType.THREE),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE), Tile(Suit.SOUZU, TileType.SIX),
            Tile(Suit.SOUZU, TileType.SEVEN), Tile(Suit.SOUZU, TileType.EIGHT), Tile(Suit.SOUZU, TileType.NINE),
            Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR),
        ]
        rs = self.game.get_game_perspective(1)
        self.assertTrue(rs.can_ron())

    def test_legal_moves_action_phase_for_current_player(self):
        # At start, current player is 0 with 11 tiles; tsumo should be absent, discards present
        moves = self.game.legal_moves(0)
        discard_moves = [m for m in moves if isinstance(m, Discard)]
        tsumo_moves = [m for m in moves if isinstance(m, Tsumo)]
        self.assertEqual(len(discard_moves), len(self.game.hand(0)))
        self.assertEqual(len(tsumo_moves), 0)

    def test_legal_moves_action_phase_others_have_none(self):
        # Non-current players should have no legal moves during action phase
        for pid in [1, 2, 3]:
            self.assertEqual(self.game.legal_moves(pid), [])

    def test_legal_moves_reaction_phase_includes_ron(self):
        # Setup: player 0 discards 3p; player 1 can ron on it
        game = SimpleJong([Player(0), Player(1), Player(2), Player(3)])
        game._player_hands[0] = [Tile(Suit.PINZU, TileType.THREE)] + [Tile(Suit.SOUZU, TileType.ONE)] * 10
        base_s = [
            Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.TWO), Tile(Suit.SOUZU, TileType.THREE),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE), Tile(Suit.SOUZU, TileType.SIX),
            Tile(Suit.SOUZU, TileType.SEVEN), Tile(Suit.SOUZU, TileType.EIGHT), Tile(Suit.SOUZU, TileType.NINE),
        ]
        game._player_hands[1] = base_s + [Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR)]
        game.tiles = []
        game.current_player_idx = 0
        self.assertTrue(game.step(0, Discard(Tile(Suit.PINZU, TileType.THREE))))
        moves_p1 = game.legal_moves(1)
        self.assertTrue(any(isinstance(m, Ron) for m in moves_p1))
        # Discarder cannot react
        self.assertEqual(game.legal_moves(0), [])

    def test_legal_moves_reaction_phase_chi_left_only(self):
        game = SimpleJong([Player(0), Player(1), Player(2), Player(3)])
        game._player_hands[0] = [Tile(Suit.PINZU, TileType.THREE)] + [Tile(Suit.SOUZU, TileType.ONE)] * 10
        # Use souzu filler that cannot form three melds to avoid accidental Ron
        non_partitionable_souzu = [
            Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.TWO),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE),
            Tile(Suit.SOUZU, TileType.SEVEN), Tile(Suit.SOUZU, TileType.SEVEN), Tile(Suit.SOUZU, TileType.EIGHT), Tile(Suit.SOUZU, TileType.NINE)
        ]
        game._player_hands[1] = [Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR)] + non_partitionable_souzu
        game.tiles = []
        game.current_player_idx = 0
        self.assertTrue(game.step(0, Discard(Tile(Suit.PINZU, TileType.THREE))))
        moves_left = game.legal_moves(1)
        self.assertTrue(any(isinstance(m, Chi) and len(m.tiles) == 2 for m in moves_left))
        moves_not_left = game.legal_moves(2)
        self.assertFalse(any(isinstance(m, Chi) for m in moves_not_left))

    def test_legal_moves_reaction_phase_pon_any_player(self):
        game = SimpleJong([Player(0), Player(1), Player(2), Player(3)])
        game._player_hands[0] = [Tile(Suit.SOUZU, TileType.FIVE)] + [Tile(Suit.SOUZU, TileType.ONE)] * 10
        # Ensure player 3 is not in Ron state on 5s; filler cannot form three melds by itself
        non_partitionable_souzu = [
            Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.TWO),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.SIX),
            Tile(Suit.SOUZU, TileType.SEVEN), Tile(Suit.SOUZU, TileType.SEVEN), Tile(Suit.SOUZU, TileType.EIGHT), Tile(Suit.SOUZU, TileType.NINE)
        ]
        game._player_hands[3] = [Tile(Suit.SOUZU, TileType.FIVE), Tile(Suit.SOUZU, TileType.FIVE)] + non_partitionable_souzu
        game.tiles = []
        game.current_player_idx = 0
        self.assertTrue(game.step(0, Discard(Tile(Suit.SOUZU, TileType.FIVE))))
        moves = game.legal_moves(3)
        self.assertTrue(any(isinstance(m, Pon) and len(m.tiles) == 2 for m in moves))

    def test_called_set_recorded_on_chi(self):
        """After a Chi, the called set should be recorded for the caller (player 1)."""
        game = SimpleJong([Player(0), Player(1), Player(2), Player(3)])
        # Player 0 will discard 3p
        game._player_hands[0] = [Tile(Suit.PINZU, TileType.THREE)] + [Tile(Suit.SOUZU, TileType.ONE)] * 10
        # Player 1 can chi with 2p and 4p; filler to avoid ron
        filler = [
            Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.TWO),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE),
            Tile(Suit.SOUZU, TileType.SEVEN), Tile(Suit.SOUZU, TileType.SEVEN), Tile(Suit.SOUZU, TileType.EIGHT), Tile(Suit.SOUZU, TileType.NINE)
        ]
        game._player_hands[1] = [Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR)] + filler
        game.tiles = []
        game.current_player_idx = 0
        # Discard 3p by player 0
        self.assertTrue(game.step(0, Discard(Tile(Suit.PINZU, TileType.THREE))))
        # Player 1 chi on [2p,4p]
        chi_tiles = [Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR)]
        self.assertTrue(game.is_legal(1, Chi(chi_tiles)))
        self.assertTrue(game.step(1, Chi(chi_tiles)))
        # Check called set recorded
        csets = game.called_sets(1)
        self.assertEqual(len(csets), 1)
        cs = csets[0]
        self.assertEqual(cs.call_type, 'chi')
        self.assertEqual(str(cs.called_tile), '3p')
        # Tiles should contain 2p,3p,4p (order not enforced in record)
        tiles_str = sorted(str(t) for t in cs.tiles)
        self.assertEqual(tiles_str, ['2p', '3p', '4p'])
 
 
if __name__ == '__main__':
     unittest.main(verbosity=2)
 
 


