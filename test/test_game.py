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
    
        # Check that game is not over initially
        self.assertFalse(self.game.is_game_over())
        self.assertIsNone(self.game.get_winner())
    
    def test_game_state_creation(self):
        """Test that game state is created correctly for each player"""
        game_state = self.game.get_game_perspective(0)
        
        self.assertEqual(game_state.player_id, 0)
        self.assertEqual(len(game_state.player_hand), 11)
        self.assertEqual(game_state.remaining_tiles, 28)
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
        winner = self.game.play_round()
        
        # Game should be over after playing a round
        self.assertTrue(self.game.is_game_over())
        
        # Winner should be None or a valid player ID
        if winner is not None:
            self.assertIn(winner, [0, 1, 2, 3])
            # If there's a winner, they should have a winning hand
            self.assertTrue(self.players[winner].can_win())

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

    def test_play_round_multi_ron_priority(self):
        """If two opponents can ron the same discard, both should win and the game ends immediately."""
        players = [Player(0), Player(1), Player(2), Player(3)]
        game = SimpleJong(players)
        # Current player 0 will discard 3p; configure hands so players 1 and 2 can ron on 3p
        game._player_hands[0] = [Tile(Suit.PINZU, TileType.THREE)] + [Tile(Suit.SOUZU, TileType.ONE)] * 10
        # Players 1 and 2: 11 tiles each that become 4 melds with 3p (2p,4p present)
        base_s = [
            Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.TWO), Tile(Suit.SOUZU, TileType.THREE),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE), Tile(Suit.SOUZU, TileType.SIX),
            Tile(Suit.SOUZU, TileType.SEVEN), Tile(Suit.SOUZU, TileType.EIGHT), Tile(Suit.SOUZU, TileType.NINE),
        ]
        game._player_hands[1] = base_s + [Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR)]
        game._player_hands[2] = base_s + [Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR)]
        game._player_hands[3] = [Tile(Suit.SOUZU, TileType.ONE)] * 11
        game.tiles = []
        game.current_player_idx = 0

        # Force discard 3p by player 0
        gs0 = game.get_game_perspective(0)
        action = Discard(Tile(Suit.PINZU, TileType.THREE))
        game._player_hands[0].remove(action.tile)
        game.player_discards[0].append(action.tile)
        game.last_discarded_tile = action.tile
        game.last_discard_player = 0
        # Now trigger reaction resolution by calling play_round step (no tiles to draw => loop ends after reactions)
        winner = game.play_round()
        winners = game.get_winners()
        self.assertTrue(game.is_game_over())
        self.assertEqual(set(winners), {1, 2})
        self.assertIn(winner, [1, 2])

    def test_play_round_pon_changes_turn(self):
        """Pon should transfer turn to the caller and skip the draw on that next action."""
        class FirstDiscardFiveS(Player):
            def play(self, game_state: GamePerspective):
                t = Tile(Suit.SOUZU, TileType.FIVE)
                if t in game_state.player_hand:
                    return Discard(t)
                return super().play(game_state)

        players = [FirstDiscardFiveS(0), Player(1), Player(2), Player(3)]
        game = SimpleJong(players)
        # Player 0 has 5s to discard; player 2 can pon with two 5s
        game._player_hands[0] = [Tile(Suit.SOUZU, TileType.FIVE)] + [Tile(Suit.SOUZU, TileType.ONE)] * 10
        game._player_hands[1] = [Tile(Suit.SOUZU, TileType.ONE)] * 11
        game._player_hands[2] = [Tile(Suit.SOUZU, TileType.FIVE), Tile(Suit.SOUZU, TileType.FIVE), Tile(Suit.SOUZU, TileType.NINE)] + [Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.TWO)] * 4
        game._player_hands[3] = [Tile(Suit.SOUZU, TileType.ONE)] * 11
        # Prevent further draws to focus on turn change behavior
        game.tiles = [Tile(Suit.SOUZU, TileType.NINE)]
        game.current_player_idx = 0

        # Step one iteration of play_round; after discard, player 2 should pon and become current player
        # We emulate just one loop by calling play_round; with no tiles, it should end quickly after resolution
        game.play_round()
        # After resolution, game ends due to no remaining tiles; ensure last acting player index was updated to pon caller
        # We cannot observe current_player_idx after game_over, but we can ensure that player 2 consumed tiles (hand reduced by 2)
        self.assertEqual(len([t for t in game._player_hands[2] if t.tile_type == TileType.FIVE and t.suit == Suit.SOUZU]), 0)

    def test_play_round_chi_when_no_ron_or_pon(self):
        """Chi should occur if no ron or pon is available for the left player."""
        players = [Player(0), Player(1), Player(2), Player(3)]
        game = SimpleJong(players)
        # Only player 1 (left) can chi; no one can ron or pon
        game._player_hands[0] = [Tile(Suit.SOUZU, TileType.ONE)] * 11
        non_partitionable_souzu = [
            Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.TWO),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE),
            Tile(Suit.SOUZU, TileType.SEVEN), Tile(Suit.SOUZU, TileType.SEVEN), Tile(Suit.SOUZU, TileType.EIGHT), Tile(Suit.SOUZU, TileType.NINE)
        ]
        game._player_hands[1] = [Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR)] + non_partitionable_souzu
        game._player_hands[2] = [Tile(Suit.SOUZU, TileType.ONE)] * 11
        game._player_hands[3] = [Tile(Suit.SOUZU, TileType.ONE)] * 11
        game.tiles = []
        game.last_discarded_tile = Tile(Suit.PINZU, TileType.THREE)
        game.last_discard_player = 0

        game.play_round()
        # Player 1 should have consumed 2p and 4p due to chi call
        self.assertNotIn(Tile(Suit.PINZU, TileType.TWO), game._player_hands[1])
        self.assertNotIn(Tile(Suit.PINZU, TileType.FOUR), game._player_hands[1])

    def test_reaction_priority_ron_over_pon_and_chi(self):
        """When chi, pon, and ron are all available, ron must occur and chi/pon must not."""
        players = [Player(0), Player(1), Player(2), Player(3)]
        game = SimpleJong(players)
        # Prepare hands and outstanding discard 3p by player 0
        game._player_hands[0] = [Tile(Suit.SOUZU, TileType.ONE)] * 11
        # Player 1: chi-capable with 2p and 4p
        filler_no_meld = [
            Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.FOUR),
            Tile(Suit.SOUZU, TileType.SIX), Tile(Suit.SOUZU, TileType.EIGHT),
            Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.FOUR),
            Tile(Suit.SOUZU, TileType.SIX), Tile(Suit.SOUZU, TileType.EIGHT),
            Tile(Suit.SOUZU, TileType.TWO),
        ]
        game._player_hands[1] = [Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR)] + filler_no_meld.copy()
        # Player 2: pon-capable with two 3p
        game._player_hands[2] = [Tile(Suit.PINZU, TileType.THREE), Tile(Suit.PINZU, TileType.THREE)] + filler_no_meld.copy()
        # Player 3: ron-capable (needs 3p to complete 2-3-4p), already has 123s,456s,789s and 2p,4p
        base_s = [
            Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.TWO), Tile(Suit.SOUZU, TileType.THREE),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE), Tile(Suit.SOUZU, TileType.SIX),
            Tile(Suit.SOUZU, TileType.SEVEN), Tile(Suit.SOUZU, TileType.EIGHT), Tile(Suit.SOUZU, TileType.NINE),
        ]
        game._player_hands[3] = base_s + [Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR)]
        # No further draws; set outstanding discard from player 0
        game.tiles = []
        game.last_discarded_tile = Tile(Suit.PINZU, TileType.THREE)
        game.last_discard_player = 0

        # Resolve reactions: expect ron by player 3
        winner = game.play_round()
        self.assertTrue(game.is_game_over())
        self.assertEqual(set(game.get_winners()), {3})
        self.assertEqual(winner, 3)
        # Ensure chi/pon did not consume tiles from players 1 and 2
        self.assertIn(Tile(Suit.PINZU, TileType.TWO), game._player_hands[1])
        self.assertIn(Tile(Suit.PINZU, TileType.FOUR), game._player_hands[1])
        self.assertEqual(sum(1 for t in game._player_hands[2] if t.suit == Suit.PINZU and t.tile_type == TileType.THREE), 2)

    def test_deterministic_discard_player_triggers_chi(self):
        """A deterministic discarder (always 3p) should produce a discard that the left player can chi."""

        class TestDiscardPlayer(Player):
            def play(self, game_state: GamePerspective):
                target = Tile(Suit.PINZU, TileType.THREE)
                if target in game_state.player_hand:
                    return Discard(target)
                return super().play(game_state)

        players = [TestDiscardPlayer(0), Player(1), Player(2), Player(3)]
        game = SimpleJong(players)
        # Configure hands: player 0 has 3p; player 1 has 2p and 4p to chi
        game._player_hands[0] = [Tile(Suit.PINZU, TileType.THREE)] + [Tile(Suit.SOUZU, TileType.ONE)] * 10
        game._player_hands[1] = [Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR)] + [Tile(Suit.SOUZU, TileType.ONE)] * 9
        game._player_hands[2] = []
        game._player_hands[3] = []
        game.tiles = []  # prevent draw
        game.current_player_idx = 0

        gs0 = game.get_game_perspective(0)
        action = players[0].play(gs0)
        self.assertIsInstance(action, Discard)
        self.assertEqual(str(action.tile), '3p')

        # Apply minimal discard effects
        game._player_hands[0].remove(action.tile)
        game.player_discards[0].append(action.tile)
        game.last_discarded_tile = action.tile
        game.last_discard_player = 0

        rs = game.get_game_perspective(1)
        opts = game.get_call_options(rs)
        self.assertGreaterEqual(len(opts['chi']), 1)

    def test_loser_recorded_on_single_ron(self):
        """On a Ron, the loser should be the discarder."""
        game = self.game
        # Set last discard 3p by player 0
        game.last_discarded_tile = Tile(Suit.PINZU, TileType.THREE)
        game.last_discard_player = 0
        # Player 1 has 11 tiles that become 4 melds with 3p (2p,4p present)
        base_s = [
            Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.TWO), Tile(Suit.SOUZU, TileType.THREE),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE), Tile(Suit.SOUZU, TileType.SIX),
            Tile(Suit.SOUZU, TileType.SEVEN), Tile(Suit.SOUZU, TileType.EIGHT), Tile(Suit.SOUZU, TileType.NINE),
        ]
        game._player_hands[1] = base_s + [Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR)]
        # Prevent draws; trigger immediate reaction resolution
        game.tiles = []
        winner = game.play_round()
        self.assertTrue(game.is_game_over())
        self.assertEqual(winner, 1)
        self.assertEqual(game.get_winners(), [1])
        self.assertEqual(game.get_loser(), 0)

    def test_loser_recorded_on_multi_ron(self):
        """On multiple Rons, loser remains the single discarder."""
        players = [Player(0), Player(1), Player(2), Player(3)]
        game = SimpleJong(players)
        # Discard 3p by player 0; players 1 and 2 can ron
        game.last_discarded_tile = Tile(Suit.PINZU, TileType.THREE)
        game.last_discard_player = 0
        base_s = [
            Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.TWO), Tile(Suit.SOUZU, TileType.THREE),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE), Tile(Suit.SOUZU, TileType.SIX),
            Tile(Suit.SOUZU, TileType.SEVEN), Tile(Suit.SOUZU, TileType.EIGHT), Tile(Suit.SOUZU, TileType.NINE),
        ]
        game._player_hands[1] = base_s + [Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR)]
        game._player_hands[2] = base_s + [Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR)]
        game._player_hands[3] = [Tile(Suit.SOUZU, TileType.ONE)] * 11
        game.tiles = []
        winner = game.play_round()
        winners = set(game.get_winners())
        self.assertTrue(game.is_game_over())
        self.assertEqual(winners, {1, 2})
        self.assertIn(winner, [1, 2])
        self.assertEqual(game.get_loser(), 0)

    def test_loser_none_on_tsumo(self):
        """On a Tsumo win, loser should remain None."""
        game = self.game
        # Configure player 0 with 12 tiles forming 4 melds (tsumo)
        tiles = [
            Tile(Suit.PINZU, TileType.ONE), Tile(Suit.PINZU, TileType.TWO),
            Tile(Suit.PINZU, TileType.FOUR), Tile(Suit.PINZU, TileType.FIVE), Tile(Suit.PINZU, TileType.SIX),
            Tile(Suit.PINZU, TileType.SEVEN), Tile(Suit.PINZU, TileType.EIGHT), Tile(Suit.PINZU, TileType.NINE),
            Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.TWO), Tile(Suit.SOUZU, TileType.THREE),
        ]
        game._player_hands[0] = tiles.copy()
        game.current_player_idx = 0
        game.tiles = [Tile(Suit.PINZU, TileType.THREE)]
        winner = game.play_round()
        self.assertTrue(game.is_game_over())
        self.assertEqual(winner, 0)
        self.assertEqual(game.get_winners(), [0])
        self.assertIsNone(game.get_loser())

    def test_decline_pon_allows_left_chi(self):
        """If a pon is available but declined, a left player's chosen chi should execute."""
        class ScriptedDiscardPlayer(Player):
            def __init__(self, pid, target: Tile):
                super().__init__(pid)
                self.target = target
            def play(self, game_state: GamePerspective):
                if self.target in game_state.player_hand:
                    return Discard(self.target)
                return super().play(game_state)

        class DeclinePonPlayer(Player):
            def choose_reaction(self, game_state: GamePerspective, options):
                # Decline even if pon available
                return PassCall()

        class AcceptChiPlayer(Player):
            def choose_reaction(self, game_state: GamePerspective, options):
                if options.get('chi'):
                    return Chi(options['chi'][0])
                return PassCall()

        # Seats: 0 (left of 3), 1, 2, 3 (discarder)
        target = Tile(Suit.PINZU, TileType.THREE)
        players = [AcceptChiPlayer(0), DeclinePonPlayer(1), Player(2), ScriptedDiscardPlayer(3, target)]
        game = SimpleJong(players)
        # Configure hands explicitly
        game._player_hands[3] = [target] + [Tile(Suit.SOUZU, TileType.ONE)] * 10
        # Player 0 (left of discarder) can chi with 2p and 4p
        game._player_hands[0] = [Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR)] + [Tile(Suit.SOUZU, TileType.ONE)] * 9
        # Player 1 can pon with two 3p but will decline
        game._player_hands[1] = [Tile(Suit.PINZU, TileType.THREE), Tile(Suit.PINZU, TileType.THREE)] + [Tile(Suit.SOUZU, TileType.ONE)] * 9
        game._player_hands[2] = [Tile(Suit.SOUZU, TileType.ONE)] * 11
        # One tile in deck per preference
        game.tiles = [Tile(Suit.SOUZU, TileType.NINE)]
        game.current_player_idx = 3

        game.play_round()
        # Player 0 should have consumed 2p and 4p due to chi call
        self.assertNotIn(Tile(Suit.PINZU, TileType.TWO), game._player_hands[0])
        self.assertNotIn(Tile(Suit.PINZU, TileType.FOUR), game._player_hands[0])

    def test_all_players_skip_reactions_then_next_draw(self):
        """When all players skip reactions, the game proceeds with next draw and no tiles are consumed for calls."""
        class ScriptedDiscardPlayer(Player):
            def __init__(self, pid, target: Tile):
                super().__init__(pid)
                self.target = target
            def play(self, game_state: GamePerspective):
                if self.target in game_state.player_hand:
                    return Discard(self.target)
                return super().play(game_state)

        class PassAllPlayer(Player):
            def choose_reaction(self, game_state: GamePerspective, options):
                return PassCall()

        target = Tile(Suit.PINZU, TileType.THREE)
        players = [PassAllPlayer(0), PassAllPlayer(1), PassAllPlayer(2), ScriptedDiscardPlayer(3, target)]
        game = SimpleJong(players)
        # Hands: discarder has target; others have potential calls but will pass
        game._player_hands[3] = [target] + [Tile(Suit.SOUZU, TileType.ONE)] * 10
        # Player 0 (left) could chi
        game._player_hands[0] = [Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR)] + [Tile(Suit.SOUZU, TileType.ONE)] * 9
        # Player 1 could pon
        game._player_hands[1] = [Tile(Suit.PINZU, TileType.THREE), Tile(Suit.PINZU, TileType.THREE)] + [Tile(Suit.SOUZU, TileType.ONE)] * 9
        game._player_hands[2] = [Tile(Suit.SOUZU, TileType.ONE)] * 11
        # One tile in deck
        game.tiles = [Tile(Suit.SOUZU, TileType.NINE)]
        game.current_player_idx = 3

        game.play_round()
        # No tiles should have been consumed from players 0 or 1 for calls
        self.assertIn(Tile(Suit.PINZU, TileType.TWO), game._player_hands[0])
        self.assertIn(Tile(Suit.PINZU, TileType.FOUR), game._player_hands[0])
        self.assertEqual(sum(1 for t in game._player_hands[1] if t.suit == Suit.PINZU and t.tile_type == TileType.THREE), 2)

    def test_draw_when_wall_empty_and_no_wins(self):
        """If the wall is empty and no Ron/Tsumo occurs, the round should cleanly end in a draw (game_over)."""
        players = [Player(0), Player(1), Player(2), Player(3)]
        game = SimpleJong(players)
        # Empty the wall and ensure no pending discard
        game.tiles = []
        game.last_discarded_tile = None
        game.last_discard_player = None
        # Run the round; with no tiles and no pending reactions, it should end immediately as draw
        winner = game.play_round()
        self.assertTrue(game.is_game_over())
        self.assertIsNone(winner)
        self.assertEqual(game.get_winners(), [])

if __name__ == "__main__":
    # Run the tests
    unittest.main(verbosity=2)