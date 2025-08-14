#!/usr/bin/env python3
import unittest
import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.game import SimpleJong, Player, Tile, TileType, Suit, Discard
from core.game import CalledSet
from core.game import Ron, Pon, Chi
from core.game import Tsumo


class NotConsultedPlayer(Player):
    def __init__(self, player_id: int):
        super().__init__(player_id)
        self.consulted = False

    def choose_reaction(self, game_state, options):
        # If this gets called, mark and raise to fail fast
        self.consulted = True
        raise AssertionError("Player without legal options was consulted for reaction")


class TestReactionConsulting(unittest.TestCase):
    def test_skip_players_with_no_legal_reaction(self):
        # Construct players where only left player (1) can chi on a 3p discard from player 0
        p0 = Player(0)
        p1 = Player(1)          # Will be able to chi
        p2 = NotConsultedPlayer(2)  # Must not be consulted
        p3 = NotConsultedPlayer(3)  # Must not be consulted

        g = SimpleJong([p0, p1, p2, p3])
        # Setup: 0 discards 3p. Only player 1 has 2p and 4p to chi.
        g._player_hands[0] = [Tile(Suit.PINZU, TileType.THREE)] + [Tile(Suit.SOUZU, TileType.ONE)] * 10
        g._player_hands[1] = [Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR)] + [Tile(Suit.SOUZU, TileType.ONE)] * 9
        g._player_hands[2] = [Tile(Suit.SOUZU, TileType.ONE)] * 11
        g._player_hands[3] = [Tile(Suit.SOUZU, TileType.ONE)] * 11
        g.tiles = []
        g.current_player_idx = 0

        # Apply discard
        self.assertTrue(g.is_legal(0, Discard(Tile(Suit.PINZU, TileType.THREE))))
        g.step(0, Discard(Tile(Suit.PINZU, TileType.THREE)))

        # Resolve reactions; should choose chi from player 1; players 2 and 3 should not be consulted
        g._solicit_and_apply_reactions()
        g.check_game_over()

        self.assertFalse(p2.consulted)
        self.assertFalse(p3.consulted)

    def test_ron_priority_with_three_called_sets(self):
        # Player 1 discards a tile; player 2 has three called sets already (so exactly 2 tiles in hand)
        # and the discard completes the fourth meld. The engine should recognize Ron and not Chi.
        players = [Player(0), Player(1), Player(2), Player(3)]
        g = SimpleJong(players)
        # Clear draws to control state
        g.tiles = []
        # Called sets for player 2: three completed melds (e.g., 123s, 456s, 789s)
        cs1 = CalledSet([
            Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.TWO), Tile(Suit.SOUZU, TileType.THREE)
        ], 'chi', Tile(Suit.SOUZU, TileType.TWO), caller_position=2, source_position=1)
        cs2 = CalledSet([
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE), Tile(Suit.SOUZU, TileType.SIX)
        ], 'chi', Tile(Suit.SOUZU, TileType.FIVE), caller_position=2, source_position=0)
        cs3 = CalledSet([
            Tile(Suit.SOUZU, TileType.SEVEN), Tile(Suit.SOUZU, TileType.EIGHT), Tile(Suit.SOUZU, TileType.NINE)
        ], 'chi', Tile(Suit.SOUZU, TileType.EIGHT), caller_position=2, source_position=3)
        g._player_called_sets[2] = [cs1, cs2, cs3]
        # Player 2's concealed hand has exactly two tiles that are 2p and 4p, waiting on 3p to complete 2-3-4p as the fourth meld
        g._player_hands[2] = [Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR)]
        # Discarder is player 1 (left of player 2 is player 3). Player 1 discards 3p which should allow Ron for player 2.
        g.current_player_idx = 1
        discard_tile = Tile(Suit.PINZU, TileType.THREE)
        # Ensure player 1 holds the discard and can discard it
        g._player_hands[1] = [discard_tile] + [Tile(Suit.SOUZU, TileType.ONE)] * 10
        self.assertTrue(g.is_legal(1, Discard(discard_tile)))
        g.step(1, Discard(discard_tile))

        # Player 2 reaction perspective should allow Ron
        rs2 = g.get_game_perspective(2)
        self.assertTrue(rs2.can_ron())

        # Legal moves for player 2 should include only Ron (no Pon/Chi)
        moves = g.legal_moves(2)
        self.assertTrue(any(isinstance(m, Ron) for m in moves))
        self.assertFalse(any(isinstance(m, Pon) for m in moves))
        self.assertFalse(any(isinstance(m, Chi) for m in moves))

        # Resolve reactions via the engine priority (Ron should win immediately)
        g._solicit_and_apply_reactions()
        g.check_game_over()
        self.assertTrue(g.is_game_over())
        self.assertIn(2, g.get_winners())
        self.assertEqual(g.get_loser(), 1)

    def test_tsumo_with_two_called_sets_and_drawn_completion(self):
        # Player 0 has two called sets already and draws a tile that completes two additional melds
        players = [Player(0), Player(1), Player(2), Player(3)]
        g = SimpleJong(players)
        # Prevent further draws
        g.tiles = []
        # Two called sets for player 0: 123s and 456s
        cs1 = CalledSet([
            Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.TWO), Tile(Suit.SOUZU, TileType.THREE)
        ], 'chi', Tile(Suit.SOUZU, TileType.TWO), caller_position=0, source_position=1)
        cs2 = CalledSet([
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE), Tile(Suit.SOUZU, TileType.SIX)
        ], 'chi', Tile(Suit.SOUZU, TileType.FIVE), caller_position=0, source_position=2)
        g._player_called_sets[0] = [cs1, cs2]
        # Concealed tiles before draw would be 1p,2p,7p,8p,9p. After draw, 3p completes 123p and 789p.
        draw_tile = Tile(Suit.PINZU, TileType.THREE)
        g._player_hands[0] = [
            Tile(Suit.PINZU, TileType.ONE), Tile(Suit.PINZU, TileType.TWO),
            Tile(Suit.PINZU, TileType.SEVEN), Tile(Suit.PINZU, TileType.EIGHT), Tile(Suit.PINZU, TileType.NINE),
            draw_tile,
        ]
        g.current_player_idx = 0
        g.last_drawn_tile = draw_tile
        g.last_drawn_player = 0

        # Legal moves should include Tsumo and must not include Pon/Chi
        moves = g.legal_moves(0)
        self.assertTrue(any(isinstance(m, Tsumo) for m in moves))
        self.assertFalse(any(isinstance(m, Pon) for m in moves))
        self.assertFalse(any(isinstance(m, Chi) for m in moves))

    def test_ron_with_two_called_sets_on_discard_completion(self):
        # Player 2 has two called sets and can ron on an opponent's discard that completes two additional melds
        players = [Player(0), Player(1), Player(2), Player(3)]
        g = SimpleJong(players)
        g.tiles = []
        # Two called sets for player 2: 123s and 456s
        cs1 = CalledSet([
            Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.TWO), Tile(Suit.SOUZU, TileType.THREE)
        ], 'chi', Tile(Suit.SOUZU, TileType.TWO), caller_position=2, source_position=0)
        cs2 = CalledSet([
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE), Tile(Suit.SOUZU, TileType.SIX)
        ], 'chi', Tile(Suit.SOUZU, TileType.FIVE), caller_position=2, source_position=1)
        g._player_called_sets[2] = [cs1, cs2]
        # Concealed tiles: 1p,2p,7p,8p,9p; a discard of 3p completes 123p and 789p
        g._player_hands[2] = [
            Tile(Suit.PINZU, TileType.ONE), Tile(Suit.PINZU, TileType.TWO),
            Tile(Suit.PINZU, TileType.SEVEN), Tile(Suit.PINZU, TileType.EIGHT), Tile(Suit.PINZU, TileType.NINE),
        ]
        # Simulate a real discard of 3p by player 1 to enter reaction phase for player 2
        g._player_hands[1] = [Tile(Suit.PINZU, TileType.THREE)] + [Tile(Suit.SOUZU, TileType.ONE)] * 10
        g.tiles = []
        g.current_player_idx = 1
        self.assertTrue(g.step(1, Discard(Tile(Suit.PINZU, TileType.THREE))))

        # Legal moves for player 2 should include Ron and not Pon/Chi
        moves = g.legal_moves(2)
        self.assertTrue(any(isinstance(m, Ron) for m in moves))
        self.assertFalse(any(isinstance(m, Pon) for m in moves))
        self.assertFalse(any(isinstance(m, Chi) for m in moves))


if __name__ == '__main__':
    unittest.main(verbosity=2)


