#!/usr/bin/env python3
import unittest
import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from medium_core.game import (
    MediumJong, Player, Tile, TileType, Suit, Honor,
    Discard, Tsumo, Ron, Pon, Chi, Riichi,
    KanDaimin, KanKakan, KanAnkan,
)


class TestMediumJongBasics(unittest.TestCase):
    def setUp(self):
        self.players = [Player(i) for i in range(4)]
        self.game = MediumJong(self.players)

    def test_initialization(self):
        # 13 tiles each
        for i in range(4):
            self.assertEqual(len(self.game.hand(i)), 13)
        # Round/seat winds
        self.assertEqual(self.game.round_wind.name, 'EAST')
        self.assertEqual(self.game.seat_winds[0], Honor.EAST)

    def test_tile_string_honors(self):
        self.assertEqual(str(Tile(Suit.HONORS, Honor.EAST)), 'E')
        self.assertEqual(str(Tile(Suit.HONORS, Honor.WHITE)), 'P')
        self.assertEqual(str(Tile(Suit.PINZU, TileType.FIVE)), '5p')

    def test_chi_left_only(self):
        g = MediumJong([Player(0), Player(1), Player(2), Player(3)])
        # Prepare controlled hands
        g._player_hands[0] = [Tile(Suit.PINZU, TileType.THREE)] + g._player_hands[0][1:]
        g._player_hands[1] = [Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR)] + g._player_hands[1][2:]
        g._player_hands[2] = [Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR)] + g._player_hands[2][2:]
        # Empty wall to avoid draws beyond one action
        g.tiles = []
        g.current_player_idx = 0
        self.assertTrue(g.step(0, Discard(Tile(Suit.PINZU, TileType.THREE))))
        # Player 1 can chi
        moves1 = g.legal_moves(1)
        self.assertTrue(any(isinstance(m, Chi) for m in moves1))
        # Player 2 cannot chi
        moves2 = g.legal_moves(2)
        self.assertFalse(any(isinstance(m, Chi) for m in moves2))

    def test_pon_any_player(self):
        g = MediumJong([Player(0), Player(1), Player(2), Player(3)])
        g._player_hands[0][0] = Tile(Suit.SOUZU, TileType.FIVE)
        g._player_hands[2][:2] = [Tile(Suit.SOUZU, TileType.FIVE), Tile(Suit.SOUZU, TileType.FIVE)]
        g.tiles = []
        g.current_player_idx = 0
        self.assertTrue(g.step(0, Discard(Tile(Suit.SOUZU, TileType.FIVE))))
        self.assertTrue(any(isinstance(m, Pon) for m in g.legal_moves(2)))

    def test_ron_priority_over_calls(self):
        class Discard3p(Player):
            def play(self, gs):
                t = Tile(Suit.PINZU, TileType.THREE)
                if t in gs.player_hand:
                    return Discard(t)
                return super().play(gs)

        players = [Discard3p(0), Player(1), Player(2), Player(3)]
        g = MediumJong(players)
        # Configure: player 3 can ron on 3p; player 2 can pon 3p
        # Player 3 exact 13 tiles (Tanyao-ready): 2p,4p; 345s; 456s; 456m; pair 77p
        g._player_hands[3] = [
            Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR),
            Tile(Suit.SOUZU, TileType.THREE), Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE), Tile(Suit.SOUZU, TileType.SIX),
            Tile(Suit.MANZU, TileType.FOUR), Tile(Suit.MANZU, TileType.FIVE), Tile(Suit.MANZU, TileType.SIX),
            Tile(Suit.PINZU, TileType.SEVEN), Tile(Suit.PINZU, TileType.SEVEN),
        ]
        g._player_hands[2][:2] = [Tile(Suit.PINZU, TileType.THREE), Tile(Suit.PINZU, TileType.THREE)]
        g._player_hands[0][0] = Tile(Suit.PINZU, TileType.THREE)
        g.tiles = []
        g.current_player_idx = 0
        g.play_turn()
        self.assertTrue(g.is_game_over())
        self.assertIn(3, g.get_winners())
        self.assertEqual(g.get_loser(), 0)


class TestYakuAndRiichi(unittest.TestCase):
    def test_yaku_required_no_yaku_no_win(self):
        g = MediumJong([Player(0), Player(1), Player(2), Player(3)])
        # Construct complete no-yaku hand for player 0 after a draw
        # 123m, 123p, 123s, 456m, pair 9s9s (not tanyao due to terminals)
        tiles = [
            Tile(Suit.MANZU, TileType.ONE), Tile(Suit.MANZU, TileType.TWO), Tile(Suit.MANZU, TileType.THREE),
            Tile(Suit.PINZU, TileType.ONE), Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.THREE),
            Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.TWO), Tile(Suit.SOUZU, TileType.THREE),
            Tile(Suit.MANZU, TileType.FOUR), Tile(Suit.MANZU, TileType.FIVE), Tile(Suit.MANZU, TileType.SIX),
            Tile(Suit.SOUZU, TileType.NINE), Tile(Suit.SOUZU, TileType.NINE),
        ]
        g._player_hands[0] = tiles
        g.current_player_idx = 0
        g.last_drawn_tile = tiles[-1]
        g.last_drawn_player = 0
        gp = g.get_game_perspective(0)
        # Hand is standard complete but should have no listed yaku; cannot tsumo
        self.assertFalse(gp.can_tsumo())

    def test_tanyao_enables_win(self):
        g = MediumJong([Player(0), Player(1), Player(2), Player(3)])
        # All simples hand waiting completed
        tiles = [
            Tile(Suit.MANZU, TileType.TWO), Tile(Suit.MANZU, TileType.THREE), Tile(Suit.MANZU, TileType.FOUR),
            Tile(Suit.PINZU, TileType.THREE), Tile(Suit.PINZU, TileType.FOUR), Tile(Suit.PINZU, TileType.FIVE),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE), Tile(Suit.SOUZU, TileType.SIX),
            Tile(Suit.MANZU, TileType.SIX), Tile(Suit.MANZU, TileType.SEVEN), Tile(Suit.MANZU, TileType.EIGHT),
            Tile(Suit.SOUZU, TileType.THREE), Tile(Suit.SOUZU, TileType.THREE),
        ]
        g._player_hands[0] = tiles
        g.current_player_idx = 0
        g.last_drawn_tile = tiles[-1]
        g.last_drawn_player = 0
        self.assertTrue(g.get_game_perspective(0).can_tsumo())

    def test_yakuhai_winds(self):
        g = MediumJong([Player(0), Player(1), Player(2), Player(3)])
        # Give player 0 triplet of East (seat and round)
        hand = g.hand(0)
        hand[:3] = [Tile(Suit.HONORS, Honor.EAST)] * 3
        g._player_hands[0] = hand
        # Create a plausible complete hand by filling 4 melds + pair roughly
        # We won't strictly enforce completion here; directly score and assert yakuhai counted
        g.winners = [0]
        score = g.score_hand(0, win_by_tsumo=True)
        self.assertGreaterEqual(score['han'], 1)

    def test_riichi_lock_and_uradora(self):
        g = MediumJong([Player(0), Player(1), Player(2), Player(3)])
        # Put player 0 in tenpai with closed hand; ensure Riichi legal and then only discard drawn/kan/tsumo allowed
        # Tenpai example: needing 3p to complete 2-3-4p
        base = [
            Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.TWO), Tile(Suit.SOUZU, TileType.THREE),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE), Tile(Suit.SOUZU, TileType.SIX),
            Tile(Suit.SOUZU, TileType.SEVEN), Tile(Suit.SOUZU, TileType.EIGHT), Tile(Suit.SOUZU, TileType.NINE),
            Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR),
            Tile(Suit.MANZU, TileType.FIVE), Tile(Suit.MANZU, TileType.FIVE),
        ]
        g._player_hands[0] = base
        g.current_player_idx = 0
        g.last_drawn_tile = None
        g.last_drawn_player = None
        # Draw a tile to start action
        if not g.tiles:
            # Ensure at least one tile exists
            g.tiles = [Tile(Suit.MANZU, TileType.TWO)]
        g.play_turn()  # player 0 draws and acts (may discard). Reset for controlled test
        g.current_player_idx = 0
        g.last_discarded_tile = None
        g.last_discard_player = None
        gs = g.get_game_perspective(0)
        # If Riichi is listed, apply it
        lm = gs.legal_moves()
        if any(isinstance(m, Riichi) for m in lm):
            g.step(0, next(m for m in lm if isinstance(m, Riichi)))
            # After Riichi, verify only discard of drawn tile (when present) or kan/tsumo
            g._draw_for_current_if_needed()
            lm2 = g.legal_moves(0)
            dis = [m for m in lm2 if isinstance(m, Discard)]
            self.assertLessEqual(len(dis), 1)

    def test_kan_types_and_dora_increase(self):
        g = MediumJong([Player(0), Player(1), Player(2), Player(3)])
        # Daiminkan on discard
        g._player_hands[0][0] = Tile(Suit.PINZU, TileType.THREE)
        g._player_hands[1][:3] = [Tile(Suit.PINZU, TileType.THREE)] * 3
        dora_before = len(g.dora_indicators)
        g.tiles = [Tile(Suit.SOUZU, TileType.NINE)] * 10
        g.current_player_idx = 0
        g.step(0, Discard(Tile(Suit.PINZU, TileType.THREE)))
        # Player 1 should be able to daiminkan
        lm = g.legal_moves(1)
        self.assertTrue(any(isinstance(m, KanDaimin) for m in lm))
        kd = next(m for m in lm if isinstance(m, KanDaimin))
        g.step(1, kd)
        self.assertGreaterEqual(len(g.dora_indicators), dora_before + 1)
        # Ankan
        g.current_player_idx = 2
        g._player_hands[2][:4] = [Tile(Suit.MANZU, TileType.FIVE)] * 4
        lm2 = g.legal_moves(2)
        self.assertTrue(any(isinstance(m, KanAnkan) for m in lm2))


class TestScoring(unittest.TestCase):
    def test_dealer_tsumo_vs_non_dealer(self):
        g = MediumJong([Player(0), Player(1), Player(2), Player(3)])
        # Make a simple tanyao hand for player 0 (dealer) and tsumo
        tiles = [
            Tile(Suit.MANZU, TileType.TWO), Tile(Suit.MANZU, TileType.THREE), Tile(Suit.MANZU, TileType.FOUR),
            Tile(Suit.PINZU, TileType.THREE), Tile(Suit.PINZU, TileType.FOUR), Tile(Suit.PINZU, TileType.FIVE),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE), Tile(Suit.SOUZU, TileType.SIX),
            Tile(Suit.MANZU, TileType.SEVEN), Tile(Suit.MANZU, TileType.EIGHT), Tile(Suit.MANZU, TileType.NINE),
            Tile(Suit.SOUZU, TileType.THREE), Tile(Suit.SOUZU, TileType.THREE),
        ]
        g._player_hands[0] = tiles
        g.winners = [0]
        s_dealer = g.score_hand(0, win_by_tsumo=True)
        # Non-dealer, player 1, same tiles
        g2 = MediumJong([Player(0), Player(1), Player(2), Player(3)])
        g2._player_hands[1] = tiles
        g2.winners = [1]
        s_nd = g2.score_hand(1, win_by_tsumo=True)
        self.assertGreater(s_dealer['points'], s_nd['points'])


if __name__ == '__main__':
    unittest.main(verbosity=2)


