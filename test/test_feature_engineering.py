#!/usr/bin/env python3
import os
import sys
import unittest
import numpy as np

# Make src importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.game import SimpleJong, Player  # type: ignore
from core.learn.pure_policy import PurePolicyNetwork, _KerasLikeWrapper  # type: ignore
from core.learn.pure_policy_dataset import serialize_state, extract_indexed_state  # type: ignore


class TestFeatureEngineering(unittest.TestCase):
    def test_hand_info_leak(self):
        # Fix seed for deterministic initial deal
        import random
        random.seed(20250811)

        # Base game
        base_players = [Player(i) for i in range(4)]
        g0 = SimpleJong(base_players)

        # Create three copies and mutate opponents' hands differently
        g1 = g0.copy(); g2 = g0.copy(); g3 = g0.copy()
        # Mutations on P1, P2, P3 respectively
        # Replace their hands with all '1s' (or as many as available) to change content
        from core.game import Tile, Suit, TileType  # type: ignore
        g1._player_hands[1] = [Tile(Suit.SOUZU, TileType.ONE)] * len(g1._player_hands[1])
        g2._player_hands[2] = [Tile(Suit.SOUZU, TileType.TWO)] * len(g2._player_hands[2])
        g3._player_hands[3] = [Tile(Suit.SOUZU, TileType.THREE)] * len(g3._player_hands[3])

        # Prepare network wrapper for feature preprocessing only
        net = PurePolicyNetwork()
        wrap = _KerasLikeWrapper(net)

        def p0_first_turn_features(game: SimpleJong):
            # Ensure it's P0's turn perspective
            game.current_player_idx = 0
            gp = game.get_game_perspective(0)
            sd = serialize_state(gp)
            idx = extract_indexed_state(sd)
            # Build CNN inputs via the same precompute path
            hands = idx['hand_idx'][None, :]
            discs = idx['disc_idx'][None, :, :]
            called = idx.get('called_sets_idx', np.zeros((1, 4, 4, 3), dtype=np.int32))
            gss = idx['game_state'][None, :]
            H, C, D, G = wrap._precompute_cnn(hands, discs, called, gss)
            return H, C, D, G

        f0 = p0_first_turn_features(g0)
        f1 = p0_first_turn_features(g1)
        f2 = p0_first_turn_features(g2)
        f3 = p0_first_turn_features(g3)

        # All features must be exactly equal elementwise
        for a, b in [(f0, f1), (f0, f2), (f0, f3)]:
            for x, y in zip(a, b):
                self.assertTrue(np.array_equal(x, y), "Detected leakage of other players' hand information into P0 features")

    def test_called_sets_and_discards_reflect_caller_player(self):
        # Construct a controlled game where P0 discards 3p
        import random
        random.seed(424242)
        from core.game import Tile, Suit, TileType, Discard, Pon  # type: ignore

        base_players = [Player(i) for i in range(4)]
        g = SimpleJong(base_players)
        # Build hands: ensure P1,P2,P3 each have two 3p and a 4p to discard after pon
        g._player_hands[0] = [Tile(Suit.SOUZU, TileType.ONE)] * 10 + [Tile(Suit.PINZU, TileType.THREE)]
        for pid in [1, 2, 3]:
            g._player_hands[pid] = [Tile(Suit.SOUZU, TileType.ONE)] * 8 + [
                Tile(Suit.PINZU, TileType.THREE), Tile(Suit.PINZU, TileType.THREE), Tile(Suit.PINZU, TileType.FOUR)
            ]
        # No draws to simplify
        g.tiles = []
        g.current_player_idx = 0
        # P0 discards 3p
        self.assertTrue(g.step(0, Discard(Tile(Suit.PINZU, TileType.THREE))))

        # Make three copies and let P1, P2, P3 each call pon then discard 4p
        def apply_pon_and_discard(game: SimpleJong, caller: int) -> SimpleJong:
            game_local = game.copy()
            rs = game_local.get_game_perspective(caller)
            opts = game_local.get_call_options(rs)
            self.assertTrue(len(opts.get('pon', [])) > 0, f"No pon available for P{caller}")
            tiles = opts['pon'][0]
            self.assertTrue(game_local.step(caller, Pon(tiles)))
            # Now discard 4p
            self.assertTrue(game_local.step(caller, Discard(Tile(Suit.PINZU, TileType.FOUR))))
            return game_local

        g_p1 = apply_pon_and_discard(g, 1)
        g_p2 = apply_pon_and_discard(g, 2)
        g_p3 = apply_pon_and_discard(g, 3)

        # Feature extraction for P0 perspective in each scenario
        net = PurePolicyNetwork()
        wrap = _KerasLikeWrapper(net)

        def extract_p0_C_D(game: SimpleJong):
            gp = game.get_game_perspective(0)
            sd = serialize_state(gp)
            idx = extract_indexed_state(sd)
            hands = idx['hand_idx'][None, :]
            discs = idx['disc_idx'][None, :, :]
            called = idx.get('called_sets_idx', np.zeros((1, 4, 4, 3), dtype=np.int32))
            gss = idx['game_state'][None, :]
            H, C, D, G = wrap._precompute_cnn(hands, discs, called, gss)
            return C.copy(), D.copy()

        c1, d1 = extract_p0_C_D(g_p1)
        c2, d2 = extract_p0_C_D(g_p2)
        c3, d3 = extract_p0_C_D(g_p3)

        # Discards sequences must be different across these scenarios
        self.assertFalse(np.array_equal(d1, d2))
        self.assertFalse(np.array_equal(d1, d3))
        self.assertFalse(np.array_equal(d2, d3))
        # Called sets sequences must be different across these scenarios
        self.assertFalse(np.array_equal(c1, c2))
        self.assertFalse(np.array_equal(c1, c3))
        self.assertFalse(np.array_equal(c2, c3))


if __name__ == '__main__':
    unittest.main(verbosity=2)


