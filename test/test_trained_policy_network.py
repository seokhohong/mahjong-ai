#!/usr/bin/env python3
import os
import sys
import unittest
import numpy as np

# Make src importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.game import SimpleJong, Player, Tile, TileType, Suit, Discard
from core.learn.pure_policy import PurePolicyNetwork
from core.learn.pure_policy_player import PurePolicyPlayer
from core.learn.pure_policy_dataset import serialize_state, extract_indexed_state, get_action_index_map


class TestTrainedPolicyNetwork(unittest.TestCase):
    MODEL_PATH = os.path.join('training_data', 'models', 'pure_policy_5k.pt')

    def setUp(self):
        if not os.path.exists(self.MODEL_PATH):
            self.skipTest(f"Trained model not found at {self.MODEL_PATH}")

    def _predict_from_game_state(self, net: PurePolicyNetwork, g: SimpleJong, pid: int) -> np.ndarray:
        gs = g.get_game_perspective(pid)
        sd = serialize_state(gs)
        idx = extract_indexed_state(sd)
        hands = idx['hand_idx'][None, :]
        discs = idx['disc_idx'][None, :, :]
        called = idx.get('called_sets_idx', np.zeros((1, 4, 4, 3), dtype=np.int32))
        gss = idx['game_state'][None, :]
        return net.model.predict([hands, discs, called, gss], verbose=0)[0]

    def test_tsumo_selected_on_trivial_tsumo_state(self):
        net = PurePolicyNetwork()
        net.load_model(self.MODEL_PATH)

        # Create a SimpleJong game and set player 0 hand to a complete 12-tile winning hand
        g = SimpleJong([Player(0), Player(1), Player(2), Player(3)])
        g._player_hands[0] = [
            Tile(Suit.PINZU, TileType.ONE), Tile(Suit.PINZU, TileType.ONE), Tile(Suit.PINZU, TileType.ONE),
            Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.TWO),
            Tile(Suit.SOUZU, TileType.THREE), Tile(Suit.SOUZU, TileType.THREE), Tile(Suit.SOUZU, TileType.THREE),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FOUR),
        ]
        g.tiles = []
        g.current_player_idx = 0

        probs = self._predict_from_game_state(net, g, 0)
        amap = get_action_index_map()
        pred_idx = int(np.argmax(probs))
        self.assertEqual(pred_idx, amap['tsumo'])

    def test_ron_selected_on_trivial_ron_state(self):
        net = PurePolicyNetwork()
        net.load_model(self.MODEL_PATH)

        # Create a SimpleJong game where player 1 can Ron on 3p discarded by player 0
        g = SimpleJong([Player(0), Player(1), Player(2), Player(3)])
        # Player 1: nearly complete; wins with 3p
        g._player_hands[1] = [
            Tile(Suit.PINZU, TileType.ONE), Tile(Suit.PINZU, TileType.ONE), Tile(Suit.PINZU, TileType.ONE),
            Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.TWO),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FOUR),
            Tile(Suit.SOUZU, TileType.THREE), Tile(Suit.SOUZU, TileType.THREE),
        ]
        # Player 0 will discard 3p
        g._player_hands[0] = [Tile(Suit.PINZU, TileType.THREE)] + [Tile(Suit.SOUZU, TileType.ONE)] * 10
        g.tiles = []
        g.current_player_idx = 0
        self.assertTrue(g.step(0, Discard(Tile(Suit.PINZU, TileType.THREE))))

        probs = self._predict_from_game_state(net, g, 1)
        amap = get_action_index_map()
        pred_idx = int(np.argmax(probs))
        self.assertEqual(pred_idx, amap["ron"])


    # this test is failing because the network is not predicting legal moves right now
    def test_predicts_legal(self):
        """Ensure that whenever the network is consulted (action or reaction),
        the argmax of its predicted policy corresponds to a legal move index.
        """
        if not os.path.exists(self.MODEL_PATH):
            self.skipTest(f"Trained model not found at {self.MODEL_PATH}")

        net = PurePolicyNetwork()
        net.load_model(self.MODEL_PATH)

        # Define an audited player that asserts legality of the network argmax
        class AuditedPurePolicyPlayer(PurePolicyPlayer):
            def _assert_argmax_legal(self, gs: 'SimpleJong.GamePerspective') -> None:  # type: ignore[name-defined]
                import numpy as _np  # local import to avoid polluting module
                probs = self.predict_policy_probs(gs)
                mask = self._game.legality_mask(self.player_id)  # type: ignore[attr-defined]
                idx = int(_np.argmax(probs))
                assert bool(mask[idx]), f"Masked argmax predicted illegal action index {idx}"

            def play(self, game_state):  # action phase
                self._assert_argmax_legal(game_state)
                return super().play(game_state)

            def choose_reaction(self, game_state, options):  # reaction phase
                self._assert_argmax_legal(game_state)
                return super().choose_reaction(game_state, options)

        # Seat 0 uses the audited pure policy player; others are baseline
        g = SimpleJong([
            AuditedPurePolicyPlayer(0, net),
            Player(1),
            Player(2),
            Player(3),
        ])

        # Play several rounds to exercise both action and reaction consultations
        # (A single round already consults many times; a few repeats adds coverage.)
        for _ in range(3):
            g.play_round()
            # Reinitialize a new game instance with fresh randomness for subsequent iterations
            g = SimpleJong([
                AuditedPurePolicyPlayer(0, net),
                Player(1),
                Player(2),
                Player(3),
            ])

if __name__ == '__main__':
    unittest.main(verbosity=2)


