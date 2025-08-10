#!/usr/bin/env python3
import os
import sys
import unittest
import numpy as np

# Make src importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.game import SimpleJong, Player, Tile, TileType, Suit, Discard
from core.learn.pure_policy import PurePolicyNetwork
from core.learn.pure_policy_dataset import serialize_state, extract_indexed_state, get_action_index_map


class TestTrainedPolicyNetwork(unittest.TestCase):
    MODEL_PATH = os.path.join('training_data', 'models', 'pure_policy_gen0.pt')

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


if __name__ == '__main__':
    unittest.main(verbosity=2)


