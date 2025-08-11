#!/usr/bin/env python3
import unittest
import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.game import SimpleJong
from core.learn.pure_policy_player import PurePolicyPlayer
from core.learn.pure_policy_dataset import serialize_state, serialize_action, encode_action_flat_index


class TestPurePolicyPlayer(unittest.TestCase):
    def test_random_network_can_play_one_game(self):
        # Initialize a small random PurePolicyNetwork if available
        try:
            from core.learn.pure_policy import PurePolicyNetwork  # type: ignore
        except Exception:
            self.skipTest("PurePolicyNetwork unavailable (likely missing torch)")
            return
        net = PurePolicyNetwork(hidden_size=32, embedding_dim=4, max_turns=20)

        # Four players using the same randomly initialized network
        players = [PurePolicyPlayer(i, net) for i in range(4)]
        game = SimpleJong(players, tile_copies=4)

        # Play a single round; ensure no exceptions and game ends
        game.play_round()
        self.assertTrue(game.is_game_over())
        # Winners, if any, are valid ids
        for w in game.get_winners():
            self.assertIn(w, [0, 1, 2, 3])

    def test_move_instantiation(self):
        # Use a dummy network since this mapping test doesn't require inference
        class _DummyModel:
            def predict(self, x, verbose=0):
                raise RuntimeError("predict should not be called in this test")
        class _DummyNet:
            model = _DummyModel()
        players = [PurePolicyPlayer(i, _DummyNet()) for i in range(4)]
        game = SimpleJong(players, tile_copies=4)

        # First actor and perspective
        actor_id = game.current_player_idx
        gs = game.get_game_perspective(actor_id)
        legal_moves = game.legal_moves(actor_id)

        # Get mapping from player's helper
        pairs = players[actor_id]._legal_action_indices_action_phase(gs, legal_moves)
        self.assertGreater(len(pairs), 0)

        # Verify indices equal dataset encoding of the same legal moves
        sd = serialize_state(gs)
        last = sd.get('last_discarded_tile')
        for ai, mv in pairs:
            ad = serialize_action(mv)
            enc = encode_action_flat_index(ad, last)
            self.assertEqual(int(ai), int(enc))


if __name__ == '__main__':
    unittest.main(verbosity=2)


