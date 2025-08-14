#!/usr/bin/env python3
import unittest
import sys
import os
import numpy as np

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.learn.pure_policy_dataset import generate_pure_policy_dataset
from core.learn.pure_policy import PurePolicyNetwork


class TestPurePolicyTraining(unittest.TestCase):
    def test_train_tiny_network_on_single_game(self):
        # Generate dataset for 1 game
        out_dir = os.path.join(os.path.dirname(__file__), '..', 'training_data')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'pure_policy_test_1game.npz')
        if os.path.exists(out_path):
            os.remove(out_path)

        path = generate_pure_policy_dataset(num_games=1, seed=321, out_path=out_path)
        data = np.load(path, allow_pickle=True)

        states = data['states']       # object array of dicts
        y_flat = data['y_flat']       # (N, num_actions)

        # Sanity checks
        self.assertEqual(states.ndim, 1)
        self.assertGreater(y_flat.shape[1], 60)

        # Train a tiny network quickly
        net = PurePolicyNetwork(hidden_size=8, embedding_dim=4, max_turns=50)

        # Fit a single epoch; tiny data, should run in seconds
        # Build model inputs from indexed states
        hands = []
        discs = []
        called = []
        gss = []
        for s in states:
            s = s.item() if hasattr(s, 'item') else s
            hands.append(s['hand_idx'])
            discs.append(s['disc_idx'])
            from core.constants import MAX_CALLED_SETS_PER_PLAYER as _MCSP  # type: ignore
            called.append(s.get('called_sets_idx', np.zeros((4,_MCSP,3), dtype=np.int32)))
            gss.append(s['game_state'])
        hands = np.asarray(hands, dtype=np.int32)
        discs = np.asarray(discs, dtype=np.int32)
        called = np.asarray(called, dtype=np.int32)
        gss = np.asarray(gss, dtype=np.float32)

        # Provide explicit sample weights (normally rewards). Use ones for this tiny test.
        sample_w = np.ones((states.shape[0],), dtype=np.float32)
        net.model.fit(
            [hands, discs, called, gss],
            {'policy_flat': y_flat},
            epochs=1,
            batch_size=max(1, min(8, states.shape[0])),
            verbose=0,
            sample_weight=sample_w,
            legality_masks=(data['legal_masks'].astype(bool)),
        )

        # Quick forward pass
        outs = net.model.predict([hands, discs, called, gss], verbose=0)
        # Single softmax over flattened actions; outs is (N, num_actions)
        self.assertEqual(outs.ndim, 2)
        self.assertEqual(outs.shape[1], y_flat.shape[1])

        # Save and reload the model; verify prediction compatibility
        model_path = os.path.join(out_dir, 'tiny_test_model.pt')
        try:
            if os.path.exists(model_path):
                os.remove(model_path)
        except Exception:
            pass
        net.save_model(model_path)

        net2 = PurePolicyNetwork(hidden_size=8, embedding_dim=4, max_turns=50)
        net2.load_model(model_path)
        outs2 = net2.model.predict([hands, discs, called, gss], verbose=0)
        self.assertEqual(outs2.shape, outs.shape)
        # Softmax sanity
        self.assertTrue(np.allclose(np.sum(outs2, axis=1), 1.0, atol=1e-5))
        # Check predictions are close to the pre-save outputs
        self.assertTrue(np.allclose(outs2, outs, atol=1e-5))

if __name__ == '__main__':
    unittest.main(verbosity=2)


