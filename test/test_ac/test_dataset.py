#!/usr/bin/env python3
import unittest
import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))


class TestACDataset(unittest.TestCase):
    def test_non_zero_rewards_in_replay(self):
        # Import builder from the run script without executing CLI
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'run'))
        from create_dataset import build_ac_dataset  # type: ignore

        built = build_ac_dataset(
            games=10,
            seed=123,
            hidden_size=32,
            embedding_dim=4,
            max_turns=20,
            temperature=1.0,
            zero_network_reward=True,  # force network reward zero to test terminal reward injection
            n_step=3,
            gamma=0.99,
        )

        returns = built['returns']
        self.assertGreater(returns.size, 0)
        # There should be some non-zero returns due to overwritten terminal outcomes
        self.assertGreater(float(np.count_nonzero(np.abs(returns) > 1e-8)), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)


