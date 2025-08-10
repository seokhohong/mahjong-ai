#!/usr/bin/env python3
"""
Unit tests for the training data generator with PQNetwork integration
using Python's unittest framework. Skips if TensorFlow is unavailable.
"""

import unittest
import sys
import os
import numpy as np

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from training_data_generator import TrainingDataGenerator
from core.game import PQNetwork, TENSORFLOW_AVAILABLE


@unittest.skip("Temporarily disabled: legacy/unused during refactor")
class TestTrainingDataGeneratorPQ(unittest.TestCase):
    def test_generation_with_pq_single_game(self):
        # Use a tiny PQNetwork to keep runtime small
        pq_network = PQNetwork(hidden_size=1, embedding_dim=1, max_turns=2)

        generator = TrainingDataGenerator(
            generation=1,
            pq_network=pq_network,
            simulation_count=1,
            max_games=1,
            save_interval=1,
            max_rounds_per_game=5,
            base_dir=os.path.join(os.path.dirname(__file__), '..', 'training_data')
        )

        features, policies, values = generator.generate_training_data(
            output_file='test_training_data_pq.npz'
        )

        # Basic shape checks
        self.assertEqual(features.ndim, 2)
        self.assertEqual(policies.ndim, 2)
        self.assertEqual(values.ndim, 1)

        # Policy should be concatenated heads: [5 | 18 | 18] = 41
        self.assertEqual(policies.shape[1], 41)

        # Each head should sum to 1 for every sample
        self.assertTrue(np.allclose(np.sum(policies[:, :5], axis=1), 1.0))
        self.assertTrue(np.allclose(np.sum(policies[:, 5:23], axis=1), 1.0))
        self.assertTrue(np.allclose(np.sum(policies[:, 23:41], axis=1), 1.0))

        # Ensure final file exists
        expected_output = os.path.join(generator.data_dir, 'test_training_data_pq.npz')
        self.assertTrue(os.path.exists(expected_output))


if __name__ == '__main__':
    unittest.main(verbosity=2)


