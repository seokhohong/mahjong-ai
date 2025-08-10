#!/usr/bin/env python3
import unittest
import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.game import SimpleJong
from core.learn.pure_policy import PurePolicyNetwork
from core.learn.pure_policy_player import PurePolicyPlayer


class TestPurePolicyPlayer(unittest.TestCase):
    def test_random_network_can_play_one_game(self):
        # Initialize a small random PurePolicyNetwork
        net = PurePolicyNetwork(hidden_size=32, embedding_dim=4, max_turns=20)

        # Four players using the same randomly initialized network
        players = [PurePolicyPlayer(i, net) for i in range(4)]
        game = SimpleJong(players, tile_copies=4)

        # Play a single round; ensure no exceptions and game ends
        winner = game.play_round()
        self.assertTrue(game.is_game_over())
        # Winner can be None (draw) or 0..3
        if winner is not None:
            self.assertIn(winner, [0, 1, 2, 3])


if __name__ == '__main__':
    unittest.main(verbosity=2)


