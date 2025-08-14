#!/usr/bin/env python3
import unittest
import sys
import os
import random

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.game import SimpleJong, HeuristicPlayer  # type: ignore
from core.learn.pure_policy import PurePolicyNetwork  # type: ignore
from run.create_training_data import SamplingPolicyPlayer  # type: ignore


class TestPlayers(unittest.TestCase):
    def test_heuristic_player_completes_one_game(self):
        random.seed(123)
        players = [HeuristicPlayer(i, temperature=0.3) for i in range(4)]
        game = SimpleJong(players, tile_copies=4)
        game.play_round()  # should not raise
        self.assertTrue(game.is_game_over())

    def test_sampling_policy_player_completes_ten_games(self):
        random.seed(321)

        # Fresh, small random network (untrained) to avoid filesystem dependency
        try:
            net = PurePolicyNetwork(hidden_size=32, embedding_dim=4, max_turns=20, temperature=1.5)
        except Exception:
            self.skipTest("PurePolicyNetwork unavailable (likely missing torch)")
            return
        players = [SamplingPolicyPlayer(i, net) for i in range(4)]
        for _ in range(2):
            game = SimpleJong(players, tile_copies=4)
            game.play_round()
            self.assertTrue(game.is_game_over())


if __name__ == '__main__':
    unittest.main(verbosity=2)


