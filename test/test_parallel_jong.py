#!/usr/bin/env python3
import unittest
import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.game import SimpleJong, Player, Discard  # type: ignore
from core.parallel_jong import ParallelJong  # type: ignore
from core.learn.pure_policy_player import PurePolicyPlayer  # type: ignore
from core.learn.parallel_policy_player import ParallelPolicyPlayer, get_or_create_predictor  # type: ignore


class TestParallelJong(unittest.TestCase):
    def _make_games(self):
        try:
            from core.learn.pure_policy import PurePolicyNetwork  # type: ignore
        except Exception:
            self.skipTest("PurePolicyNetwork unavailable (likely missing torch)")
            return []
        net = PurePolicyNetwork(hidden_size=32, embedding_dim=4, max_turns=20)
        predictor = get_or_create_predictor(net, max_batch_size=8, max_wait_ms=5)
        games = []
        # Game 1: all baseline
        games.append(SimpleJong([Player(i) for i in range(4)], tile_copies=4))
        # Game 2: all PurePolicyPlayer (dummy net)
        games.append(SimpleJong([PurePolicyPlayer(i, net) for i in range(4)], tile_copies=4))
        # Game 3: all ParallelPolicyPlayer (shared predictor)
        games.append(SimpleJong([ParallelPolicyPlayer(i, net, predictor=predictor) for i in range(4)], tile_copies=4))
        return games

    def test_runs_three_games_parallel(self):
        games = self._make_games()
        pj = ParallelJong(games, threads=3, progress_desc='UnitTest')
        result_games = pj.run(show_progress=False)
        self.assertEqual(len(result_games), 3)
        for g in result_games:
            self.assertTrue(g.is_game_over())

    def test_mixed_players_decisions_consistency(self):
        # Verify that ParallelPolicyPlayer remains consistent with PurePolicyPlayer on snapshots under actual net
        try:
            from core.learn.pure_policy import PurePolicyNetwork  # type: ignore
        except Exception:
            self.skipTest("PurePolicyNetwork unavailable (likely missing torch)")
            return
        net = PurePolicyNetwork(hidden_size=32, embedding_dim=4, max_turns=20)
        predictor = get_or_create_predictor(net, max_batch_size=8, max_wait_ms=5)
        game = SimpleJong([Player(0), Player(1), Player(2), Player(3)], tile_copies=4)
        actor = game.current_player_idx
        gs = game.get_game_perspective(actor)
        p_base = PurePolicyPlayer(actor, net)
        p_par = ParallelPolicyPlayer(actor, net, predictor=predictor)
        m1 = p_base.play(gs)
        m2 = p_par.play(gs)
        same = type(m1) is type(m2)
        if isinstance(m1, Discard) and isinstance(m2, Discard):
            same = same and (m1.tile == m2.tile)
        self.assertTrue(same)


if __name__ == '__main__':
    unittest.main(verbosity=2)


