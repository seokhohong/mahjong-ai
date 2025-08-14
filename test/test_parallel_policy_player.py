#!/usr/bin/env python3
import unittest
import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.game import SimpleJong, Discard  # type: ignore
from core.learn.pure_policy_player import PurePolicyPlayer  # type: ignore
from core.learn.parallel_policy_player import ParallelPolicyPlayer, get_or_create_predictor  # type: ignore


class TestParallelPolicyPlayer(unittest.TestCase):
    def setUp(self):
        try:
            from core.learn.pure_policy import PurePolicyNetwork  # type: ignore
        except Exception:
            self.skipTest("PurePolicyNetwork unavailable (likely missing torch)")
            return
        # Small torch net for speed
        self.net = PurePolicyNetwork(hidden_size=32, embedding_dim=4, max_turns=20)

    def test_policy_probs_match_single_state(self):
        # Create one game state and compare per-state probability heads
        players = [PurePolicyPlayer(i, self.net) for i in range(4)]
        game = SimpleJong(players, tile_copies=4)
        actor_id = game.current_player_idx
        gs = game.get_game_perspective(actor_id)

        base = PurePolicyPlayer(actor_id, self.net)
        par = ParallelPolicyPlayer(actor_id, self.net, max_batch_size=8, max_wait_ms=10)

        p_base = base.predict_policy_probs(gs)
        p_par = par.predict_policy_probs(gs)

        self.assertEqual(p_base.shape, p_par.shape)
        self.assertTrue(np.allclose(p_base, p_par, atol=1e-7))

    def test_decision_equivalence_over_many_states(self):
        # Compare chosen actions across many actionable states within multiple games
        rng = np.random.RandomState(0)
        total_decisions = 0
        same_decisions = 0

        # Share a predictor to exercise batching when tests run in parallelized CI too
        predictor = get_or_create_predictor(self.net, max_batch_size=16, max_wait_ms=5)

        for _ in range(3):
            players = [PurePolicyPlayer(i, self.net) for i in range(4)]
            game = SimpleJong(players, tile_copies=4)

            # Step through the game; at each actionable turn, compare decisions using frozen snapshot
            steps = 0
            while not game.is_game_over() and steps < 200:
                actor_id = game.current_player_idx
                gs = game.get_game_perspective(actor_id)

                base = PurePolicyPlayer(actor_id, self.net)
                par = ParallelPolicyPlayer(actor_id, self.net, predictor=predictor)

                move_base = base.play(gs)
                move_par = par.play(gs)

                total_decisions += 1
                same = type(move_base) is type(move_par)
                if isinstance(move_base, Discard) and isinstance(move_par, Discard):
                    same = same and (move_base.tile == move_par.tile)
                if same:
                    same_decisions += 1

                # Let engine advance exactly one turn
                game.play_turn()
                steps += 1

        self.assertGreater(total_decisions, 0)
        self.assertEqual(same_decisions, total_decisions, f"Mismatch in {total_decisions - same_decisions}/{total_decisions} decisions")


if __name__ == '__main__':
    unittest.main(verbosity=2)


