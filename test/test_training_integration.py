#!/usr/bin/env python3
import os
import sys
import unittest
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.game import SimpleJong, Player  # type: ignore
from core.learn.pure_policy import PurePolicyNetwork  # type: ignore
from core.learn.pure_policy_player import PurePolicyPlayer  # type: ignore
from core.learn.pure_policy_dataset import Recorder, serialize_state, encode_action_flat_index  # type: ignore


class RecordingPurePolicyPlayer(PurePolicyPlayer):
    """Extend PurePolicyPlayer to record actions via provided Recorder.

    The same instance will be attached to all four seats.
    """
    def __init__(self, player_id: int, network: PurePolicyNetwork, recorder: Recorder):
        super().__init__(player_id, network)
        self._recorder = recorder

    def play(self, game_state):  # type: ignore[override]
        move = super().play(game_state)
        if self._recorder is not None:
            # Record the move with the raw GamePerspective for later verification
            self._recorder.record(game_state, self.player_id, move, self.predict_policy_probs(game_state), self._game.legality_mask(self.player_id) if getattr(self, "_game", None) is not None else None)
        return move

    def choose_reaction(self, game_state, options):  # type: ignore[override]
        move = super().choose_reaction(game_state, options)
        if self._recorder is not None:
            self._recorder.record(game_state, self.player_id, move, self.predict_policy_probs(game_state), self._game.legality_mask(self.player_id) if getattr(self, "_game", None) is not None else None)
        return move


class TestTrainingIntegration(unittest.TestCase):
    def test_feature_consistency(self):
        # Tiny network for speed
        import random
        random.seed(0)
        net = PurePolicyNetwork(hidden_size=16, embedding_dim=4)

        # Shared recorder and a single player instance used across all seats
        rec = Recorder()
        players = [RecordingPurePolicyPlayer(i, net, rec) for i in range(4)]

        g = SimpleJong(players)
        g.play_round()

        self.assertGreater(len(rec.events), 0, "should record some events")

        # For each recorded (GamePerspective, action), verify parity
        for idx, compound in enumerate(zip(rec.events, rec.event_probs)):
            tup, event_prob = compound
            actor_id, gp, action_obj = tup
            from core.learn.pure_policy_dataset import serialize_action as _ser_act  # type: ignore
            # Use the player's probability head on the raw GamePerspective. All players should have the same net
            probs = players[0].predict_policy_probs(gp)
            self.assertTrue(np.allclose(probs, players[1].predict_policy_probs(gp)))
            self.assertTrue(np.allclose(probs, event_prob))


if __name__ == '__main__':
    unittest.main(verbosity=2)


