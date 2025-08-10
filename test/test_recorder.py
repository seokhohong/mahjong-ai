#!/usr/bin/env python3
import unittest
import sys
import os
import random

# Ensure `src` is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.game import SimpleJong, Action, Reaction  # type: ignore
from core.learn.pure_policy_dataset import Recorder, RecordingPlayer, serialize_action  # type: ignore


class TestRecorder(unittest.TestCase):
    def test_ron_is_last_action_when_present(self):
        # Fixed seed to keep this test reasonably stable; can be changed if flaky
        random.seed(13579)
        for _ in range(20):
            rec = Recorder()
            players = [RecordingPlayer(i, rec) for i in range(4)]
            game = SimpleJong(players)
            game.play_round()
            action_types = []
            phases = []
            for (actor_id, gp, action_obj) in rec.events:
                ad = serialize_action(action_obj)
                action_types.append(ad.get('type'))
                phases.append(gp.state)
            if 'ron' in action_types:
                last_ron_idx = max(i for i, t in enumerate(action_types) if t == 'ron')
                tail_phases = phases[last_ron_idx + 1:]
                self.assertTrue(all(p is Reaction for p in tail_phases),
                                f"Found non-reaction entries after last ron; phases={tail_phases}")
    def test_pass_actions_recorded_when_always_passing(self):
        # Extend the recorder-enabled player to always Pass on reactions
        class AlwaysPassRecordingPlayer(RecordingPlayer):
            def choose_reaction(self, game_state, options):  # type: ignore[override]
                # Force a Pass reaction; base class records automatically
                return super().choose_reaction(game_state, {})

        random.seed(97531)
        any_pass = False
        for _ in range(5):
            rec = Recorder()
            players = [AlwaysPassRecordingPlayer(i, rec) for i in range(4)]
            game = SimpleJong(players)
            game.play_round()
            for (_, gp, action_obj) in rec.events:
                ad = serialize_action(action_obj)
                if ad.get('type') == 'pass':
                    any_pass = True
                    break
            if any_pass:
                break
        self.assertTrue(any_pass, "Expected at least one recorded 'pass' action when players always pass reactions")


if __name__ == '__main__':
    unittest.main(verbosity=2)


