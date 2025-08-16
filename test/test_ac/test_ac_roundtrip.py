#!/usr/bin/env python3
import unittest
import sys
import os
import random
import numpy as np

from core.learn_ac import ACNetwork

# Ensure both project root and src are on sys.path for mixed absolute/relative imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from core.game import SimpleJong
from core.learn_ac.recording_ac_player import RecordingACPlayer
from core.learn_ac.policy_utils import flat_index_for_action, build_move_from_flat
from core.learn.pure_policy_dataset import serialize_state, serialize_action




class TestACRoundTrip(unittest.TestCase):
    def setUp(self):
        random.seed(123)
        np.random.seed(123)
        net = ACNetwork()
        self.players = [RecordingACPlayer(i, network=net, temperature=1.0) for i in range(4)]
        # Play a single round to collect experiences
        game = SimpleJong(self.players)
        game.play_round()

    def _actions_equal(self, a, b) -> bool:
        if a.__class__ is not b.__class__:
            return False
        name = a.__class__.__name__
        if name == 'Discard':
            return (a.tile.suit == b.tile.suit) and (a.tile.tile_type == b.tile.tile_type)
        if name in ('Pon', 'Chi'):
            at = sorted([(t.suit.value, t.tile_type.value) for t in a.tiles])
            bt = sorted([(t.suit.value, t.tile_type.value) for t in b.tiles])
            return at == bt
        return True

    def test_round_trip_flat_index(self):
        # Verify flat_index_for_action and build_move_from_flat are inverses on recorded experiences
        for p in self.players:
            exp = p.experience
            for state, action in zip(exp.states, exp.actions):
                sd = serialize_state(state)
                ad = serialize_action(action)
                idx = flat_index_for_action(sd, ad)
                # Skip invalid encodings
                if idx < 0:
                    continue
                recon = build_move_from_flat(state, idx)
                self.assertIsNotNone(recon)
                self.assertTrue(self._actions_equal(action, recon))


if __name__ == '__main__':
    unittest.main(verbosity=2)


