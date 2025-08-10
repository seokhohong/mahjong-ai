#!/usr/bin/env python3
import sys
import os
import argparse
import random

# Ensure src is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from core.game import SimpleJong  # type: ignore
from core.learn.pure_policy_dataset import Recorder, RecordingPlayer, serialize_action  # type: ignore



def main():
    random.seed(13579)

    for _ in range(20):
        rec = Recorder()
        players = [RecordingPlayer(i, rec) for i in range(4)]
        game = SimpleJong(players)

        # Play a full round; recorder captures all actions and reactions
        game.play_round()

        # Build action type log for this game
        action_types = []
        for (actor_id, gp, action_obj) in rec.events:
            ad = serialize_action(action_obj)
            action_types.append(ad.get('type'))

        if 'ron' in action_types:
            if action_types[-1] != 'ron':
                print("what")
            # If any ron occurred, it must end the game; thus the last recorded action must be ron
            #self.assertEqual(action_types[-1], 'ron',
            #                 f"Expected last action to be 'ron' when a ron occurs; got {action_types[-1]}")


if __name__ == '__main__':
    sys.exit(main())
