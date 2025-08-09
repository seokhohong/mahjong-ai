#!/usr/bin/env python3
import sys
import os
import argparse
import random

# Ensure src is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from core.game import SimpleJong, Player

# Optional: import the text viewer wrapper for detailed logs
try:
    from core.text_viewer import TextViewerPlayer
except Exception:
    TextViewerPlayer = None  # type: ignore


def run_once(seed: int, use_viewer: bool = True) -> bool:
    """Run a single game with the given seed. Returns True if exception occurred."""
    if seed is not None:
        random.seed(seed)
    lines = []

    if use_viewer and TextViewerPlayer is not None:
        base_players = [Player(i) for i in range(4)]
        players = [TextViewerPlayer(i, base_players[i], lines) for i in range(4)]
    else:
        players = [Player(i) for i in range(4)]

    game = SimpleJong(players)
    try:
        winner = game.play_round()
        print(f"Seed {seed}: winner={winner}, game_over={game.is_game_over()}, winners={game.get_winners()}, loser={game.get_loser()}")
        return False
    except Exception as e:
        print(f"Seed {seed}: Exception during play_round: {repr(e)}")
        if lines:
            print("-- Last 50 log lines --")
            for ln in lines[-50:]:
                print(ln)
        return True


def main():
    parser = argparse.ArgumentParser(description="Reproduce play_round issues across seeds")
    parser.add_argument('--start', type=int, default=0, help='Start seed (inclusive)')
    parser.add_argument('--end', type=int, default=500, help='End seed (exclusive)')
    parser.add_argument('--use_viewer', action='store_true', help='Wrap players with TextViewerPlayer for logs')
    parser.add_argument('--stop_on_first', action='store_true', help='Stop after first exception is found')
    args = parser.parse_args()

    any_fail = False
    for seed in range(args.start, args.end):
        failed = run_once(seed, use_viewer=args.use_viewer)
        if failed:
            any_fail = True
            if args.stop_on_first:
                break
    return 1 if any_fail else 0


if __name__ == '__main__':
    sys.exit(main())
