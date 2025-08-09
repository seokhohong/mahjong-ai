#!/usr/bin/env python3
"""
CLI to generate a pure-policy dataset of (state, action, reward) tuples using the simplified engine.

Usage:
  python3 generate_pure_policy_dataset.py --games 100 --seed 42
"""

import sys
import os
import argparse

# Add src to path
SRC_DIR = os.path.join(os.path.dirname(__file__), 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from core.learn.pure_policy_dataset import generate_pure_policy_dataset  # type: ignore


def main() -> int:
    parser = argparse.ArgumentParser(description='Generate pure-policy dataset (.npz) into training_data/')
    parser.add_argument('--games', type=int, default=100, help='Number of games to simulate')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--out', type=str, default=None, help='Optional explicit output path')
    args = parser.parse_args()

    path = generate_pure_policy_dataset(args.games, seed=args.seed, out_path=args.out)
    print(f'Saved dataset to {path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


