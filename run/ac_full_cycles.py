#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import argparse
from typing import Optional

# Ensure src on path for run/* imports that expect it
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def create_dataset(games: int, model_path: Optional[str]) -> str:
    """Call the dataset builder and return the saved .npz path."""
    from run.create_dataset import build_ac_dataset  # type: ignore
    built = build_ac_dataset(
        games=games,
        seed=None,
        use_heuristic=bool(not model_path),
        model_path=model_path,
    )
    # Write to default location via the existing CLI main to maintain format
    # Reuse run/create_dataset saving logic by calling its main path
    # But since build_ac_dataset returns dict, save here to a timestamped file to avoid duplication
    import numpy as np
    import time
    base_dir = 'training_data'
    os.makedirs(base_dir, exist_ok=True)
    ts = time.strftime('%Y%m%d_%H%M%S')
    out_path = os.path.join(base_dir, f'ac_{ts}.npz')
    np.savez(
        out_path,
        states=built['states'],
        actions=built['actions'],
        returns=built['returns'],
        advantages=built['advantages'],
        old_log_probs=built['old_log_probs'],
        game_ids=built['game_ids'],
        step_ids=built['step_ids'],
        actor_ids=built['actor_ids'],
        flat_policies=built.get('flat_policies', []),
    )
    print(f"Saved AC dataset to {out_path}")
    return out_path


def train_from_dataset(npz_path: str, init_model: Optional[str]) -> str:
    """Call the AC trainer and return the saved model path."""
    from run.train_ac import train_ppo  # type: ignore
    final = train_ppo(
        dataset_path=npz_path,
        epochs=100,
        batch_size=1024,
        lr=3e-4,
        epsilon=0.2,
        value_coeff=0.5,
        entropy_coeff=0.01,
        device=None,
        patience=5,
        val_split=0.1,
        init_model=init_model,
        warm_up_acc=0.7,
        warm_up_max_epochs=20,
    )
    return final


def main():
    ap = argparse.ArgumentParser(description='Alternate AC dataset creation and training cycles.')
    ap.add_argument('--games', type=int, default=5000, help='Games per dataset cycle')
    ap.add_argument('--cycles', type=int, default=10, help='Number of self-play + training cycles')
    ap.add_argument('--init', type=str, default='', help='Initial model to bootstrap first self-play (optional)')
    args = ap.parse_args()

    current_model: Optional[str] = args.init or None
    for c in range(int(max(1, args.cycles))):
        print(f"=== Cycle {c+1}/{args.cycles} ===")
        # 1) Create dataset via self-play using current_model
        ds_path = create_dataset(games=int(max(1, args.games)), model_path=current_model)
        # 2) Train a new model from the dataset (warm-up if needed), initialize from current_model
        new_model = train_from_dataset(ds_path, init_model=current_model)
        current_model = new_model
        print(f"Cycle {c+1} complete. New model: {current_model}")

    print(f"All cycles complete. Final model: {current_model}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())



