#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import argparse
from typing import List, Tuple

import numpy as np

# Ensure project root and src are importable when running as a script/module
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_SRC = os.path.join(_ROOT, 'src')
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from core.learn.pure_policy import PurePolicyNetwork  # type: ignore


def _load_dataset(npz_path: str) -> Tuple[List[dict], np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)
    states = list(data['states'])
    y_flat = np.asarray(data['y_flat'])
    rewards = np.asarray(data['rewards']).astype(np.float32)
    legal_masks = np.asarray(data['legal_masks']).astype(bool) if 'legal_masks' in data else None
    if legal_masks is None:
        raise ValueError('Dataset is missing legal_masks; required to match training metric.')
    return states, y_flat, rewards, legal_masks


def _stack_indexed_states(states: List[dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hands = np.stack([s['hand_idx'] for s in states], axis=0)
    discs = np.stack([s['disc_idx'] for s in states], axis=0)
    # called_sets_idx is present in generated datasets; fallback to zeros if missing
    try:
        called = np.stack([s.get('called_sets_idx') for s in states], axis=0)
    except Exception:
        # Fallback shape inference (N, 4, sets, 3)
        from core.constants import MAX_CALLED_SETS_PER_PLAYER as _MCSP  # type: ignore
        called = np.zeros((len(states), 4, int(_MCSP), 3), dtype=np.int32)
    gss = np.stack([s['game_state'] for s in states], axis=0)
    return hands, discs, called, gss


def evaluate_performance(model_path: str, dataset_path: str, batch_size: int = 512) -> dict:
    states, y_flat, rewards, legal_masks = _load_dataset(dataset_path)
    labels = np.argmax(y_flat, axis=1).astype(np.int64)
    hands, discs, called, gss = _stack_indexed_states(states)

    net = PurePolicyNetwork()
    net.load_model(model_path)

    N = hands.shape[0]
    bs = max(1, min(batch_size, N))

    true_action_probs: List[float] = []
    win_mask_all = (rewards == 1.0)
    lose_mask_all = (rewards == -1.0)
    neutral_mask_all = (rewards == 0.0)

    # Batched inference with legality masking (renormalize over legal actions)
    for start in range(0, N, bs):
        end = min(N, start + bs)
        probs = net.model.predict([
            hands[start:end],
            discs[start:end],
            called[start:end],
            gss[start:end],
        ], verbose=0)
        probs = np.asarray(probs, dtype=np.float64)
        lm = legal_masks[start:end]
        # Zero-out illegal, renormalize across legal slice-wise
        probs[~lm] = 0.0
        row_sums = probs.sum(axis=1, keepdims=True)
        # Avoid division by zero: leave all-zero rows as-is (true action prob will be 0)
        nz_rows = (row_sums[:, 0] > 0)
        if np.any(nz_rows):
            idx_rows = np.where(nz_rows)[0]
            probs[idx_rows] = probs[idx_rows] / row_sums[idx_rows]
        idx = labels[start:end]
        tap = probs[np.arange(end - start), idx]
        true_action_probs.extend(tap.tolist())

    true_action_probs = np.asarray(true_action_probs, dtype=np.float64)

    # Aggregate by reward categories
    win_probs = true_action_probs[win_mask_all]
    lose_probs = true_action_probs[lose_mask_all]
    neutral_probs = true_action_probs[neutral_mask_all]

    avg_win = float(np.mean(win_probs)) if win_probs.size else 0.0
    avg_lose = float(np.mean(lose_probs)) if lose_probs.size else 0.0
    avg_neutral = float(np.mean(neutral_probs)) if neutral_probs.size else 0.0
    performance = avg_win - avg_lose

    return {
        'model': model_path,
        'num_samples': int(N),
        'win_count': int(win_mask_all.sum()),
        'lose_count': int(lose_mask_all.sum()),
        'neutral_count': int(neutral_mask_all.sum()),
        'avg_win_prob': avg_win,
        'avg_lose_prob': avg_lose,
        'avg_neutral_prob': avg_neutral,
        'performance': performance,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate policy-gradient performance metric on a dataset')
    parser.add_argument('--model', '-m', action='append', required=True,
                        help='Path to model .pt file (can be passed multiple times for side-by-side)')
    parser.add_argument('--dataset', '-d', required=True, help='Path to dataset .npz file')
    parser.add_argument('--batch-size', type=int, default=512, help='Inference batch size')
    args = parser.parse_args()

    # Support comma-separated list in a single --model
    model_paths: List[str] = []
    for item in (args.model or []):
        model_paths.extend([p.strip() for p in item.split(',') if p.strip()])
    if not model_paths:
        raise SystemExit('No model paths provided')

    results = []
    for mp in model_paths:
        res = evaluate_performance(mp, args.dataset, batch_size=int(args.batch_size))
        results.append(res)

    # Pretty print concise comparison
    for res in results:
        print(f"Model: {res['model']}")
        print(f"  Samples: {res['num_samples']}  |  Win: {res['win_count']}  Lose: {res['lose_count']}  Neutral: {res['neutral_count']}")
        print(f"  Avg pi(true) -> Win: {res['avg_win_prob']:.4f}  Lose: {res['avg_lose_prob']:.4f}  Neutral: {res['avg_neutral_prob']:.4f}")
        print(f"  Performance (Win - Lose): {res['performance']:.4f}")
        print()


if __name__ == '__main__':
    main()


