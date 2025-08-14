#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import argparse
import shutil
from typing import List, Dict, Any, Tuple

import numpy as np


def _list_npz_files(directory: str) -> List[str]:
    files = [f for f in os.listdir(directory) if f.lower().endswith('.npz')]
    # Prefer numeric ordering if filenames are like N.npz; otherwise lexicographic
    def key(name: str) -> Tuple[int, str]:
        stem = os.path.splitext(name)[0]
        try:
            return (int(stem), name)
        except Exception:
            return (1 << 30, name)
    files.sort(key=key)
    return [os.path.join(directory, f) for f in files]


def _equal_action_labels(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape:
        return False
    try:
        # object arrays of strings
        return bool(np.all(a == b))
    except Exception:
        # Fallback to elementwise compare
        return list(a) == list(b)


def merge_shards(session_dir: str, output_path: str) -> str:
    if not os.path.isdir(session_dir):
        raise FileNotFoundError(f"Not a directory: {session_dir}")

    shard_paths = _list_npz_files(session_dir)
    if not shard_paths:
        raise FileNotFoundError(f"No .npz shards found in {session_dir}")

    states_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    rewards_list: List[np.ndarray] = []
    gid_list: List[np.ndarray] = []
    sid_list: List[np.ndarray] = []
    masks_list: List[np.ndarray] = []
    action_labels_ref: np.ndarray | None = None

    for p in shard_paths:
        with np.load(p, allow_pickle=True) as data:
            # Required arrays
            states_list.append(np.asarray(data['states'], dtype=object))
            y_list.append(np.asarray(data['y_flat'], dtype=np.float32))
            rewards_list.append(np.asarray(data['rewards'], dtype=np.float32))
            gid_list.append(np.asarray(data['game_ids'], dtype=np.int32))
            sid_list.append(np.asarray(data['step_ids'], dtype=np.int32))
            masks_list.append(np.asarray(data['legal_masks'], dtype=bool))
            # Optional but expected
            labels = np.asarray(data['action_labels'], dtype=object)
            if action_labels_ref is None:
                action_labels_ref = labels
            else:
                if not _equal_action_labels(action_labels_ref, labels):
                    raise ValueError(f"action_labels mismatch in shard: {p}")

    # Concatenate along first dimension
    states = np.concatenate(states_list, axis=0)
    y_flat = np.concatenate(y_list, axis=0)
    rewards = np.concatenate(rewards_list, axis=0)
    game_ids = np.concatenate(gid_list, axis=0)
    step_ids = np.concatenate(sid_list, axis=0)
    legal_masks = np.concatenate(masks_list, axis=0)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez(
        output_path,
        states=states,
        y_flat=y_flat,
        rewards=rewards,
        game_ids=game_ids,
        step_ids=step_ids,
        action_labels=(action_labels_ref if action_labels_ref is not None else np.array([], dtype=object)),
        legal_masks=legal_masks,
    )

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Merge per-process training shards into a single .npz and remove the shard directory')
    parser.add_argument('session_dir', type=str, help='Path to the timestamped shard directory (e.g., training_data/20250101_120000)')
    parser.add_argument('--output', type=str, default=None, help='Output .npz path (default: parent_dir/basename(session_dir).npz)')
    parser.add_argument('--keep', action='store_true', help='Keep the shard directory instead of deleting it')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    session_dir = os.path.abspath(args.session_dir)
    parent_dir = os.path.dirname(session_dir)
    base = os.path.basename(session_dir.rstrip(os.sep))
    output_path = args.output or os.path.join(parent_dir, f"{base}.npz")

    print(f"Merging shards from: {session_dir}")
    merged_path = merge_shards(session_dir, output_path)
    print(f"Merged file written to: {merged_path}")

    if not args.keep:
        print(f"Removing shard directory: {session_dir}")
        shutil.rmtree(session_dir)
    else:
        print("--keep specified; shard directory retained")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())


