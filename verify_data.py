#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import argparse
import glob
from typing import Tuple

import numpy as np


def _find_latest_dataset(patterns: Tuple[str, ...]) -> str:
    candidates: list[str] = []
    for pat in patterns:
        candidates.extend(glob.glob(pat))
    if not candidates:
        raise SystemExit("No dataset files found. Provide --file or generate data under training_data/.")
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def _idx_to_tile(idx0: int) -> str:
    rank = (idx0 // 2) + 1
    suit = 'p' if (idx0 % 2) == 0 else 's'
    return f"{rank}{suit}"


def _format_tiles_from_plus1(indices: np.ndarray) -> str:
    tiles = []
    for v in indices.tolist():
        v = int(v)
        if v <= 0:
            continue
        tiles.append(_idx_to_tile(v - 1))
    return '[' + ', '.join(tiles) + ']'


def _format_called_sets_from_plus1(csets: np.ndarray) -> str:
    parts = []
    # csets shape: (4, MAX_CALLED_SETS_PER_PLAYER, 3)
    for row in range(min(csets.shape[0], 4)):
        row_parts = []
        for i in range(csets.shape[1]):
            trio = csets[row, i]
            trio_tiles = [int(x) for x in trio.tolist() if int(x) > 0]
            if not trio_tiles:
                continue
            row_parts.append('[' + ', '.join(_idx_to_tile(v - 1) for v in trio_tiles) + ']')
        if row_parts:
            prefix = f"P{row}: "
            parts.append(prefix + '; '.join(row_parts))
    return '{' + ' | '.join(parts) + '}'


def _format_discards_from_plus1(disc_idx: np.ndarray) -> str:
    # disc_idx shape: (4, MAX_TURNS)
    parts = []
    for row in range(min(disc_idx.shape[0], 4)):
        tiles = [int(x) for x in disc_idx[row].tolist() if int(x) > 0]
        if not tiles:
            continue
        parts.append(f"P{row}: " + ' '.join(_idx_to_tile(v - 1) for v in tiles))
    return '{' + ' | '.join(parts) + '}'


def verify_dataset(path: str) -> int:
    data = np.load(path, allow_pickle=True)
    y_flat = data["y_flat"]  # shape (N, A)
    legal_masks = data["legal_masks"]  # shape (N, A)
    states = data.get("states", None)
    game_ids = data.get("game_ids", None)
    step_ids = data.get("step_ids", None)
    action_labels = data.get("action_labels", None)

    if y_flat.ndim != 2 or legal_masks.ndim != 2:
        print("Unexpected array shapes for y_flat or legal_masks", file=sys.stderr)
        return 2
    if y_flat.shape != legal_masks.shape:
        print(f"Shape mismatch: y_flat{y_flat.shape} vs legal_masks{legal_masks.shape}", file=sys.stderr)
        return 2

    num_samples, num_actions = y_flat.shape

    # Basic sanity checks
    if "action_labels" in data and data["action_labels"].shape[0] != num_actions:
        print("action_labels length does not match action dimension", file=sys.stderr)
        return 2

    # One-hot enforcement and legality check
    row_sums = y_flat.sum(axis=1)
    # Allow for minor floating error
    one_hot_ok = np.isclose(row_sums, 1.0, atol=1e-6)
    chosen_indices = np.argmax(y_flat, axis=1)
    chosen_legal = legal_masks[np.arange(num_samples), chosen_indices]

    illegal_mask = ~chosen_legal
    not_one_hot_mask = ~one_hot_ok

    num_illegal = int(illegal_mask.sum())
    num_not_one_hot = int(not_one_hot_mask.sum())

    # Also ensure there is at least one legal move available per sample
    has_any_legal = legal_masks.any(axis=1)
    num_no_legal_available = int((~has_any_legal).sum())

    print(f"File: {path}")
    print(f"Samples: {num_samples}")
    print(f"Actions: {num_actions}")
    print(f"Illegal chosen moves: {num_illegal}")
    print(f"Non one-hot rows in y_flat: {num_not_one_hot}")
    print(f"Rows with no legal actions available: {num_no_legal_available}")

    # If there are issues, print a few examples
    def _print_examples(mask: np.ndarray, title: str, limit: int = 10) -> None:
        idxs = np.where(mask)[0][:limit]
        if idxs.size:
            print(f"\n{title} (showing up to {limit}):")
            for i in idxs:
                gi = int(game_ids[i]) if game_ids is not None else -1
                si = int(step_ids[i]) if step_ids is not None else -1
                chosen = int(chosen_indices[i])
                chosen_label = None
                try:
                    if action_labels is not None:
                        chosen_label = str(action_labels[chosen])
                except Exception:
                    chosen_label = None
                print(f"  sample={int(i)} game={gi} step={si} chosen={chosen} ({chosen_label}) row_sum={float(row_sums[i]):.3f}")
                # Print a concise, human-readable state when available
                if states is not None:
                    try:
                        st = states[i].item() if hasattr(states[i], 'item') else states[i]
                        hand_idx = st.get('hand_idx')
                        disc_idx = st.get('disc_idx')
                        called_idx = st.get('called_sets_idx')
                        if hand_idx is not None:
                            print(f"    Hand: {_format_tiles_from_plus1(np.asarray(hand_idx))}")
                        if disc_idx is not None:
                            print(f"    Discards(rotated): {_format_discards_from_plus1(np.asarray(disc_idx))}")
                        if called_idx is not None:
                            print(f"    CalledSets(rotated): {_format_called_sets_from_plus1(np.asarray(called_idx))}")
                        # Show a short list of legal options by index (and labels when available)
                        legal_idxs = np.where(legal_masks[i])[0].tolist()
                        legal_preview = legal_idxs[:15]
                        if action_labels is not None:
                            lbls = [str(action_labels[j]) for j in legal_preview]
                            print(f"    Legal (first {len(legal_preview)}): {list(zip(legal_preview, lbls))} ... total={len(legal_idxs)}")
                        else:
                            print(f"    Legal (first {len(legal_preview)}): {legal_preview} ... total={len(legal_idxs)}")
                    except Exception as e:
                        print(f"    [state formatting error: {e}]")

    _print_examples(illegal_mask, "Illegal chosen moves")
    _print_examples(not_one_hot_mask, "Non one-hot rows")
    _print_examples(~has_any_legal, "No legal actions available")

    # Return non-zero if problems found
    return 0 if (num_illegal == 0 and num_not_one_hot == 0 and num_no_legal_available == 0) else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify that each recorded move in a dataset is legal.")
    parser.add_argument("--file", type=str, default="", help="Path to .npz dataset. If omitted, pick latest under training_data/.")
    args = parser.parse_args()

    path = args.file.strip()
    if not path:
        base = os.path.join(os.path.dirname(__file__), "training_data")
        patterns = (
            os.path.join(base, "pure_policy_pool_*.npz"),
            os.path.join(base, "pure_policy_*.npz"),
        )
        path = _find_latest_dataset(patterns)

    if not os.path.isfile(path):
        raise SystemExit(f"File not found: {path}")

    code = verify_dataset(path)
    if code == 0:
        print("\nVerification PASSED: All chosen actions are legal.")
    else:
        print("\nVerification FAILED: See counts above.")
    sys.exit(code)


if __name__ == "__main__":
    main()


