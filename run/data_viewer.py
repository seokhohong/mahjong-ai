#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np

# Allow running from repo root or run/ directory
this_dir = os.path.dirname(__file__)
repo_root = os.path.abspath(os.path.join(this_dir, '..'))
# Support importing both as 'core.*' and 'src.core.*' depending on sys.path
for p in [repo_root, os.path.join(repo_root, 'src')]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from core.learn.pure_policy_dataset import get_action_labels  # type: ignore
except Exception:
    from src.core.learn.pure_policy_dataset import get_action_labels  # type: ignore


def _fmt_tiles_from_indices(indices_row):
    # indices are +1 padded; 0 means PAD
    # We only print non-zero entries as '<rank><suit>' strings
    # Map 0..17 to tile strings: rank = idx//2 + 1, suit: even->p, odd->s
    tile_strs = []
    for v in indices_row:
        if v <= 0:
            continue
        idx = int(v) - 1
        rank = idx // 2 + 1
        suit = 'p' if (idx % 2) == 0 else 's'
        tile_strs.append(f"{rank}{suit}")
    return tile_strs


def _fmt_called_sets(called_sets_tensor):
    # called_sets_tensor shape: (4, MAX_CALLED_SETS_PER_PLAYER, 3)
    out = []
    for row in called_sets_tensor:
        sets = []
        for triplet in row:
            tiles = _fmt_tiles_from_indices(triplet)
            if tiles:
                sets.append(f"[{', '.join(tiles)}]")
        out.append(sets)
    return out


def _render_state(idx_state: dict) -> str:
    hand_idx = idx_state.get('hand_idx')  # (12,)
    disc_idx = idx_state.get('disc_idx')  # (4, T)
    called_sets_idx = idx_state.get('called_sets_idx')  # (4, K, 3)
    game_state = idx_state.get('game_state')  # (GAME_STATE_VEC_LEN,)

    lines = []
    # Hand
    hand_tiles = _fmt_tiles_from_indices(hand_idx)
    lines.append(f"hand: {' '.join(hand_tiles)}")

    # Called sets (rotated by perspective; row 0 is viewer)
    if called_sets_idx is not None:
        csets = _fmt_called_sets(called_sets_idx)
        for i, sets in enumerate(csets):
            if sets:
                lines.append(f"called_sets[row {i}]: {', '.join(sets)}")

    # Basic game flags from game_state vector if available
    # By construction in pure_policy_dataset._encode_game_state_50:
    # [0]=remaining_tiles_norm, [1]=can_call, [2]=len(hand)/MAX_HAND, [3]=your_called_norm, [4]=opponent_sets_norm,
    # [... many features ...], last 4 entries are relative one-hot last_discard_player
    if isinstance(game_state, np.ndarray) and game_state.size >= 8:
        can_call = bool(game_state[1] > 0.5)
        can_tsumo = bool(game_state[6] > 0.5)
        can_ron = bool(game_state[7] > 0.5)
        lines.append(f"flags: can_call={can_call} can_tsumo={can_tsumo} can_ron={can_ron}")

    return '\n'.join(lines)


def _index_to_tile_str(idx: int) -> str:
    rank = idx // 2 + 1
    suit = 'p' if (idx % 2) == 0 else 's'
    return f"{rank}{suit}"


def _prettify_action_label(label: str) -> str:
    try:
        if label.startswith('discard_'):
            i = int(label.split('_', 1)[1])
            return f"discard_{_index_to_tile_str(i)}"
        if label.startswith('pon_'):
            i = int(label.split('_', 1)[1])
            return f"pon_{_index_to_tile_str(i)}"
        if label.startswith('chi_'):
            # chi_{base}_{variant}
            parts = label.split('_')
            if len(parts) >= 3:
                i = int(parts[1])
                variant = parts[2]
                return f"chi_{_index_to_tile_str(i)}_{variant}"
            return label
    except Exception:
        return label
    return label


def main():
    parser = argparse.ArgumentParser(description='View rows from a generated training dataset (.npz)')
    parser.add_argument('npz_path', type=str, help='Path to .npz dataset under training_data/')
    parser.add_argument('-n', '--num', type=int, default=10, help='Number of rows to display')
    parser.add_argument('--start', type=int, default=0, help='Start index (default 0)')
    args = parser.parse_args()

    npz_path = args.npz_path
    if not os.path.isabs(npz_path):
        npz_path = os.path.join(repo_root, npz_path)
    if not os.path.exists(npz_path):
        print(f"Error: file not found: {npz_path}")
        sys.exit(1)

    data = np.load(npz_path, allow_pickle=True)

    states = data['states']
    y_flat = data['y_flat']  # one-hot rows
    game_ids = data['game_ids'] if 'game_ids' in data else None
    step_ids = data['step_ids'] if 'step_ids' in data else None
    action_labels = list(data['action_labels']) if 'action_labels' in data else get_action_labels()

    start = max(0, int(args.start))
    end = min(len(states), start + max(0, int(args.num)))

    for i in range(start, end):
        idx_state = states[i].item() if isinstance(states[i], np.ndarray) else states[i]
        action_idx = int(np.argmax(y_flat[i]))
        label = action_labels[action_idx] if 0 <= action_idx < len(action_labels) else f"idx_{action_idx}"

        print("=" * 60)
        prefix = []
        if game_ids is not None:
            prefix.append(f"game={int(game_ids[i])}")
        if step_ids is not None:
            prefix.append(f"step={int(step_ids[i])}")
        prefix.append(f"row={i}")
        print(" ".join(prefix))
        pretty = _prettify_action_label(label)
        print(f"action: {pretty} (idx={action_idx})")
        print(_render_state(idx_state))


if __name__ == '__main__':
    main()


