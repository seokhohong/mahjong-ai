from __future__ import annotations

import os
from typing import Dict, Any, Tuple, Optional

import numpy as np

try:
    import torch  # type: ignore
    TORCH_AVAILABLE = True
except Exception as e:  # pragma: no cover
    TORCH_AVAILABLE = False
    raise ImportError('PyTorch is required for training. Please install torch.') from e

from src.core.learn.rule_copy_network import RuleCopyNetwork


def _batch_from_states(states_obj: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hands = []
    discs = []
    called = []
    gss = []
    for s in states_obj:
        s = s.item() if hasattr(s, 'item') else s
        hands.append(s['hand_idx'])
        discs.append(s['disc_idx'])
        try:
            from src.core.constants import MAX_CALLED_SETS_PER_PLAYER as _MCSP
        except Exception:
            _MCSP = 3
        called.append(s.get('called_sets_idx', np.zeros((4, _MCSP, 3), dtype=np.int32)))
        gss.append(s['game_state'])
    hands = np.asarray(hands, dtype=np.int32)
    discs = np.asarray(discs, dtype=np.int32)
    called = np.asarray(called, dtype=np.int32)
    gss = np.asarray(gss, dtype=np.float32)
    return hands, discs, called, gss


def train_rule_copy(
    dataset_path: str,
    model_out: str,
    hidden_size: int = 128,
    embedding_dim: int = 4,
    max_turns: int = 50,
    epochs: int = 3,
    batch_size: int = 128,
    verbose: int = 1,
    early_stopping_patience: int = 5,
) -> str:
    """Train RuleCopyNetwork with unweighted cross-entropy to imitate labels.

    - Loads dataset from generate_pure_policy_dataset
    - Ignores rewards; uses legality masks to mask logits
    - Saves a .pt checkpoint compatible with players using RuleCopyNetwork
    """
    if not dataset_path.endswith('.npz'):
        raise ValueError('dataset_path must be a .npz file produced by generate_pure_policy_dataset')
    data = np.load(dataset_path, allow_pickle=True)

    states = data['states']
    y_flat = data['y_flat']
    legal_masks = data['legal_masks'].astype(np.bool_) if 'legal_masks' in data.files else None
    game_ids = data['game_ids'].astype(np.int64) if 'game_ids' in data.files else None

    # Build inputs
    hands, discs, called, gss = _batch_from_states(states)

    # Holdout split by games
    num_samples = hands.shape[0]
    if game_ids is None:
        split_idx = max(1, int(0.8 * num_samples))
        train_idx = np.arange(0, split_idx)
        hold_idx = np.arange(split_idx, num_samples)
    else:
        unique_games = np.unique(game_ids)
        rng = np.random.RandomState(42)
        rng.shuffle(unique_games)
        split_g = max(1, int(0.8 * len(unique_games)))
        train_games = set(unique_games[:split_g].tolist())
        hold_games = set(unique_games[split_g:].tolist())
        train_mask = np.array([gid in train_games for gid in game_ids], dtype=bool)
        hold_mask = np.array([gid in hold_games for gid in game_ids], dtype=bool)
        train_idx = np.where(train_mask)[0]
        hold_idx = np.where(hold_mask)[0]

    hands_tr, discs_tr, called_tr, gss_tr = hands[train_idx], discs[train_idx], called[train_idx], gss[train_idx]
    y_tr = y_flat[train_idx]
    lm_tr = legal_masks[train_idx] if legal_masks is not None else None
    hands_ho, discs_ho, called_ho, gss_ho = hands[hold_idx], discs[hold_idx], called[hold_idx], gss[hold_idx]
    y_ho = y_flat[hold_idx]
    lm_ho = legal_masks[hold_idx] if legal_masks is not None else None

    net = RuleCopyNetwork(hidden_size=hidden_size, embedding_dim=embedding_dim, max_turns=max_turns)

    # Device selection similar to policy_trainer
    try:
        cuda_avail = bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
        requested_device = torch.device('cuda') if cuda_avail else torch.device('cpu')
        try:
            net.to(requested_device)
        except Exception:
            requested_device = torch.device('cpu')
            net.to(requested_device)
        any_param = next(iter(net._ppn.parameters()))  # type: ignore[attr-defined]
        effective_device = any_param.device
        using_gpu = (effective_device.type == 'cuda' and cuda_avail)
        gpu_name = torch.cuda.get_device_name(0) if using_gpu else 'CPU'
        print(f"Using GPU: {'Yes' if using_gpu else 'No'} ({gpu_name}) | cuda.is_available={torch.cuda.is_available()} | device_count={torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    except Exception:
        print("Using GPU: No (CPU)")

    # Fit with unweighted CE
    net.model.fit(
        [hands_tr, discs_tr, called_tr, gss_tr],
        {'policy_flat': y_tr},
        epochs=epochs,
        batch_size=min(batch_size, hands_tr.shape[0] if hands_tr.shape[0] > 0 else 1),
        verbose=verbose,
        shuffle=True,
        early_stopping_patience=early_stopping_patience,
        val_x_list=[hands_ho, discs_ho, called_ho, gss_ho],
        val_targets={'policy_ flat': y_ho},
        legality_masks=lm_tr,
        val_legality_masks=lm_ho,
        learning_rate=1e-3
    )

    if not model_out.endswith('.pt'):
        model_out += '.pt'
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    net.save_model(model_out)
    return model_out


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train rule-copy (imitation) policy with cross-entropy')
    parser.add_argument('--data', required=True, help='Path to .npz dataset from generate_pure_policy_dataset')
    parser.add_argument('--out', required=True, help='Output model path (.pt)')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--embed', type=int, default=4)
    from src.core.constants import MAX_TURNS as CONST_MAX_TURNS
    parser.add_argument('--max_turns', type=int, default=int(CONST_MAX_TURNS))
    parser.add_argument('--patience', type=int, default=5)
    args = parser.parse_args()

    path = train_rule_copy(
        dataset_path=args.data,
        model_out=args.out,
        hidden_size=args.hidden,
        embedding_dim=args.embed,
        max_turns=args.max_turns,
        epochs=args.epochs,
        batch_size=args.batch,
        verbose=1,
        early_stopping_patience=args.patience,
    )
    print(f'Saved model to {path}')



