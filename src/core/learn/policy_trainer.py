from __future__ import annotations

import os
from typing import Dict, Any, Tuple

import numpy as np

try:
    import torch  # type: ignore
    TORCH_AVAILABLE = True
except Exception as e:  # pragma: no cover
    TORCH_AVAILABLE = False
    raise ImportError('PyTorch is required for training. Please install torch.') from e

from .pure_policy import PurePolicyNetwork


def _batch_from_states(states_obj: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert object array of dict states into model inputs.

    Returns (hands_idx, disc_idx, game_state) with shapes:
    - hands_idx: (N, 12)
    - disc_idx: (N, 4, max_turns)
    - game_state: (N, 50)
    """
    hands = []
    discs = []
    called = []
    gss = []
    for s in states_obj:
        s = s.item() if hasattr(s, 'item') else s
        hands.append(s['hand_idx'])
        discs.append(s['disc_idx'])
        # called_sets_idx optional for backward compatibility
        called.append(s.get('called_sets_idx', np.zeros((4, 4, 3), dtype=np.int32)))
        gss.append(s['game_state'])
    hands = np.asarray(hands, dtype=np.int32)
    discs = np.asarray(discs, dtype=np.int32)
    called = np.asarray(called, dtype=np.int32)
    gss = np.asarray(gss, dtype=np.float32)
    return hands, discs, called, gss


def train_policy_gradient(
    dataset_path: str,
    model_out: str,
    hidden_size: int = 128,
    embedding_dim: int = 4,
    max_turns: int = 50,
    epochs: int = 3,
    batch_size: int = 128,
    learning_rate: float = 1e-4,
    verbose: int = 1,
    early_stopping_patience: int = 5,
) -> str:
    """Train a policy-only network with reward-weighted loss (policy gradient surrogate).

    - Loads dataset produced by generate_pure_policy_dataset
    - Builds inputs (hand_idx, disc_idx, game_state)
    - Uses sample weights equal to reward for the policy heads so positive-reward actions dominate
    - Saves the trained model to model_out (.keras)
    """
    if not dataset_path.endswith('.npz'):
        raise ValueError('dataset_path must be a .npz file produced by generate_pure_policy_dataset')
    data = np.load(dataset_path, allow_pickle=True)

    states = data['states']
    y_flat = data['y_flat']
    rewards = data['rewards'].astype(np.float32)

    # Build inputs
    hands, discs, called, gss = _batch_from_states(states)

    # Build model
    net = PurePolicyNetwork(
        hidden_size=hidden_size,
        embedding_dim=embedding_dim,
        max_turns=max_turns,
    )
    # Robust device selection & reporting
    try:
        # GPU truly available only if CUDA reports device_count > 0
        cuda_avail = bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
        requested_device = torch.device('cuda') if cuda_avail else torch.device('cpu')
        # Force placement
        try:
            net.to(requested_device)
        except Exception as _e:
            requested_device = torch.device('cpu')
            net.to(requested_device)
        # Effective device from parameters
        try:
            any_param = next(iter(net.parameters()))
            effective_device = any_param.device
        except StopIteration:
            effective_device = requested_device
        using_gpu = (effective_device.type == 'cuda' and cuda_avail)
        gpu_name = torch.cuda.get_device_name(0) if using_gpu else 'CPU'
        print(f"Using GPU: {'Yes' if using_gpu else 'No'} ({gpu_name}) | cuda.is_available={torch.cuda.is_available()} | device_count={torch.cuda.device_count() if torch.cuda.is_available() else 0}")
    except Exception:
        using_gpu = False
        print("Using GPU: No (CPU)")

    # If CPU-only, increase thread counts to speed up matmul on multi-core CPUs
    if not using_gpu:
        try:
            import os as _os
            num_cores = max(1, (_os.cpu_count() or 1))
            torch.set_num_threads(num_cores)
            # Keep interop small to avoid oversubscription
            torch.set_num_interop_threads(min(4, num_cores))
            print(f"PyTorch CPU threads set: intra={num_cores}, inter={min(4, num_cores)}")
        except Exception:
            pass
    # Note: our minimal wrapper does not expose optimizer; learning_rate kept for API compat

    # Prepare targets and sample weights
    targets: Dict[str, np.ndarray] = {
        'policy_flat': y_flat,
    }
    sample_weight: Dict[str, np.ndarray] = {
        'policy_flat': rewards,
    }
    # No value head in the pure-policy model

    # Fit
    net.model.fit(
        [hands, discs, called, gss],
        targets,
        sample_weight=sample_weight,
        epochs=epochs,
        batch_size=min(batch_size, hands.shape[0] if hands.shape[0] > 0 else 1),
        verbose=verbose,
        shuffle=True,
        early_stopping_patience=early_stopping_patience,
    )

    # Save
    if not model_out.endswith('.pt'):
        model_out += '.pt'
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    net.save_model(model_out)
    return model_out


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train pure-policy network with reward-weighted loss')
    parser.add_argument('--data', required=True, help='Path to .npz dataset from generate_pure_policy_dataset')
    parser.add_argument('--out', required=True, help='Output model path (.pt)')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience (epochs without improvement)')
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--embed', type=int, default=4)
    from ..constants import MAX_TURNS as CONST_MAX_TURNS
    parser.add_argument('--max_turns', type=int, default=int(CONST_MAX_TURNS))
    args = parser.parse_args()

    path = train_policy_gradient(
        dataset_path=args.data,
        model_out=args.out,
        hidden_size=args.hidden,
        embedding_dim=args.embed,
        max_turns=args.max_turns,
        epochs=args.epochs,
        batch_size=args.batch,
        learning_rate=args.lr,
        verbose=1,
        early_stopping_patience=args.patience,
    )
    print(f'Saved model to {path}')


