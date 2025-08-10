from __future__ import annotations

import argparse
import cProfile
import os
import pstats
from typing import Dict

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

from src.core.learn.policy_trainer import _batch_from_states
from src.core.learn.pure_policy import PurePolicyNetwork


def profile_training(
    dataset_path: str,
    batch_size: int = 1024,
    epochs: int = 1,
    hidden_size: int = 128,
    embedding_dim: int = 4,
    max_turns: int = 50,
    num_batches: int = 5,
    stats_out: str | None = None,
) -> None:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    data = np.load(dataset_path, allow_pickle=True)
    states = data['states']
    y_flat = data['y_flat']
    rewards = data['rewards'].astype(np.float32)

    # Build model inputs
    hands, discs, called, gss = _batch_from_states(states)
    # Limit to a small, fixed number of batches for short profiling runs
    total_samples = hands.shape[0]
    max_samples = min(total_samples, max(1, num_batches) * max(1, batch_size))
    hands = hands[:max_samples]
    discs = discs[:max_samples]
    called = called[:max_samples]
    gss = gss[:max_samples]
    y_flat = y_flat[:max_samples]
    rewards = rewards[:max_samples]

    # Construct model and select device
    net = PurePolicyNetwork(hidden_size=hidden_size, embedding_dim=embedding_dim, max_turns=max_turns)
    using_gpu = False
    if torch is not None and torch.cuda.is_available() and torch.cuda.device_count() > 0:
        net.to(torch.device('cuda'))
        using_gpu = True
    else:
        # Improve CPU throughput for a fair comparison
        try:
            ncores = max(1, (os.cpu_count() or 1))
            torch.set_num_threads(ncores)  # type: ignore[attr-defined]
            torch.set_num_interop_threads(min(4, ncores))  # type: ignore[attr-defined]
        except Exception:
            pass

    print(f"Using GPU: {'Yes' if using_gpu else 'No'}")

    targets: Dict[str, np.ndarray] = {'policy_flat': y_flat}
    sample_weight: Dict[str, np.ndarray] = {'policy_flat': rewards}

    # Ensure stats directory
    if stats_out is None:
        logs_dir = os.path.join('training_data', 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        stats_out = os.path.join(logs_dir, 'pure_policy_profile.stats')

    pr = cProfile.Profile()

    # Profile only the fit call; pre-processing is intentionally outside
    pr.enable()
    net.model.fit(
        [hands, discs, called, gss],
        targets,
        sample_weight=sample_weight,
        epochs=epochs,
        batch_size=min(batch_size, hands.shape[0] if hands.shape[0] > 0 else 1),
        verbose=1,
        shuffle=True,
    )
    if using_gpu and torch is not None:
        # Synchronize to capture kernel time in the profiling window
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
    pr.disable()

    pr.dump_stats(stats_out)
    print(f"Saved cProfile stats to {stats_out}")

    print("Top functions by cumulative time:")
    ps = pstats.Stats(pr).sort_stats(pstats.SortKey.CUMULATIVE)
    ps.print_stats(30)


def main() -> None:
    parser = argparse.ArgumentParser(description='Profile PurePolicyNetwork training loop time hotspots')
    parser.add_argument('--data', type=str, default=os.path.join('data', 'generation_0', 'pure_policy_gen0_5000.npz'))
    parser.add_argument('--batch', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batches', type=int, default=5, help='Number of batches to run during profiling')
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--embed', type=int, default=4)
    from src.core.constants import MAX_TURNS as CONST_MAX_TURNS
    parser.add_argument('--max_turns', type=int, default=int(CONST_MAX_TURNS))
    parser.add_argument('--stats', type=str, default=None, help='Optional path to write .stats file')
    args = parser.parse_args()

    profile_training(
        dataset_path=args.data,
        batch_size=args.batch,
        epochs=args.epochs,
        num_batches=args.batches,
        hidden_size=args.hidden,
        embedding_dim=args.embed,
        max_turns=args.max_turns,
        stats_out=args.stats,
    )


if __name__ == '__main__':
    main()


