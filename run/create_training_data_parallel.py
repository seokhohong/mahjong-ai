#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import time
import argparse
import multiprocessing as mp
from typing import List, Tuple, Callable, Any, Optional, Dict

import numpy as np


# Ensure project root and src are importable when running as a script/module
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_SRC = os.path.join(_ROOT, 'src')
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


def _set_cpu_thread_env(max_threads: int) -> None:
    """Limit intra-op threading libraries to avoid oversubscription across processes."""
    max_threads_str = str(max(1, int(max_threads)))
    for var in (
        'OMP_NUM_THREADS',
        'MKL_NUM_THREADS',
        'OPENBLAS_NUM_THREADS',
        'NUMEXPR_NUM_THREADS',
    ):
        os.environ.setdefault(var, max_threads_str)


def _shard_counts(total: int, parts: int) -> List[int]:
    base = total // max(1, parts)
    rem = total % max(1, parts)
    return [base + (1 if i < rem else 0) for i in range(max(1, parts))]


def _worker_run(games: int, seed: int, shard_idx: int, session_dir: str) -> Tuple[int, str]:
    """Run a shard of game generation in an isolated process.

    Returns (shard_idx, output_path)
    """
    # Late import inside subprocess to avoid pickling issues with closures
    from run.create_training_data import create_dataset_from_pool  # type: ignore

    # Ensure session directory exists and compute shard filename
    os.makedirs(session_dir, exist_ok=True)
    out_path = os.path.join(session_dir, f'{int(shard_idx)}.npz')

    # Use a per-shard seed to ensure different streams
    pid = os.getpid()
    shard_seed = int(seed) + int(shard_idx) * 9973 + int(pid) % 1000

    # Build pool locally (cannot pass closures across processes on Windows)
    pool = default_pool()
    result_path = create_dataset_from_pool(num_games=int(games), player_pool=pool, seed=shard_seed, out_path=out_path)
    return shard_idx, result_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Create training data in parallel using multiple processes')
    parser.add_argument('--games', type=int, default=200, help='Total number of games to generate (split across processes)')
    parser.add_argument('--procs', type=int, default=max(1, mp.cpu_count() // 2), help='Number of worker processes')
    parser.add_argument('--seed', type=int, default=123, help='Base random seed')
    parser.add_argument('--threads-per-proc', type=int, default=1, help='Max BLAS/OMP threads per process')
    # Deprecated: previously used as filename prefix; now we always create a session dir per run
    parser.add_argument('--out-prefix', type=str, default='pure_policy_pool_parallel', help='(Ignored) legacy prefix; shards are saved as N.npz in a timestamped folder')
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Safer start method for Windows; harmless elsewhere
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # Limit intra-op threads for each subprocess
    _set_cpu_thread_env(args.threads_per_proc)

    # Compute a single timestamped session directory for this run
    ts = time.strftime('%Y%m%d_%H%M%S')
    session_dir = os.path.join(_ROOT, 'training_data', ts)

    # Compute shards
    counts = _shard_counts(int(args.games), int(args.procs))
    # Drop empty shards
    shards = [(i, c) for i, c in enumerate(counts) if c > 0]
    if not shards:
        print('Nothing to do (0 games).')
        return 0

    print(f"Output directory: {session_dir}")
    print(f"Spawning {len(shards)} processes for {sum(c for _, c in shards)} total games...")

    with mp.Pool(processes=len(shards)) as pool:
        jobs = []
        for shard_idx, num_games in shards:
            jobs.append(
                pool.apply_async(
                    _worker_run,
                    kwds=dict(
                        games=int(num_games),
                        seed=int(args.seed),
                        shard_idx=int(shard_idx),
                        session_dir=str(session_dir),
                    ),
                )
            )

        outputs: List[Tuple[int, str]] = []
        for j in jobs:
            outputs.append(j.get())

    outputs.sort(key=lambda t: t[0])
    print('Completed shards:')
    for idx, path in outputs:
        print(f"  shard {idx}: {path}")

    print(f'All shards saved under: {session_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


# ===== Local pool configuration (duplicated for independent tuning) =====

from core.game import SimpleJong, Player, HeuristicPlayer, Discard, Ron, Chi, Pon  # type: ignore
from core.learn.pure_policy import PurePolicyNetwork  # type: ignore
from core.learn.pure_policy_dataset import (
    serialize_state,
    extract_indexed_state,
    encode_action_flat_index,
)  # type: ignore


PlayerFactory = Callable[[int], Player]


class SamplingPolicyPlayer(Player):
    def __init__(self, player_id: int, network: Any):
        super().__init__(player_id)
        self.network = network

    def _encode_inputs(self, gs):
        sd = serialize_state(gs)
        idx = extract_indexed_state(sd)
        hand_idx = idx['hand_idx']
        disc_idx = idx['disc_idx']
        try:
            from core.constants import MAX_CALLED_SETS_PER_PLAYER as _MCSP
        except Exception:
            _MCSP = 3
        called_idx = idx.get('called_sets_idx', np.zeros((4, _MCSP, 3), dtype=np.int32))
        game_state = idx['game_state']
        return hand_idx, disc_idx, called_idx, game_state

    def _legal_pairs(self, gs, legal_moves: List[Any]) -> List[Tuple[int, Any]]:
        pairs: List[Tuple[int, Any]] = []
        sd = serialize_state(gs)
        for m in legal_moves:
            if isinstance(m, Discard):
                ad = {'type': 'discard', 'tile': str(m.tile)}
            elif isinstance(m, Ron):
                ad = {'type': 'ron'}
            elif isinstance(m, Pon):
                ad = {'type': 'pon', 'tiles': [str(t) for t in m.tiles]}
            elif isinstance(m, Chi):
                ad = {'type': 'chi', 'tiles': [str(t) for t in m.tiles]}
            else:
                continue
            ldt = sd.get('last_discarded_tile')
            ai = encode_action_flat_index(ad, ldt)
            pairs.append((ai, m))
        return pairs

    def _sample_move(self, gs, legal_moves: List[Any]) -> Optional[Any]:
        if not legal_moves:
            return None
        hand_idx, disc_idx, called_idx, game_state = self._encode_inputs(gs)
        probs = self.network.model.predict([
            hand_idx[None, :],
            disc_idx[None, :, :],
            called_idx[None, :, :, :],
            game_state[None, :],
        ], verbose=0)[0]
        mask = gs.legality_mask()
        action_pairs = self._legal_pairs(gs, legal_moves)
        prob_vec = np.array(probs, dtype=np.float64)
        if mask is not None and mask.shape[0] == prob_vec.shape[0]:
            prob_vec[~mask] = 0.0
        legal_indices = [int(ai) for ai, _ in action_pairs]
        legal_probs = prob_vec[legal_indices]
        total = float(np.sum(legal_probs))
        if np.isclose(total, 0):
            import numpy as _np
            return _np.random.choice(legal_moves)
        legal_probs = legal_probs / total
        choice = int(np.random.choice(len(action_pairs), p=legal_probs))
        return action_pairs[choice][1]

    def play(self, game_state):  # type: ignore[override]
        legal = game_state.legal_moves()
        mv = self._sample_move(game_state, legal)
        if mv is not None:
            if not game_state.is_legal(mv):
                raise SimpleJong.IllegalMoveException("SamplingPolicyPlayer produced illegal action")
            return mv
        fallback = super().play(game_state)
        if not game_state.is_legal(fallback):
            raise SimpleJong.IllegalMoveException("SamplingPolicyPlayer fallback produced illegal action")
        return fallback

    def choose_reaction(self, game_state, options):  # type: ignore[override]
        legal: List[Any] = []
        if game_state.can_ron():
            legal.append(Ron())
        for tiles in options.get('pon', []):
            legal.append(Pon(tiles))
        for tiles in options.get('chi', []):
            legal.append(Chi(tiles))
        mv = self._sample_move(game_state, legal)
        if mv is not None:
            if not game_state.is_legal(mv):
                raise SimpleJong.IllegalMoveException("SamplingPolicyPlayer produced illegal reaction")
            return mv
        fallback = super().choose_reaction(game_state, options)
        if not game_state.is_legal(fallback):
            raise SimpleJong.IllegalMoveException("SamplingPolicyPlayer fallback produced illegal reaction")
        return fallback


_net_cache: Dict[str, PurePolicyNetwork] = {}


def make_policy_player_from_path(model_path: str, temperature: float = 1.0) -> PlayerFactory:
    key = f"{os.path.abspath(model_path)}::temp={float(temperature):.6f}"

    def factory(pid: int) -> Player:
        net = _net_cache.get(key)
        if net is None:
            net = PurePolicyNetwork(embedding_dim=8, temperature=temperature)
            net.load_model(model_path)
            _net_cache[key] = net
        return SamplingPolicyPlayer(pid, net)

    return factory


def make_heuristic_player(temp: float) -> PlayerFactory:
    def factory(pid: int) -> Player:
        return HeuristicPlayer(pid, temperature=float(temp))

    return factory


def default_pool() -> List[PlayerFactory]:
    pool: List[PlayerFactory] = [
        make_policy_player_from_path(os.path.join(_ROOT, 'models', 'pure_policy_brief.pt'), temperature=0),
        make_policy_player_from_path(os.path.join(_ROOT, 'models', 'pure_policy_brief.pt'), temperature=0.1),
        make_policy_player_from_path(os.path.join(_ROOT, 'models', 'policy_grad_gen1.pt'), temperature=0.0),
        make_policy_player_from_path(os.path.join(_ROOT, 'models', 'policy_grad_gen1.pt'), temperature=0.1),
        make_policy_player_from_path(os.path.join(_ROOT, 'models', 'copy_gen1.pt'), temperature=0.0),
        make_heuristic_player(temp=0.05)
    ]
    return pool


