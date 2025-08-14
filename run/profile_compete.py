#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
import cProfile
import pstats
from typing import Any, Dict, List, Optional

# Ensure project root and src are importable when running as a script
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_SRC = os.path.join(_ROOT, 'src')
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from core.game import SimpleJong, Player  # type: ignore
from core.parallel_jong import ParallelJong  # type: ignore
from core.learn.pure_policy_player import PurePolicyPlayer  # type: ignore
from core.learn.parallel_policy_player import ParallelPolicyPlayer, get_or_create_predictor  # type: ignore
from core.learn.pure_policy import PurePolicyNetwork  # type: ignore


def _build_shared_nets(models_csv: Optional[str]) -> Dict[str, PurePolicyNetwork]:
    cache: Dict[str, PurePolicyNetwork] = {}
    if not models_csv:
        return cache
    paths = [m.strip() for m in models_csv.split(',')]
    for mp in paths:
        if mp and mp != '-' and mp not in cache:
            net = PurePolicyNetwork(embedding_dim=8)
            net.load_model(mp)
            cache[mp] = net
    return cache


def build_players_for_impl(
    models_csv: Optional[str],
    impl: str,
    shared_nets: Optional[Dict[str, PurePolicyNetwork]] = None,
    shared_predictors: Optional[Dict[str, Any]] = None,
    batch_size: int = 32,
    wait_ms: int = 2,
    use_random_model_if_dash: bool = False,
) -> List[Player]:
    model_paths = ['-','-','-','-']
    if models_csv:
        model_paths = [m.strip() for m in models_csv.split(',')]
        if len(model_paths) != 4:
            raise ValueError('--models must have exactly 4 comma-separated entries (use - for none)')

    nets: Dict[str, PurePolicyNetwork] = {} if shared_nets is None else dict(shared_nets)
    predictors: Dict[str, Any] = {} if shared_predictors is None else dict(shared_predictors)

    players: List[Player] = []
    for i in range(4):
        mp = model_paths[i] if i < len(model_paths) else '-'
        if not mp or mp == '-':
            if not use_random_model_if_dash:
                players.append(Player(i))
                continue
            # Build a fresh random network when requested to force policy path
            net = PurePolicyNetwork(embedding_dim=8)
            nets[f"__random__:{i}"] = net
            mp_key = f"__random__:{i}"
        else:
            mp_key = mp
        if mp_key not in nets:
            net = PurePolicyNetwork(embedding_dim=8)
            if mp_key and not mp_key.startswith('__random__'):
                net.load_model(mp_key)
            nets[mp_key] = net
        net = nets[mp_key]
        if impl == 'pure':
            players.append(PurePolicyPlayer(i, net))
        elif impl == 'parallel':
            if mp_key not in predictors:
                predictors[mp_key] = get_or_create_predictor(net, max_batch_size=batch_size, max_wait_ms=wait_ms)
            players.append(ParallelPolicyPlayer(i, net, predictor=predictors[mp_key]))
        else:
            raise ValueError(f'Unknown impl: {impl}')
    return players


def run_games(
    n: int,
    models: Optional[str],
    impl: str,
    threads: int,
    batch_size: int,
    wait_ms: int,
    show_progress: bool,
    use_random_model_if_dash: bool,
) -> None:
    threads = max(1, int(threads))
    shared_nets = _build_shared_nets(models) if threads > 1 else None
    shared_predictors = None
    if impl == 'parallel' and shared_nets is not None:
        # Create shared predictors keyed by path to allow cross-game batching
        shared_predictors = {mp: get_or_create_predictor(net, max_batch_size=batch_size, max_wait_ms=wait_ms) for mp, net in shared_nets.items()}

    games: List[SimpleJong] = []
    for _ in range(n):
        players = build_players_for_impl(models, impl, shared_nets=shared_nets, shared_predictors=shared_predictors, batch_size=batch_size, wait_ms=wait_ms, use_random_model_if_dash=use_random_model_if_dash)
        games.append(SimpleJong(players))

    pj = ParallelJong(games, threads=threads, num_concurrent=threads, progress_desc=f'Profile-{impl}')
    pj.run(show_progress=show_progress)


def main():
    parser = argparse.ArgumentParser(description='Profile compete with Pure vs Parallel policy players')
    parser.add_argument('--models', type=str, default='-,-,-,-', help='Comma-separated model paths for each seat (use - or empty for none)')
    parser.add_argument('--games', type=int, default=50, help='Number of games to run')
    parser.add_argument('--threads', type=int, default=1, help='Number of threads to use')
    parser.add_argument('--impl', type=str, default='parallel', choices=['pure','parallel'], help='Which player implementation to use for non-dash model slots')
    parser.add_argument('--batch-size', type=int, default=32, help='Parallel predictor max batch size')
    parser.add_argument('--wait-ms', type=int, default=2, help='Parallel predictor max wait before flush (ms)')
    parser.add_argument('--use-random-model', action='store_true', help='If set, force policy players even when model entries are - by creating random networks')
    parser.add_argument('--no-progress', action='store_true', help='Disable progress output during profiling run')
    parser.add_argument('--save-stats', type=str, default='', help='Optional path to save cProfile stats file (.pstats)')
    parser.add_argument('--sort', type=str, default='cumulative', help='Sort key for pstats.print_stats (e.g., cumulative, time, tottime)')
    parser.add_argument('--top', type=int, default=120, help='How many lines of hotspots to print')
    args = parser.parse_args()

    print(f"Profiling impl={args.impl}, games={args.games}, threads={args.threads}, models='{args.models}', batch={args.batch_size}, wait_ms={args.wait_ms}, use_random_model={args.use_random_model}")

    profiler = cProfile.Profile()
    start = time.time()
    profiler.enable()
    try:
        run_games(
            n=int(args.games),
            models=args.models,
            impl=args.impl,
            threads=int(args.threads),
            batch_size=int(args.batch_size),
            wait_ms=int(args.wait_ms),
            show_progress=(not args.no_progress),
            use_random_model_if_dash=bool(args.use_random_model),
        )
    finally:
        profiler.disable()
    wall = time.time() - start

    ps = pstats.Stats(profiler)
    ps.sort_stats(args.sort)
    print(f"\n=== Top {args.top} hotspots ({args.sort}) ===")
    ps.print_stats(args.top)

    # Basic throughput indicator
    g_per_s = (args.games / wall) if wall > 0 else 0.0
    print(f"\nWall time: {wall:.3f}s  |  Throughput: {g_per_s:.2f} games/s")

    if args.save_stats:
        out_path = args.save_stats
        # Ensure directory exists
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
        ps.dump_stats(out_path)
        print(f"Saved cProfile stats to: {out_path}")


if __name__ == '__main__':
    main()


