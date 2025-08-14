#!/usr/bin/env python3
from __future__ import annotations

import os
import argparse
import importlib
import threading
import time
from typing import List, Tuple, Type, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from core.game import SimpleJong, Player
from core.learn.pure_policy import PurePolicyNetwork
from core.learn.parallel_policy_player import ParallelPolicyPlayer, get_or_create_predictor
from core.parallel_jong import ParallelJong
class ProgressReporter:
    """Lightweight, thread-safe progress printer suitable for parallel execution.

    Prints a single-line progress message periodically (and on completion) without
    requiring external dependencies.
    """

    def __init__(self, total: int, desc: str = "Progress", interval_sec: float = 0.5):
        self.total = max(0, int(total))
        self.desc = desc
        self.interval_sec = max(0.05, float(interval_sec))
        self._count = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_time = 0.0

    def start(self) -> None:
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._run, name="ProgressReporter", daemon=True)
        self._thread.start()

    def update(self, n: int = 1) -> None:
        if n <= 0:
            return
        with self._lock:
            self._count += int(n)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_sec):
            self._print_line()

    def _print_line(self, final: bool = False) -> None:
        with self._lock:
            done = min(self._count, self.total)
        elapsed = max(1e-6, time.time() - self._start_time)
        rate = done / elapsed
        remaining = max(0, self.total - done)
        eta = (remaining / rate) if rate > 0 else 0.0
        pct = (100.0 * done / self.total) if self.total > 0 else 100.0
        msg = f"{self.desc}: {done}/{self.total} ({pct:5.1f}%) | {rate:6.1f}/s | ETA {eta:6.1f}s"
        endc = "\n" if final else "\r"
        try:
            sys.stderr.write(msg + endc)
            sys.stderr.flush()
        except Exception:
            pass

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        # Final line
        self._print_line(final=True)


def _import_class(class_path: str) -> Type:
    """Import a class by simple name or fully-qualified path.

    - 'Player' or 'PurePolicyPlayer' use known registry
    - 'package.module.ClassName' imports dynamically
    """
    registry = {
        'Player': Player,
        'PurePolicyPlayer': ParallelPolicyPlayer,
    }
    if class_path in registry:
        return registry[class_path]
    if '.' in class_path:
        module_name, cls_name = class_path.rsplit('.', 1)
        mod = importlib.import_module(module_name)
        return getattr(mod, cls_name)
    raise ValueError(f'Unknown player class: {class_path}')



def build_players(models_csv: Optional[str], shared_nets: Optional[Dict[str, PurePolicyNetwork]] = None, shared_predictors: Optional[Dict[str, Any]] = None) -> List[Player]:
    """Build four players inferred from model paths.

    Rule: non-empty/non-'-' model path => PurePolicyPlayer with that model; otherwise baseline Player.
    """
    model_paths = ['-','-','-','-']
    if models_csv:
        model_paths = [m.strip() for m in models_csv.split(',')]
        if len(model_paths) != 4:
            raise ValueError('If provided, --models must have exactly 4 comma-separated entries (use - for none)')

    # Use provided shared nets if available; else lazy local cache
    path_to_net: Dict[str, PurePolicyNetwork] = {} if shared_nets is None else dict(shared_nets)
    path_to_pred: Dict[str, Any] = {} if shared_predictors is None else dict(shared_predictors)
    players: List[Player] = []
    for i in range(4):
        mp = model_paths[i] if i < len(model_paths) else '-'
        if mp and mp != '-':
            if mp not in path_to_net:
                net = PurePolicyNetwork(embedding_dim=8)
                net.load_model(mp)
                path_to_net[mp] = net
            # Create or reuse a shared predictor for this net so GPU batching spans games
            if mp not in path_to_pred:
                path_to_pred[mp] = get_or_create_predictor(path_to_net[mp])
            players.append(ParallelPolicyPlayer(i, path_to_net[mp], predictor=path_to_pred[mp]))
        else:
            players.append(Player(i))
    return players


def _play_chunk(num_games: int, models_config: Optional[str], shared_nets: Optional[Dict[str, PurePolicyNetwork]], on_progress: Optional[callable] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Play num_games sequentially in this worker and return wins, losses arrays."""
    wins = np.zeros(4, dtype=np.int32)
    losses = np.zeros(4, dtype=np.int32)
    # Build N games and run them via ParallelJong with threads=1 (sequential) to reuse reporter
    games = []
    for _ in range(num_games):
        players = build_players(models_config, shared_nets=shared_nets)
        games.append(SimpleJong(players))
    pj = ParallelJong(games, threads=1, progress_desc="Chunk")
    # Wrap reporter update into ParallelJong's progress by disabling its own and calling on_progress
    if on_progress is None:
        pj.run(show_progress=False)
    else:
        for g in pj.run(show_progress=False):
            try:
                on_progress(1)
            except Exception:
                pass
    for game in games:
        if game.winners:
            for w in game.winners:
                wins[int(w)] += 1
            if game.loser is not None:
                losses[int(game.loser)] += 1
    return wins, losses


def _build_shared_nets(models_csv: Optional[str]) -> Dict[str, PurePolicyNetwork]:
    """Preload networks once to share across threads and avoid repeated disk IO."""
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


def _build_shared_predictors(shared_nets: Dict[str, PurePolicyNetwork]) -> Dict[str, Any]:
    preds: Dict[str, Any] = {}
    for mp, net in shared_nets.items():
        preds[mp] = get_or_create_predictor(net)
    return preds


def play_n_games(n: int, models_config: Optional[str], threads: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Play n games, optionally in parallel using threads, and return wins/losses per seat."""
    threads = max(1, int(threads))
    # Preload nets once to share within threads
    shared_nets = _build_shared_nets(models_config) if threads > 1 else None
    shared_predictors = _build_shared_predictors(shared_nets) if shared_nets else None

    if threads == 1:
        return _play_chunk(n, models_config, shared_nets)

    # Partition work
    base = n // threads
    remainder = n % threads
    chunk_sizes = [base + (1 if i < remainder else 0) for i in range(threads)]

    wins_total = np.zeros(4, dtype=np.int32)
    losses_total = np.zeros(4, dtype=np.int32)

    # Build all games upfront and run with ParallelJong using desired threads
    all_games: List[SimpleJong] = []
    for chunk in chunk_sizes:
        for _ in range(chunk):
            players = build_players(models_config, shared_nets=shared_nets, shared_predictors=shared_predictors)
            all_games.append(SimpleJong(players))

    # Use num_concurrent = n to allow more games in-flight than threads when beneficial for GPU batching
    pj = ParallelJong(all_games, threads=threads, num_concurrent=threads, progress_desc='Competing')
    pj.run(show_progress=True)

    for game in all_games:
        if game.winners:
            for w in game.winners:
                wins_total[int(w)] += 1
            if game.loser is not None:
                losses_total[int(game.loser)] += 1

    return wins_total, losses_total


def main():
    parser = argparse.ArgumentParser(description='Run head-to-head competitions among 4 players (parallel)')
    parser.add_argument('--models', type=str, default='-,-,-,-', help='Comma-separated model paths for each seat (use - or empty for none). Non-empty => PurePolicyPlayer; empty => baseline Player')
    parser.add_argument('--games', type=int, default=100, help='Number of games to play')
    parser.add_argument('--threads', type=int, default=1, help='Number of threads to use for parallel simulation')
    args = parser.parse_args()

    wins, losses = play_n_games(args.games, args.models, threads=args.threads)
    totals = wins + losses
    rewards = wins.astype(np.int32) - losses.astype(np.int32)
    neither = (args.games - totals).astype(np.int32)  # per-seat draws

    print('Results over', args.games, 'games:')
    for i in range(4):
        total = float(max(1, args.games))
        win_rate = float(wins[i]) / total
        loss_rate = float(losses[i]) / total
        neither_rate = float(neither[i]) / total
        print(f'Player {i}: wins={int(wins[i])}, losses={int(losses[i])}, draws/other={int(neither[i])}, win_rate={win_rate:.3f}, loss_rate={loss_rate:.3f}, neither_rate={neither_rate:.3f}, reward_sum={int(rewards[i])}')


if __name__ == '__main__':
    main()



