#!/usr/bin/env python3
from __future__ import annotations

import os
import argparse
import importlib
from typing import List, Tuple, Type, Optional

import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from core.game import SimpleJong, Player
from core.learn.pure_policy import PurePolicyNetwork
from core.learn.pure_policy_player import PurePolicyPlayer
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None  # type: ignore


def _import_class(class_path: str) -> Type:
    """Import a class by simple name or fully-qualified path.

    - 'Player' or 'PurePolicyPlayer' use known registry
    - 'package.module.ClassName' imports dynamically
    """
    registry = {
        'Player': Player,
        'PurePolicyPlayer': PurePolicyPlayer,
    }
    if class_path in registry:
        return registry[class_path]
    if '.' in class_path:
        module_name, cls_name = class_path.rsplit('.', 1)
        mod = importlib.import_module(module_name)
        return getattr(mod, cls_name)
    raise ValueError(f'Unknown player class: {class_path}')



def build_players(models_csv: Optional[str]) -> List[Player]:
    """Build four players inferred from model paths.

    Rule: non-empty/non-'-' model path => PurePolicyPlayer with that model; otherwise baseline Player.
    """
    model_paths = ['-','-','-','-']
    if models_csv:
        model_paths = [m.strip() for m in models_csv.split(',')]
        if len(model_paths) != 4:
            raise ValueError('If provided, --models must have exactly 4 comma-separated entries (use - for none)')

    path_to_net: dict[str, PurePolicyNetwork] = {}
    players: List[Player] = []
    for i in range(4):
        mp = model_paths[i] if i < len(model_paths) else '-'
        if mp and mp != '-':
            if mp not in path_to_net:
                net = PurePolicyNetwork(embedding_dim=8)
                net.load_model(mp)
                path_to_net[mp] = net
            players.append(PurePolicyPlayer(i, path_to_net[mp]))
        else:
            players.append(Player(i))
    return players


def play_n_games(n: int, models_config: Optional[str]) -> Tuple[np.ndarray, np.ndarray]:
    wins = np.zeros(4, dtype=np.int32)
    losses = np.zeros(4, dtype=np.int32)
    iterator = tqdm(range(n), desc='Competing') if tqdm else range(n)
    for _ in iterator:
        players = build_players(models_config)
        game = SimpleJong(players)
        game.play_round()
        # Outcomes: winners (one or multiple), loser (discarder) or None
        if game.winners:
            for w in game.winners:
                wins[w] += 1
            if game.loser is not None:
                losses[game.loser] += 1
        # Draws add nothing
    return wins, losses


def main():
    parser = argparse.ArgumentParser(description='Run head-to-head competitions among 4 players')
    parser.add_argument('--models', type=str, default='-,-,-,-', help='Comma-separated model paths for each seat (use - or empty for none). Non-empty => PurePolicyPlayer; empty => baseline Player')
    parser.add_argument('--games', type=int, default=100, help='Number of games to play')
    args = parser.parse_args()

    wins, losses = play_n_games(args.games, args.models)
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


