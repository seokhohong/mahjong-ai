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


def _warmup_predict(net: PurePolicyNetwork) -> None:
    import numpy as np
    hands = np.zeros((1, 12), dtype=np.int32)
    discs = np.zeros((1, 4, net.max_turns), dtype=np.int32)
    gss = np.zeros((1, 50), dtype=np.float32)
    _ = net.model.predict([hands, discs, gss], verbose=0)


def build_players(classes_csv: str, models_csv: Optional[str]) -> List[Player]:
    """Build four players from a comma-separated class list and optional per-seat models.

    Examples:
    --players "PurePolicyPlayer,Player,Player,Player" --models "/path/m1.keras,,,
    --players "core.learn.pure_policy_player.PurePolicyPlayer,Player,Player,Player" --models "m1.keras,-,-,-"
    """
    class_names = [c.strip() for c in classes_csv.split(',')]
    assert len(class_names) == 4, 'Exactly 4 player classes required'
    model_paths = [''] * 4
    if models_csv:
        model_paths = [m.strip() for m in models_csv.split(',')]
        if len(model_paths) != 4:
            raise ValueError('If provided, --models must have exactly 4 comma-separated entries (use - for none)')

    # Cache nets per unique model path
    path_to_net: dict[str, PurePolicyNetwork] = {}

    players: List[Player] = []
    for i, cname in enumerate(class_names):
        cls = _import_class(cname)
        if issubclass(cls, PurePolicyPlayer):
            mp = model_paths[i] if i < len(model_paths) else ''
            if not mp or mp == '-':
                raise ValueError(f'Model path required for PurePolicyPlayer at seat {i}')
            if mp not in path_to_net:
                net = PurePolicyNetwork()
                net.load_model(mp)
                _warmup_predict(net)
                path_to_net[mp] = net
            players.append(cls(i, path_to_net[mp]))
        elif issubclass(cls, Player):
            players.append(cls(i))
        else:
            raise ValueError(f'Unsupported player class: {cname}')
    return players


def play_n_games(n: int, players_config: str, models_config: Optional[str]) -> Tuple[np.ndarray, np.ndarray]:
    wins = np.zeros(4, dtype=np.int32)
    losses = np.zeros(4, dtype=np.int32)
    iterator = tqdm(range(n), desc='Competing') if tqdm else range(n)
    for _ in iterator:
        players = build_players(players_config, models_config)
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
    parser.add_argument('--players', type=str, default='PurePolicyPlayer,Player,Player,Player', help='Comma-separated class names for 4 seats (e.g., PurePolicyPlayer,Player,Player,Player)')
    parser.add_argument('--models', type=str, default='training_data/generation_1/pure_policy_model_flat.keras,-,-,-', help='Comma-separated model paths for each seat (use - for none)')
    parser.add_argument('--games', type=int, default=100, help='Number of games to play')
    args = parser.parse_args()

    wins, losses = play_n_games(args.games, args.players, args.models)
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


