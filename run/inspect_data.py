#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import argparse
from typing import Any, Dict, List

import numpy as np

# Ensure src on path for optional helpers
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def _fmt_called_sets(sd: Dict[str, Any], pid: int) -> str:
    called_ser = sd.get('called_sets', {}) or {}
    my_sets = called_ser.get(pid, []) if isinstance(called_ser, dict) else []
    if not my_sets:
        return "[]"
    parts: List[str] = []
    for cs in my_sets:
        try:
            ctype = cs.get('call_type', 'call')
            tiles = cs.get('tiles', []) or []
            parts.append(f"{ctype}(" + " ".join(str(t) for t in tiles if t) + ")")
        except Exception:
            continue
    return "[" + ", ".join(parts) + "]"


def _format_state(sd: Dict[str, Any]) -> str:
    pid = int(sd.get('player_id', 0))
    hand = sd.get('player_hand', []) or []
    ldt = sd.get('last_discarded_tile') or '-'
    ldp = sd.get('last_discard_player')
    ldp_str = '-' if ldp is None else str(ldp)
    called_repr = _fmt_called_sets(sd, pid)
    can_tsumo = bool(sd.get('can_tsumo', False))
    can_ron = bool(sd.get('can_ron', False))
    return (
        f"P{pid} Hand [" + " ".join(str(t) for t in hand if t) + "] "
        f"| Called {called_repr} | Last discard: {ldt} (by {ldp_str}) "
        f"| can_tsumo={can_tsumo} can_ron={can_ron}"
    )


def _format_action(ad: Dict[str, Any]) -> str:
    at = ad.get('type', 'unknown')
    if at in ('tsumo', 'ron', 'pass'):
        return at
    if at == 'discard':
        return f"discard({ad.get('tile')})"
    if at == 'pon':
        return "pon(" + " ".join(ad.get('tiles', []) or []) + ")"
    if at == 'chi':
        return "chi(" + " ".join(ad.get('tiles', []) or []) + ")"
    return str(ad)


def main():
    ap = argparse.ArgumentParser(description='Inspect AC dataset: print first N games with states, actions, and rewards')
    ap.add_argument('dataset', type=str, help='Path to .npz produced by create_dataset.py')
    ap.add_argument('--games', type=int, default=1, help='Number of games to print')
    ap.add_argument('--max_steps', type=int, default=None, help='Optional cap on steps per game')
    args = ap.parse_args()

    data = np.load(args.dataset, allow_pickle=True)
    states = data['states']  # object array of serialized states (dicts)
    actions = data['actions']  # object array of serialized actions (dicts)
    returns = data['returns']  # float array
    advantages = data['advantages']
    game_ids = data['game_ids']  # int array
    step_ids = data['step_ids']  # int array
    actor_ids = data['actor_ids']  # int array

    # Collect indices per game, keep order by step_ids
    unique_games = []
    seen = set()
    for gid in game_ids.tolist():
        if gid not in seen:
            seen.add(int(gid))
            unique_games.append(int(gid))

    to_show = unique_games[: max(0, int(args.games))]
    if not to_show:
        print('No games to display')
        return

    for gi, gid in enumerate(to_show, start=1):
        idxs = [i for i, g in enumerate(game_ids.tolist()) if int(g) == gid]
        idxs.sort(key=lambda i: int(step_ids[i]))
        if args.max_steps is not None:
            idxs = idxs[: max(0, int(args.max_steps))]

        print("\n" + "=" * 80)
        print(f"Game {gi} (id={gid}) | steps={len(idxs)}")
        print("-" * 80)
        for pos, i in enumerate(idxs):
            sd = states[i].item() if hasattr(states[i], 'item') else states[i]
            ad = actions[i].item() if hasattr(actions[i], 'item') else actions[i]
            rew = float(returns[i])
            adv = float(advantages[i]) if advantages is not None else None
            actor = int(actor_ids[i])
            step = int(step_ids[i])
            if adv is not None:
                print(f"Step {step:03d} | actor P{actor} | reward={rew:+.3f} | advantage={adv:+.3f}")
            else:
                print(f"Step {step:03d} | actor P{actor} | reward={rew:+.3f}")
            try:
                print("  ", _format_state(sd))
            except Exception:
                print("   <state pretty-print failed>")
            try:
                print("   ->", _format_action(ad))
            except Exception:
                print("   -> <action pretty-print failed>")


if __name__ == '__main__':
    main()


