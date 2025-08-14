#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import time
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Ensure project root and src are importable when running as a script/module
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_SRC = os.path.join(_ROOT, 'src')
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from core.game import SimpleJong, Player, HeuristicPlayer, Discard, Ron, Chi, Pon  # type: ignore
from core.learn.pure_policy import PurePolicyNetwork  # type: ignore
from core.learn.pure_policy_dataset import (
    Recorder,
    RecordingWrapper,
    serialize_state,
    serialize_action,
    extract_indexed_state,
    encode_action_flat_index,
    get_action_labels,
)  # type: ignore

try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None


PlayerFactory = Callable[[int], Player]


class SamplingPolicyPlayer(Player):
    """Policy player that samples among legal actions using network probabilities.

    Uses the same encoding and legality handling as PurePolicyPlayer but draws
    stochastically from the legal subset (renormalized), leveraging network temperature.
    """

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
            return np.random.choice(legal_moves)
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


def create_dataset_from_pool(
    num_games: int,
    player_pool: List[PlayerFactory],
    seed: Optional[int] = None,
    out_path: Optional[str] = None,
) -> str:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Accumulators identical to pure_policy_dataset output
    states: List[Dict[str, np.ndarray]] = []
    y_flat_idx: List[int] = []
    rewards: List[float] = []
    records_game_id: List[int] = []
    records_step_id: List[int] = []
    legality_masks: List[np.ndarray] = []

    iterator = tqdm(range(num_games), desc='Generating games from pool') if tqdm else range(num_games)

    for gi in iterator:
        factories = [random.choice(player_pool) for _ in range(4)]
        recorder = Recorder()
        base_players = [f(i) for i, f in enumerate(factories)]
        players = [RecordingWrapper(i, base_players[i], recorder) for i in range(4)]
        try:
            from core.constants import TILE_COPIES_DEFAULT as _TCP
        except Exception:
            _TCP = 4
        game = SimpleJong(players, tile_copies=int(_TCP))
        for i in range(4):
            try:
                if isinstance(players[i], RecordingWrapper) and hasattr(players[i], '_base'):
                    setattr(players[i]._base, '_game', game)
            except Exception:
                pass

        game.play_round()

        winners = game.get_winners()
        loser = game.get_loser()
        per_player_rewards = [0.0, 0.0, 0.0, 0.0]
        for w in winners:
            per_player_rewards[int(w)] = 1.0
        if loser is not None:
            per_player_rewards[int(loser)] = -1.0

        step_counter = 0
        for step_i, (actor_id, gp, action_obj) in enumerate(recorder.events):
            sd = serialize_state(gp)
            ad = serialize_action(action_obj)
            idx_state = extract_indexed_state(sd)
            act_idx = encode_action_flat_index(ad, sd.get('last_discarded_tile'))
            states.append(idx_state)
            y_flat_idx.append(int(act_idx))
            rewards.append(float(per_player_rewards[actor_id]))
            records_game_id.append(gi)
            records_step_id.append(step_counter)
            mask_arr = np.asarray(recorder.event_legal_masks[step_i], dtype=bool)
            legality_masks.append(mask_arr)
            step_counter += 1

    base_dir = _ROOT
    out_dir = os.path.join(base_dir, 'training_data')
    os.makedirs(out_dir, exist_ok=True)
    if out_path is None:
        ts = time.strftime('%Y%m%d_%H%M%S')
        out_path = os.path.join(out_dir, f'pure_policy_pool_{ts}.npz')

    n = len(y_flat_idx)
    cols = np.asarray(y_flat_idx, dtype=np.int32)
    num_actions = len(get_action_labels())
    np_policy = np.zeros((n, num_actions), dtype=np.float32)
    np_policy[np.arange(n, dtype=np.int32), cols] = 1.0
    np_states = np.asarray(states, dtype=object)
    np_rewards = np.asarray(rewards, dtype=np.float32)
    np_game_ids = np.asarray(records_game_id, dtype=np.int32)
    np_step_ids = np.asarray(records_step_id, dtype=np.int32)
    np_legal_masks = np.asarray(legality_masks, dtype=np.bool_)

    np.savez(
        out_path,
        states=np_states,
        y_flat=np_policy,
        rewards=np_rewards,
        game_ids=np_game_ids,
        step_ids=np_step_ids,
        action_labels=np.asarray(get_action_labels(), dtype=object),
        legal_masks=np_legal_masks,
    )

    return out_path


def default_pool() -> List[PlayerFactory]:
    pool: List[PlayerFactory] = []
    model = os.path.join(_ROOT, 'models', 'pure_policy_brief.pt')
    if os.path.exists(model):
        pool.extend([
            make_policy_player_from_path(model, temperature=0),
            make_policy_player_from_path(model, temperature=0.1)
        ])
    pool.extend([
        make_heuristic_player(0.0),
        make_heuristic_player(0.05),
    ])
    return pool


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Create training data from a configurable player pool')
    parser.add_argument('--games', type=int, default=50)
    args = parser.parse_args()
    out = create_dataset_from_pool(num_games=int(args.games), player_pool=default_pool(), seed=123)
    print('Saved dataset to', out)

 
