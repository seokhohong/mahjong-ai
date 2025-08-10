from __future__ import annotations

import os
import json
import time
import random
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from scipy import sparse as sp  # efficient batch one-hot
from tqdm import tqdm  # type: ignore

from core.game import (
    SimpleJong,
    Player,
    GamePerspective,
    Tile,
    TileType,
    Suit,
    Tsumo,
    Ron,
    Discard,
    Pon,
    Chi,
    PassCall,
)
from core.constants import (
    TOTAL_TILES,
    MAX_CALLED_SETS_PER_PLAYER,
    MAX_CALLED_SETS_ALL_OPPONENTS,
    EMBEDDING_DIM,
    MAX_TURNS as CONST_MAX_TURNS,
    GAME_STATE_VEC_LEN,
    MAX_HAND_TILES,
)
from core.encoding import tile_str_to_index


def _tile_str(tile: Optional[Tile]) -> Optional[str]:
    if tile is None:
        return None
    return f"{tile.tile_type.value}{tile.suit.value}"


def serialize_state(state: GamePerspective) -> Dict[str, Any]:
    """Serialize a `GamePerspective` into a JSON-friendly dict."""
    called_sets_ser: Dict[int, List[Dict[str, Any]]] = {}
    for pid, sets in state.called_sets.items():
        called_sets_ser[pid] = []
        for s in sets:
            try:
                called_sets_ser[pid].append({
                    'tiles': [_tile_str(t) for t in s.tiles],
                    'call_type': s.call_type,
                    'called_tile': _tile_str(s.called_tile),
                    'caller_position': s.caller_position,
                    'source_position': s.source_position,
                })
            except Exception:
                # Fallback minimal set
                called_sets_ser[pid].append({'tiles': [], 'call_type': 'unknown'})

    player_discards = {}
    try:
        # state.player_discards provided by engine snapshot; already strings
        for i in range(4):
            player_discards[i] = list(state.player_discards.get(i, []))  # type: ignore[attr-defined]
    except Exception:
        player_discards = {i: [] for i in range(4)}

    return {
        'player_id': state.player_id,
        'player_hand': [_tile_str(t) for t in state.player_hand],
        'remaining_tiles': state.remaining_tiles,
        'other_players_discarded': {k: [_tile_str(t) for t in v] for k, v in state.other_players_discarded.items()},
        'called_sets': called_sets_ser,
        'last_discarded_tile': _tile_str(state.last_discarded_tile),
        'last_discard_player': state.last_discard_player,
        'can_call': state.can_call,
        'player_discards': player_discards,
    }


def serialize_action(action: Any) -> Dict[str, Any]:
    if isinstance(action, Tsumo):
        return {'type': 'tsumo'}
    if isinstance(action, Ron):
        return {'type': 'ron'}
    if isinstance(action, Discard):
        return {'type': 'discard', 'tile': _tile_str(action.tile)}
    if isinstance(action, Pon):
        return {'type': 'pon', 'tiles': [_tile_str(t) for t in action.tiles]}
    if isinstance(action, Chi):
        return {'type': 'chi', 'tiles': [_tile_str(t) for t in action.tiles]}
    if isinstance(action, PassCall):
        return {'type': 'pass'}
    # Unknown fallback
    return {'type': 'unknown'}


def _assign_rewards(num_players: int, winners: List[int], loser: Optional[int]) -> List[float]:
    """Reward rule:
    - +1 for any winner (tsumo, single ron, or multi-ron)
    - -1 for loss (discarder) whenever defined
    - 0 for any other outcome (draw or non-involved players)
    """
    rewards = [0.0] * num_players

    # Assign winner rewards
    for winner in winners:
        rewards[winner] = 1.0

    # Assign loser penalty
    if loser is not None:
        rewards[loser] = -1.0

    return rewards


# --- Action space (flattened) and feature extraction compatible with PurePolicyNetwork ---
EMBED_DIM = EMBEDDING_DIM
MAX_TURNS = CONST_MAX_TURNS

# Flattened action space definition
# Order:
# 0: pass (no reaction)
# 1: ron
# 2: tsumo
# 3..20: discard_{tile_idx} for tile_idx in [0..17]
# 21..38: pon_{tile_idx} for tile_idx in [0..17]
# 39..92: chi_{tile_idx}_{variant} for tile_idx in [0..17], variant in ['left','mid','right']
# Note: Some chi variants are invalid at tile edges; these slots will simply never be legal/used.
ACTION_LABELS: List[str] = []
ACTION_LABELS.append('pass')
ACTION_LABELS.append('ron')
ACTION_LABELS.append('tsumo')
for i in range(18):
    ACTION_LABELS.append(f'discard_{i}')
for i in range(18):
    ACTION_LABELS.append(f'pon_{i}')
for i in range(18):
    for v in ['left', 'mid', 'right']:
        ACTION_LABELS.append(f'chi_{i}_{v}')
ACTION_INDEX: Dict[str, int] = {label: idx for idx, label in enumerate(ACTION_LABELS)}


def get_action_labels() -> List[str]:
    return list(ACTION_LABELS)


def get_action_index_map() -> Dict[str, int]:
    return dict(ACTION_INDEX)


def get_num_actions() -> int:
    return len(ACTION_LABELS)


def _tile_index_from_str(tile_str: str) -> int:
    return tile_str_to_index(tile_str)


def _get_tile_index(tile: Tile) -> int:
    from ..encoding import tile_to_index
    return tile_to_index(tile)


def _tile_embedding_from_index(idx: int) -> np.ndarray:
    np.random.seed(idx)
    emb = np.random.randn(EMBED_DIM) * 0.1
    np.random.seed()
    return emb


def _encode_hand_indices(player_hand: List[str]) -> np.ndarray:
    """Return (12,) int indices +1 (0=PAD)."""
    arr = np.zeros((12,), dtype=np.int32)
    for i, tile_str in enumerate(player_hand[:12]):
        if tile_str is None:
            continue
        idx = _tile_index_from_str(tile_str)
        arr[i] = idx + 1  # shift by +1 for pad=0
    return arr


def _encode_discards_indices(player_discards: List[str]) -> np.ndarray:
    disc = np.zeros((MAX_TURNS,), dtype=np.int32)
    for i, tile_str in enumerate(player_discards[:MAX_TURNS]):
        try:
            idx = _tile_index_from_str(tile_str)
            disc[i] = idx + 1  # pad=0
        except Exception:
            continue
    return disc


def _encode_game_state_50(sd: Dict[str, Any]) -> np.ndarray:
    features: List[float] = []
    features.append(float(sd.get('remaining_tiles', 0)) / float(TOTAL_TILES))
    features.append(1.0 if sd.get('can_call') else 0.0)
    ph = sd.get('player_hand', [])
    features.append(len(ph) / float(MAX_HAND_TILES))
    your_called = sd.get('called_sets', {}).get(sd.get('player_id', 0), [])
    features.append(len(your_called) / float(MAX_CALLED_SETS_PER_PLAYER))
    total_opponent_sets = 0
    for pid, sets in sd.get('called_sets', {}).items():
        if int(pid) != int(sd.get('player_id', 0)):
            total_opponent_sets += len(sets)
    features.append(total_opponent_sets / float(MAX_CALLED_SETS_ALL_OPPONENTS))
    vis = []
    try:
        pdis = sd.get('player_discards', {})
        if isinstance(pdis, dict):
            for i in range(4):
                vis.extend(pdis.get(i, []))
    except Exception:
        pass
    features.append(len(vis) / float(TOTAL_TILES))
    # last discard embedding
    ldst = sd.get('last_discarded_tile')
    if ldst:
        idx = _tile_index_from_str(ldst)
        features.extend(_tile_embedding_from_index(idx).tolist())
    else:
        features.extend([0.0] * EMBED_DIM)
    # last discard player one-hot (append at end for stable position)
    ldp = sd.get('last_discard_player')
    oh = [0.0] * 4
    if ldp is not None and 0 <= int(ldp) < 4:
        # Encode relative to viewer for player-invariance
        viewer = int(sd.get('player_id', 0))
        rel = (int(ldp) - viewer) % 4
        oh[rel] = 1.0
    # pad first so that oh occupies the last 4 positions
    while len(features) < max(0, GAME_STATE_VEC_LEN - 4):
        features.append(0.0)
    features.extend(oh)
    return np.asarray(features[:GAME_STATE_VEC_LEN], dtype=np.float32)


def extract_indexed_state(sd: Dict[str, Any]) -> Dict[str, np.ndarray]:
    hand_idx = _encode_hand_indices(sd.get('player_hand', []))
    pdis = sd.get('player_discards', {})
    disc_idx = np.zeros((4, MAX_TURNS), dtype=np.int32)
    # Rotate so row 0 is current player's discards, then clockwise order
    pid = int(sd.get('player_id', 0))
    for i in range(4):
        source_pid = (pid + i) % 4
        disc_idx[i] = _encode_discards_indices(pdis.get(source_pid, []))
    # Encode all players' called sets rotated by perspective as (4, MAX_CALLED_SETS_PER_PLAYER, 3)
    called_sets_idx = np.zeros((4, MAX_CALLED_SETS_PER_PLAYER, 3), dtype=np.int32)
    all_csets = sd.get('called_sets', {})
    for row in range(4):
        source_pid = (pid + row) % 4
        csets = all_csets.get(source_pid, []) if isinstance(all_csets, dict) else []
        for i, cs in enumerate(csets[:MAX_CALLED_SETS_PER_PLAYER]):
            tiles = cs.get('tiles', []) if isinstance(cs, dict) else []
            for j, t in enumerate(tiles[:3]):
                if t is None:
                    continue
                try:
                    called_sets_idx[row, i, j] = _tile_index_from_str(t) + 1  # +1 for PAD
                except Exception:
                    continue
    game_state = _encode_game_state_50(sd)
    return {
        'hand_idx': hand_idx,
        'disc_idx': disc_idx,
        'game_state': game_state,
        'called_sets_idx': called_sets_idx,
    }


def _chi_variant(last_tile: str, tiles: List[str]) -> Optional[str]:
    try:
        # Compare within same suit; we operate on tile numbers
        tnum = int(last_tile[:-1]); tsuit = last_tile[-1]
        nums = sorted(int(t[:-1]) for t in tiles if t and t[-1] == tsuit)
        if len(nums) != 2:
            return None
        if nums == [tnum + 1, tnum + 2]:
            return 'left'
        if nums == [tnum - 1, tnum + 1]:
            return 'mid'
        if nums == [tnum - 2, tnum - 1]:
            return 'right'
    except Exception:
        return None
    return None


def encode_action_flat_index(ad: Dict[str, Any], last_discarded_tile: Optional[str] = None) -> int:
    """Map a serialized action dict into a flattened action index.

    last_discarded_tile: optional string like '3p' to disambiguate Pon/Chi.
    """
    atype = ad.get('type')
    if atype == 'pass':
        return ACTION_INDEX['pass']
    if atype == 'ron':
        return ACTION_INDEX['ron']
    if atype == 'tsumo':
        return ACTION_INDEX['tsumo']
    if atype == 'discard':
        tile = ad.get('tile')
        if tile is not None:
            idx = _tile_index_from_str(tile)
            return ACTION_INDEX[f'discard_{idx}']
        return ACTION_INDEX['pass']
    if atype == 'pon':
        if last_discarded_tile:
            idx = _tile_index_from_str(last_discarded_tile)
            return ACTION_INDEX[f'pon_{idx}']
        return ACTION_INDEX['pass']
    if atype == 'chi':
        tiles = ad.get('tiles', [])
        if last_discarded_tile and tiles:
            base = _tile_index_from_str(last_discarded_tile)
            var = _chi_variant(last_discarded_tile, tiles)
            if var in ('left', 'mid', 'right'):
                return ACTION_INDEX[f'chi_{base}_{var}']
        return ACTION_INDEX['pass']
    # Unknowns map to pass
    return ACTION_INDEX['pass']


def _encode_policy_indices_from_action(sd: Dict[str, Any], ad: Dict[str, Any]) -> Tuple[int, Optional[int], Optional[int]]:
    """Return compact indices for action and tiles to allow fast sparse one-hot batching.

    - action_idx in [0..4] for ['discard','ron','tsumo','pon','chi'] (default 0 if unknown)
    - tile1_idx in [0..17] or None when not applicable
    - tile2_idx in [0..17] or None when not applicable
    """
    action_order = ['discard', 'ron', 'tsumo', 'pon', 'chi']
    atype = ad.get('type')
    action_idx = action_order.index(atype) if atype in action_order else 0

    tile1_idx: Optional[int] = None
    tile2_idx: Optional[int] = None

    try:
        if atype == 'discard':
            tile = ad.get('tile')
            if tile:
                idx = _tile_index_from_str(tile)
                if 0 <= idx < 18:
                    tile1_idx = idx
        elif atype == 'pon':
            last = sd.get('last_discarded_tile')
            if last:
                idx = _tile_index_from_str(last)
                if 0 <= idx < 18:
                    tile1_idx = idx
        elif atype == 'chi':
            tiles = ad.get('tiles', [])
            if len(tiles) >= 2:
                i1 = _tile_index_from_str(tiles[0])
                i2 = _tile_index_from_str(tiles[1])
                if 0 <= i1 < 18:
                    tile1_idx = i1
                if 0 <= i2 < 18:
                    tile2_idx = i2
    except Exception:
        # fall back to None indices on any parsing issue
        pass

    return action_idx, tile1_idx, tile2_idx


class Recorder:
    """Collects raw (actor_id, GamePerspective, action_obj) events in order.

    Storing unprocessed `GamePerspective` ensures feature parity checks can be
    performed by re-encoding from the same information used during inference.
    """
    def __init__(self) -> None:
        self.events: List[Tuple[int, GamePerspective, Any]] = []
        # Optional aligned lists of policy probability vectors and legality masks per event
        self.event_probs: List[Optional[np.ndarray]] = []
        self.event_legal_masks: List[Optional[np.ndarray]] = []

    def record(
        self,
        gs: GamePerspective,
        actor_id: int,
        action_obj: Any,
        probs: Optional[np.ndarray],
        legal_mask: np.ndarray,
    ) -> None:
        self.events.append((actor_id, gs, action_obj))
        self.event_probs.append(None if probs is None else np.asarray(probs, dtype=np.float32))
        self.event_legal_masks.append(np.asarray(legal_mask, dtype=np.float32))


class RecordingPlayer(Player):
    """Player that records the state and the chosen action/reaction."""
    def __init__(self, player_id: int, recorder: Recorder):
        super().__init__(player_id)
        self._rec = recorder

    def play(self, game_state: GamePerspective):  # type: ignore[override]
        # Capture legality mask at the moment of decision
        legal_mask = self._game.legality_mask(self.player_id)
        action = super().play(game_state)
        self._rec.record(game_state, self.player_id, action, None, legal_mask)
        return action

    def choose_reaction(self, game_state: GamePerspective, options: Dict[str, List[List[Tile]]]):  # type: ignore[override]
        legal_mask = self._game.legality_mask(self.player_id)
        reaction = super().choose_reaction(game_state, options)
        self._rec.record(game_state, self.player_id, reaction, None, legal_mask)
        return reaction


def _simulate_game_collecting() -> Tuple[List[Tuple[int, Dict[str, Any], Dict[str, Any]]], List[int], Optional[int], List[np.ndarray]]:
    """Simulate one full game capturing each player's actions and reactions with states.

    Returns (log, winners, loser) where log is a list of
    (actor_id, serialized_state, serialized_action) in chronological order.
    """
    recorder = Recorder()

    # Always create a fresh game with recording players to ensure full coverage
    players = [RecordingPlayer(i, recorder) for i in range(4)]
    game_local = SimpleJong(players)

    # Run the game to completion
    game_local.play_round()

    # Convert raw recorded events to serialized log and capture legality masks
    log: List[Tuple[int, Dict[str, Any], Dict[str, Any]]] = []
    masks: List[np.ndarray] = []
    for idx, (actor_id, gp, action_obj) in enumerate(recorder.events):
        sd = serialize_state(gp)
        ad = serialize_action(action_obj)
        log.append((actor_id, sd, ad))
        mask = recorder.event_legal_masks[idx]
        masks.append(np.asarray(mask, dtype=bool))

    winners = game_local.get_winners()
    loser = game_local.get_loser()
    return log, list(winners), loser, masks


def generate_pure_policy_dataset(
    num_games: int,
    seed: Optional[int] = None,
    out_path: Optional[str] = None,
) -> str:
    """Simulate num_games and save (state, action, reward) tuples into training_data/*.npz.

    Returns the path to the saved .npz file.
    """
    if seed is not None:
        random.seed(seed)

    # State objects (per-sample) with indexed inputs
    states: List[Dict[str, np.ndarray]] = []
    # Flattened action indices for single softmax head
    y_flat_idx: List[int] = []                    # scalar in [0..NUM_ACTIONS-1]
    rewards: List[float] = []
    records_game_id: List[int] = []
    records_step_id: List[int] = []
    legality_masks: List[np.ndarray] = []

    # Progress iterator
    iterator = tqdm(range(num_games), desc='Generating pure-policy games') if tqdm else range(num_games)
    game_counter = 0
    for game_idx in iterator:
        game_counter = int(game_idx)
        # Simulate a full game and collect chronological (state, action) pairs
        log, winners, loser, masks = _simulate_game_collecting()
        per_player_rewards = _assign_rewards(4, winners, loser)

        # Append
        step_counter = 0
        for step_i, (actor_id, state_dict, action_dict) in enumerate(log):
            # Safety: if a 'ron' action is recorded, there must be a defined loser (the discarder)
            if action_dict.get('type') == 'ron':
                assert loser is not None, "Recorded 'ron' action but loser is None"
            idx_state = extract_indexed_state(state_dict)
            act_idx = encode_action_flat_index(action_dict, state_dict.get('last_discarded_tile'))
            states.append(idx_state)
            y_flat_idx.append(int(act_idx))
            rewards.append(float(per_player_rewards[actor_id]))
            records_game_id.append(game_counter)
            records_step_id.append(step_counter)
            # Append legality mask aligned with this step
            # step_i maps into masks
            # Ensure shape matches num_actions
            mask_arr = np.asarray(masks[step_i], dtype=bool)
            assert np.sum(mask_arr) > 0, f"Found step with no legal actions: {step_i}"
            legality_masks.append(mask_arr)
            step_counter += 1

        game_counter += 1

    # Prepare output directory and filename
    base_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..')
    base_dir = os.path.abspath(base_dir)
    out_dir = os.path.join(base_dir, 'training_data')
    os.makedirs(out_dir, exist_ok=True)
    if out_path is None:
        ts = time.strftime('%Y%m%d_%H%M%S')
        out_path = os.path.join(out_dir, f'pure_policy_{ts}.npz')

    # Convert to numpy arrays
    np_states = np.asarray(states, dtype=object)
    # Build fast one-hot for flattened policy
    n = len(y_flat_idx)
    rows = np.arange(n, dtype=np.int32)
    cols = np.asarray(y_flat_idx, dtype=np.int32)
    data = np.ones(n, dtype=np.float32)
    num_actions = get_num_actions()
    np_policy = sp.coo_matrix((data, (rows, cols)), shape=(n, num_actions)).toarray().astype(np.float32)
    np_rewards = np.asarray(rewards, dtype=np.float32)
    np_game_ids = np.asarray(records_game_id, dtype=np.int32)
    np_step_ids = np.asarray(records_step_id, dtype=np.int32)
    np_legal_masks = np.asarray(legality_masks, dtype=np.bool_)

    # Save
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


if __name__ == '__main__':
    # Minimal CLI for quick generation
    import argparse
    parser = argparse.ArgumentParser(description='Generate pure-policy (state, action, reward) dataset')
    parser.add_argument('--games', type=int, default=10, help='Number of games to simulate')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--out', type=str, default=None, help='Output .npz path under training_data/')
    args = parser.parse_args()
    path = generate_pure_policy_dataset(args.games, seed=args.seed, out_path=args.out)
    print(f'Saved dataset to {path}')


