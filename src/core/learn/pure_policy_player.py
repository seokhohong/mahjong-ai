from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..game import (
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
    Reaction,
    PassCall,
)
from ..game import SimpleJong  # for IllegalMoveException
# Optional import to avoid hard dependency on torch during lightweight tests
from .pure_policy import PurePolicyNetwork  # type: ignore
from .pure_policy_dataset import (
    get_action_index_map,
    serialize_state,
    extract_indexed_state,
    encode_action_flat_index,
)
from ..encoding import tile_to_index


def _tile_index(tile: Tile) -> int:
    return tile_to_index(tile)


class PurePolicyPlayer(Player):
    """A Player that uses a trained PurePolicyNetwork (flattened action space) to act.

    It maps legal actions to indices in the flattened action space and selects
    the highest-scoring legal action according to the network's policy.
    """

    def __init__(self, player_id: int, network: Any):
        super().__init__(player_id)
        self.network = network
        self._action_index = get_action_index_map()

    # --- Model input encoding (indices) ---
    def _encode_inputs(self, gs: GamePerspective) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Reuse dataset encoders via serialization path
        sd = serialize_state(gs)
        idx = extract_indexed_state(sd)
        hand_idx = idx['hand_idx']
        disc_idx = idx['disc_idx']
        try:
            from ..constants import MAX_CALLED_SETS_PER_PLAYER as _MCSP
        except Exception:
            _MCSP = 3
        called_idx = idx.get('called_sets_idx', np.zeros((4, _MCSP, 3), dtype=np.int32))
        game_state = idx['game_state']
        return hand_idx, disc_idx, called_idx, game_state

    # --- Mapping legal moves to flattened indices ---
    def _legal_action_indices_action_phase(self, gs: GamePerspective, legal_moves: List[Any]) -> List[Tuple[int, Any]]:
        pairs: List[Tuple[int, Any]] = []
        sd = serialize_state(gs)
        for m in legal_moves:
            ad = {'type': 'tsumo'} if isinstance(m, Tsumo) else (
                {'type': 'discard', 'tile': str(m.tile)} if isinstance(m, Discard) else None)
            if ad is None:
                continue
            ldt = sd.get('last_discarded_tile')
            ai = encode_action_flat_index(ad, ldt)
            pairs.append((ai, m))
        return pairs

    def _legal_action_indices_reaction_phase(self, gs: GamePerspective, legal_moves: List[Any]) -> List[Tuple[int, Any]]:
        pairs: List[Tuple[int, Any]] = []
        sd = serialize_state(gs)
        for m in legal_moves:
            if isinstance(m, Ron):
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

    def _select_best_legal(self, gs: GamePerspective, legal_moves: List[Any]) -> Optional[Any]:
        if not legal_moves:
            return None
        probs = self.predict_policy_probs(gs)
        # Use legality mask to ensure only legal actions are considered
        mask = gs.legality_mask()

        # Determine phase for mapping legal moves -> flattened indices
        if gs.last_discarded_tile is not None and gs.last_discard_player is not None and gs.last_discard_player != gs.player_id:
            action_pairs = self._legal_action_indices_reaction_phase(gs, legal_moves)
        else:
            action_pairs = self._legal_action_indices_action_phase(gs, legal_moves)

        if not action_pairs:
            return None

        # Score using masked probabilities; choose exact argmax index if present
        if mask is not None and mask.shape[0] == probs.shape[0]:
            masked_probs = np.array(probs, dtype=np.float64)
            masked_probs[~mask] = -np.inf
            target_idx = int(np.argmax(masked_probs))
            for ai, mv in action_pairs:
                if int(ai) == target_idx:
                    return mv
            # Fallback: pick best among legal pairs using masked probs
            best_ai, best_mv = max(action_pairs, key=lambda p: float(masked_probs[p[0]]))
            return best_mv

        # Fallback when mask unavailable: pick best among legal pairs using raw probs
        best_idx, best_move = max(action_pairs, key=lambda p: float(probs[p[0]]))
        return best_move

    # --- Overrides ---
    def play(self, game_state: GamePerspective):
        legal = game_state.legal_moves()
        chosen = self._select_best_legal(game_state, legal)
        move = chosen if chosen is not None else (legal[0] if legal else Discard(game_state.player_hand[0]))
        if not game_state.is_legal(move):
            raise SimpleJong.IllegalMoveException("PurePolicyPlayer produced illegal action")
        return move

    def choose_reaction(self, game_state: GamePerspective, options: Dict[str, List[List[Tile]]]) -> Reaction:
        # Build legal moves from options
        legal: List[Any] = []
        if game_state.can_ron():
            legal.append(Ron())
        for tiles in options.get('pon', []):
            legal.append(Pon(tiles))
        for tiles in options.get('chi', []):
            legal.append(Chi(tiles))
        chosen = self._select_best_legal(game_state, legal)
        # Fallback priority if network yields no signal
        if chosen is None:
            if game_state.can_ron():
                move = Ron()
            elif options.get('pon'):
                move = Pon(options['pon'][0])
            elif options.get('chi'):
                move = Chi(options['chi'][0])
            else:
                # Must return a Reaction; default to Pass
                move = PassCall()
        else:
            move = chosen
        if not game_state.is_legal(move):
            raise SimpleJong.IllegalMoveException("PurePolicyPlayer produced illegal reaction")
        return move

    # --- Exposed inference path for integration consistency checks ---
    def predict_policy_probs(self, gs: GamePerspective) -> np.ndarray:
        """Return flattened action probabilities for the given `GamePerspective`.

        This mirrors the exact encoding pathway used within the player during
        action selection and is intended for tests that must verify that
        recorder-based feature serialization is consistent with inference.
        """
        hand_idx, disc_idx, called_idx, game_state = self._encode_inputs(gs)
        probs = self.network.model.predict([
            hand_idx[None, :],
            disc_idx[None, :, :],
            called_idx[None, :, :, :],
            game_state[None, :]
        ], verbose=0)[0]
        return probs



