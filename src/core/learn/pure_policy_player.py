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
)
from .pure_policy import PurePolicyNetwork
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

    def __init__(self, player_id: int, network: PurePolicyNetwork):
        super().__init__(player_id)
        self.network = network
        self._action_index = get_action_index_map()

    # --- Model input encoding (indices) ---
    def _encode_inputs(self, gs: GamePerspective) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Reuse dataset encoders via serialization path
        sd = serialize_state(gs)
        idx = extract_indexed_state(sd)
        hand_idx = idx['hand_idx']
        disc_idx = idx['disc_idx']
        game_state = idx['game_state']
        return hand_idx, disc_idx, game_state

    # --- Mapping legal moves to flattened indices ---
    def _legal_action_indices_action_phase(self, gs: GamePerspective, legal_moves: List[Any]) -> List[Tuple[int, Any]]:
        pairs: List[Tuple[int, Any]] = []
        sd = serialize_state(gs)
        for m in legal_moves:
            ad = {'type': 'tsumo'} if isinstance(m, Tsumo) else (
                {'type': 'discard', 'tile': str(m.tile)} if isinstance(m, Discard) else None)
            if ad is None:
                continue
            ai = encode_action_flat_index(sd, ad)
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
            ai = encode_action_flat_index(sd, ad)
            pairs.append((ai, m))
        return pairs

    def _select_best_legal(self, gs: GamePerspective, legal_moves: List[Any]) -> Optional[Any]:
        if not legal_moves:
            return None
        hand_idx, disc_idx, game_state = self._encode_inputs(gs)
        probs = self.network.model.predict([hand_idx[None, :], disc_idx[None, :, :], game_state[None, :]], verbose=0)[0]

        # Determine phase: if last_discard exists and not from self -> reaction phase
        if gs.last_discarded_tile is not None and gs.last_discard_player is not None and gs.last_discard_player != gs.player_id:
            action_pairs = self._legal_action_indices_reaction_phase(gs, legal_moves)
        else:
            action_pairs = self._legal_action_indices_action_phase(gs, legal_moves)

        if not action_pairs:
            return None

        # Pick the legal action with highest probability
        best_idx, best_move = max(action_pairs, key=lambda p: float(probs[p[0]]))
        return best_move

    # --- Overrides ---
    def play(self, game_state: GamePerspective):
        # Current player's action phase legal moves
        if self._game is None:
            return Discard(game_state.player_hand[0])
        legal = self._game.legal_moves(self.player_id)
        chosen = self._select_best_legal(game_state, legal)
        return chosen if chosen is not None else (legal[0] if legal else Discard(game_state.player_hand[0]))

    def choose_reaction(self, game_state: GamePerspective, options: Dict[str, List[List[Tile]]]):
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
                return Ron()
            if options.get('pon'):
                return Pon(options['pon'][0])
            if options.get('chi'):
                return Chi(options['chi'][0])
        return chosen


