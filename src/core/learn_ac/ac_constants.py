from __future__ import annotations

# Canonical order for AC main policy head
# Indices correspond to logits/probabilities emitted by ACNetwork.head_action
MAIN_HEAD_ORDER = ['chi', 'pon', 'ron', 'tsumo', 'discard', 'pass']

# Mapping from action type string to main head index
MAIN_HEAD_INDEX = {name: idx for idx, name in enumerate(MAIN_HEAD_ORDER)}


def action_type_to_main_index(action_type: str) -> int:
    """Map a serialized action type to the main head index.

    Falls back to 'pass' on unknown types.
    """
    at = (action_type or 'pass').lower()
    return MAIN_HEAD_INDEX.get(at, MAIN_HEAD_INDEX['pass'])


def chi_variant_index(last_discarded_tile: 'Tile', tiles: list['Tile']) -> int:
    """Return chi variant index relative to the last discarded tile.

    0: left (low)  -> tiles ranks == [d-2, d-1]
    1: mid         -> tiles ranks == [d-1, d+1]
    2: right (high)-> tiles ranks == [d+1, d+2]

    Returns -1 if not a recognizable chi pair relative to last.
    """
    # Local import to avoid circulars
    try:
        from ..game import Tile  # type: ignore
    except Exception:
        Tile = object  # type: ignore
    if last_discarded_tile is None or len(tiles) < 2:
        return -1
    d = int(getattr(last_discarded_tile.tile_type, 'value', 0))
    try:
        ranks = sorted(int(getattr(t.tile_type, 'value', 0)) for t in tiles)
    except Exception:
        return -1
    if ranks == [d - 2, d - 1]:
        return 0
    if ranks == [d - 1, d + 1]:
        return 1
    if ranks == [d + 1, d + 2]:
        return 2
    return -1


