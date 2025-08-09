from __future__ import annotations

from .game import Tile, Suit, TileType


def tile_to_index(tile: Tile) -> int:
    """Map a Tile to an integer index in [0..17]: 9 ranks x 2 suits (Pinzu first).

    Index = (rank-1)*2 + suit_index, where suit_index is 0 for Pinzu, 1 for Souzu.
    """
    return (tile.tile_type.value - 1) * 2 + (0 if tile.suit == Suit.PINZU else 1)


def tile_str_to_index(tile_str: str) -> int:
    """Map a tile string like '3p' or '7s' to the same [0..17] index."""
    rank = int(tile_str[:-1])
    suit = Suit(tile_str[-1])
    return (rank - 1) * 2 + (0 if suit == Suit.PINZU else 1)


