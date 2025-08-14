from __future__ import annotations

# Core game constants for simplified Riichi-like rules in this project

# Players
NUM_PLAYERS: int = 4

# Tile system
NUM_SUITS: int = 2           # Pinzu, Souzu
NUM_RANKS: int = 9           # 1..9
TILE_COPIES_DEFAULT: int = 6 # number of identical copies per rank/suit

# Totals
TOTAL_TILES: int = NUM_SUITS * NUM_RANKS * TILE_COPIES_DEFAULT  # 72

# Hand / melds
MAX_HAND_TILES: int = 14
MAX_CALLED_SETS_PER_PLAYER: int = 3
MAX_CALLED_SETS_ALL_OPPONENTS: int = (NUM_PLAYERS - 1) * MAX_CALLED_SETS_PER_PLAYER  # 9
# Number of tiles that define a called set (pon/chi)
MAX_TILES_PER_CALLED_SET: int = 3

# Encodings
EMBEDDING_DIM: int = 4
# Maximum turns per player round (upper bound):
# After initial dealing (11 tiles per player in SimpleJong), remaining draws are TOTAL_TILES - 11*NUM_PLAYERS.
# Each draw is followed by a discard, and up to 4 players can have reaction opportunities.
# We cap the per-player discard/turn sequence vector length as 4 * remaining_draws for headroom.
REMAINING_AFTER_DEAL: int = max(0, TOTAL_TILES - (11 * NUM_PLAYERS))
MAX_TURNS: int = 4 * REMAINING_AFTER_DEAL
# Maximum number of discards we will track per player for learning inputs
MAX_DISCARDS_PER_PLAYER: int = max(1, REMAINING_AFTER_DEAL // NUM_PLAYERS)
GAME_STATE_VEC_LEN: int = 50


