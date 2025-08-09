from __future__ import annotations

# Core game constants for simplified Riichi-like rules in this project

# Players
NUM_PLAYERS: int = 4

# Tile system
NUM_SUITS: int = 2           # Pinzu, Souzu
NUM_RANKS: int = 9           # 1..9
TILE_COPIES_DEFAULT: int = 4 # number of identical copies per rank/suit

# Totals
TOTAL_TILES: int = NUM_SUITS * NUM_RANKS * TILE_COPIES_DEFAULT  # 72

# Hand / melds
MAX_HAND_TILES: int = 14
MAX_CALLED_SETS_PER_PLAYER: int = 4
MAX_CALLED_SETS_ALL_OPPONENTS: int = (NUM_PLAYERS - 1) * MAX_CALLED_SETS_PER_PLAYER  # 12

# Encodings
EMBEDDING_DIM: int = 4
MAX_TURNS: int = 50
GAME_STATE_VEC_LEN: int = 50


