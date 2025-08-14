from __future__ import annotations

# Players / seating
NUM_PLAYERS: int = 4
DEALER_ID_START: int = 0

# Tiles
TILE_COPIES_DEFAULT: int = 4
STARTING_HAND_TILES: int = 13

# Dora/Uradora
INITIAL_DORA_INDICATORS: int = 1
INITIAL_URADORA_INDICATORS: int = 1

# Sorting
SUIT_ORDER = {
    'm': 0,  # Manzu
    'p': 1,  # Pinzu
    's': 2,  # Souzu
    'z': 3,  # Honors
}

# Scoring constants (simplified riichi-like)
FU_CHIITOI: int = 25
FU_BASELINE: int = 30
POINTS_ROUNDING: int = 100

# Yaku han values (simplified)
CHANTA_OPEN_HAN: int = 1
CHANTA_CLOSED_HAN: int = 2
JUNCHAN_OPEN_HAN: int = 2
JUNCHAN_CLOSED_HAN: int = 3
SANANKOU_HAN: int = 2


