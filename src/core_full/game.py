import random
import math
import numpy as np
from typing import List, Optional, Dict, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import copy

# Import TensorFlow/Keras
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not available. PQNetwork will not work without TensorFlow.")


class Suit(Enum):
    """Enum for tile suits"""
    MANZU = 'm'
    PINZU = 'p'
    SOUZU = 's'
    HONOR = 'z'  # Winds and Dragons


class TileType(Enum):
    """Enum for the 9 types of tiles (1-9).
    For HONOR suit, values 1..7 map to: East, South, West, North, White, Green, Red.
    """
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9


@dataclass
class Tile:
    """Represents a single tile"""
    suit: Suit
    tile_type: TileType
    
    def __str__(self):
        return f"{self.tile_type.value}{self.suit.value}"
    
    def __eq__(self, other):
        if isinstance(other, Tile):
            return self.suit == other.suit and self.tile_type == other.tile_type
        return False
    
    def __hash__(self):
        return hash((self.suit, self.tile_type))


@dataclass
class Action:
    """Represents a player action"""
    pass


@dataclass
class Tsumo(Action):
    """Action to declare a win (tsumo)"""
    pass


@dataclass
class Ron(Action):
    """Action to declare a win on another player's discard (ron)"""
    pass


@dataclass
class Discard(Action):
    """Action to discard a tile"""
    tile: Tile


@dataclass
class Pon(Action):
    """Action to call pon (take discarded tile to complete triplet)"""
    tiles: List[Tile]  # The two tiles from hand to complete the triplet
    

@dataclass
class Chi(Action):
    """Action to call chi (take discarded tile from left player to complete sequence)"""
    tiles: List[Tile]  # The two tiles from hand to complete the sequence


@dataclass
class PassCall(Action):
    """Action to pass on a call opportunity"""
    pass


@dataclass
class Riichi(Action):
    """Action to declare Riichi (must accompany a discard)"""
    discard_tile: 'Tile'


@dataclass
class CalledSet:
    """Represents a called set (pon or chi)"""
    tiles: List[Tile]  # All 3 tiles in the set
    call_type: str  # "pon" or "chi"
    called_tile: Tile  # The tile that was called
    caller_position: int  # Position of the calling player
    source_position: int  # Position of the player who discarded the tile


@dataclass
class GameState:
    """Container for game state information available to a player"""
    player_hand: List[Tile]
    visible_tiles: List[Tile]  # Tiles on the table
    remaining_tiles: int
    player_id: int
    other_players_discarded: Dict[int, List[Tile]]  # Other players' discarded tiles
    called_sets: Dict[int, List[CalledSet]]  # Called sets by player
    last_discarded_tile: Optional[Tile] = None
    last_discard_player: Optional[int] = None
    can_call: bool = False  # Whether this player can make a call on the last discarded tile
    # Extended Riichi state
    round_wind: str = 'E'  # 'E' or 'S'
    dealer_idx: int = 0
    seat_winds: Dict[int, str] = field(default_factory=dict)  # {player_id: 'E'|'S'|'W'|'N'}
    points: List[int] = field(default_factory=lambda: [25000, 25000, 25000, 25000])
    riichi_declared: Dict[int, bool] = field(default_factory=dict)


class Player:
    """Base player class"""
    
    def __init__(self, player_id: int):
        self.player_id = player_id
        self.hand: List[Tile] = []
        self.called_sets: List[CalledSet] = []
        self.riichi_declared: bool = False
    
    def play(self, game_state: GameState) -> Action:
        """
        Player's turn - returns an Action (Tsumo, Ron, or Discard)
        First checks if the player can win with the current hand
        """
        if not self.hand:
            return Tsumo()  # If no tiles, can't do anything but declare win
        
        # Check if we can win with current hand (Tsumo). Engine will validate yaku.
        if self.can_win():
            return Tsumo()
        
        # Check if we can win with any of the recently discarded tiles (Ron)
        # Look at the most recently discarded tile (last tile in visible_tiles)
        if game_state.visible_tiles:
            last_discarded = game_state.visible_tiles[-1]
            if self.can_ron(last_discarded):
                return Ron()
        
        # If we can't win, discard a random tile
        return Discard(random.choice(self.hand))
    
    def can_call(self, game_state: GameState) -> Dict[str, List[List[Tile]]]:
        """
        Check if the player can make any calls (pon/chi) on the last discarded tile
        Returns a dict with 'pon' and 'chi' keys, each containing a list of possible tile combinations
        """
        if not game_state.last_discarded_tile:
            return {'pon': [], 'chi': []}
        
        discarded = game_state.last_discarded_tile
        results = {'pon': [], 'chi': []}
        
        # Check for pon (can call from any player)
        pon_combinations = self.get_pon_combinations(discarded)
        results['pon'] = pon_combinations
        
        # Check for chi (can only call from left player - previous player in turn order)
        # In this implementation, "left" means the previous player in turn order
        left_player_id = (self.player_id - 1) % 4
        if game_state.last_discard_player == left_player_id:
            chi_combinations = self.get_chi_combinations(discarded)
            results['chi'] = chi_combinations
        
        return results

    def is_closed_hand(self) -> bool:
        return len(self.called_sets) == 0

    def get_riichi_discard_options(self, game_state: GameState) -> List[Tile]:
        """Return tiles that can be discarded to declare riichi (closed tenpai)."""
        if not self.is_closed_hand():
            return []
        # Must have 14 tiles at start of turn (drawn tile in hand)
        if len(self.hand) != 14:
            return []
        riichi_discards: List[Tile] = []
        for idx, discard_tile in enumerate(list(self.hand)):
            # simulate discard
            self.remove_tile(discard_tile)
            # now check if in tenpai: exists any draw that wins structurally
            in_tenpai = False
            for suit in [Suit.MANZU, Suit.PINZU, Suit.SOUZU, Suit.HONOR]:
                max_val = 9 if suit != Suit.HONOR else 7
                for val in range(1, max_val + 1):
                    test_tile = Tile(suit, TileType(val))
                    self.add_tile(test_tile)
                    if self.can_win():
                        in_tenpai = True
                        self.remove_tile(test_tile)
                        break
                    self.remove_tile(test_tile)
                if in_tenpai:
                    break
            # revert discard
            self.add_tile(discard_tile)
            if in_tenpai:
                riichi_discards.append(discard_tile)
        return riichi_discards
    
    def get_pon_combinations(self, discarded_tile: Tile) -> List[List[Tile]]:
        """Get all possible pon combinations with the discarded tile"""
        combinations = []
        
        # Count how many of the discarded tile we have in hand
        matching_tiles = [tile for tile in self.hand if tile == discarded_tile]
        
        # Need at least 2 matching tiles to make a pon
        if len(matching_tiles) >= 2:
            combinations.append(matching_tiles[:2])  # Take first 2 matching tiles
        
        return combinations
    
    def get_chi_combinations(self, discarded_tile: Tile) -> List[List[Tile]]:
        """Get all possible chi combinations with the discarded tile"""
        combinations = []
        
        # Chi only works with number tiles (not honors)
        if discarded_tile.suit == Suit.HONOR:
            return combinations
        
        discarded_value = discarded_tile.tile_type.value
        suit = discarded_tile.suit
        
        # Group our hand tiles by suit
        suited_tiles = [tile for tile in self.hand if tile.suit == suit]
        tile_counts = {}
        for tile in suited_tiles:
            value = tile.tile_type.value
            tile_counts[value] = tile_counts.get(value, 0) + 1
        
        # Check for sequences: ABC where B is the discarded tile
        # Pattern 1: We have A and C, discarded is B (e.g., we have 1,3 and discard is 2)
        if (discarded_value - 1 >= 1 and discarded_value + 1 <= 9 and
            tile_counts.get(discarded_value - 1, 0) > 0 and
            tile_counts.get(discarded_value + 1, 0) > 0):
            
            # Find the actual tiles
            low_tile = next(t for t in suited_tiles if t.tile_type.value == discarded_value - 1)
            high_tile = next(t for t in suited_tiles if t.tile_type.value == discarded_value + 1)
            combinations.append([low_tile, high_tile])
        
        # Pattern 2: We have A and B, discarded is C (e.g., we have 1,2 and discard is 3)
        if (discarded_value - 2 >= 1 and
            tile_counts.get(discarded_value - 2, 0) > 0 and
            tile_counts.get(discarded_value - 1, 0) > 0):
            
            low_tile = next(t for t in suited_tiles if t.tile_type.value == discarded_value - 2)
            mid_tile = next(t for t in suited_tiles if t.tile_type.value == discarded_value - 1)
            combinations.append([low_tile, mid_tile])
        
        # Pattern 3: We have B and C, discarded is A (e.g., we have 2,3 and discard is 1)
        if (discarded_value + 2 <= 9 and
            tile_counts.get(discarded_value + 1, 0) > 0 and
            tile_counts.get(discarded_value + 2, 0) > 0):
            
            mid_tile = next(t for t in suited_tiles if t.tile_type.value == discarded_value + 1)
            high_tile = next(t for t in suited_tiles if t.tile_type.value == discarded_value + 2)
            combinations.append([mid_tile, high_tile])
        
        return combinations
    
    def make_call(self, call_type: str, tiles: List[Tile], discarded_tile: Tile, source_player: int):
        """Make a pon or chi call"""
        # Remove the tiles from hand
        for tile in tiles:
            self.remove_tile(tile)
        
        # Create the called set
        all_tiles = tiles + [discarded_tile]
        called_set = CalledSet(
            tiles=all_tiles,
            call_type=call_type,
            called_tile=discarded_tile,
            caller_position=self.player_id,
            source_position=source_player
        )
        
        self.called_sets.append(called_set)
    
    def add_tile(self, tile: Tile):
        """Add a tile to player's hand"""
        self.hand.append(tile)
    
    def remove_tile(self, tile: Tile):
        """Remove a tile from player's hand"""
        if tile in self.hand:
            self.hand.remove(tile)
    
    def can_win(self) -> bool:
        """
        Structural check only: 4 melds + 1 pair using current hand + called sets.
        Does not check yaku; engine will enforce yaku for win validity.
        """
        total_called_sets = len(self.called_sets)
        remaining_melds_needed = 4 - total_called_sets
        # Structural hand size should be: 14 - 3*called_sets
        if len(self.hand) != 14 - 3 * total_called_sets:
            return False
        # Try all possible pairs in concealed hand, then check meldability of remainder
        tile_counts: Dict[Tuple[Suit, int], int] = {}
        for t in self.hand:
            key = (t.suit, t.tile_type.value)
            tile_counts[key] = tile_counts.get(key, 0) + 1
        # Prepare counters per suit for meldability test
        def can_form_melds_without_pair(counts: Dict[Tuple[Suit, int], int], melds_needed: int) -> bool:
            # Flatten by suit: numbers allow runs, honors only triplets
            if melds_needed == 0:
                # all remaining tiles must be zero
                return all(c == 0 for c in counts.values())
            # Find first tile with count>0
            for (suit, val), cnt in list(counts.items()):
                if cnt > 0:
                    # Try triplet
                    if cnt >= 3:
                        counts[(suit, val)] -= 3
                        if can_form_melds_without_pair(counts, melds_needed - 1):
                            counts[(suit, val)] += 3
                            return True
                        counts[(suit, val)] += 3
                    # Try run for number suits only
                    if suit != Suit.HONOR and 1 <= val <= 7:
                        if counts.get((suit, val), 0) > 0 and counts.get((suit, val + 1), 0) > 0 and counts.get((suit, val + 2), 0) > 0:
                            counts[(suit, val)] -= 1
                            counts[(suit, val + 1)] -= 1
                            counts[(suit, val + 2)] -= 1
                            if can_form_melds_without_pair(counts, melds_needed - 1):
                                counts[(suit, val)] += 1
                                counts[(suit, val + 1)] += 1
                                counts[(suit, val + 2)] += 1
                                return True
                            counts[(suit, val)] += 1
                            counts[(suit, val + 1)] += 1
                            counts[(suit, val + 2)] += 1
                    # If neither works, no solution from this branching
                    return False
            return False
        # Try every possible pair position
        for (suit, val), cnt in list(tile_counts.items()):
            if cnt >= 2:
                tile_counts[(suit, val)] -= 2
                if can_form_melds_without_pair(tile_counts, remaining_melds_needed):
                    tile_counts[(suit, val)] += 2
                    return True
                tile_counts[(suit, val)] += 2
        return False

    def can_ron(self, discarded_tile: Tile) -> bool:
        """
        Check if the player can declare Ron with the given discarded tile
        This is similar to can_win() but includes the discarded tile in the hand
        """
        # Temporarily add the discarded tile to check if we can win
        self.add_tile(discarded_tile)
        can_win_with_tile = self.can_win()
        self.remove_tile(discarded_tile)  # Remove it back
        return can_win_with_tile
    
    def can_tsumo(self) -> bool:
        """
        Check if the player can declare Tsumo (win with current hand)
        This is an alias for can_win() for clarity in web interface
        """
        return self.can_win()

    def get_possible_actions(self, game_state: GameState) -> Dict[str, List]:
        """
        Get all possible actions for this player given the current game state.
        Returns a dict with keys: 'tsumo', 'ron', 'pon', 'chi', 'riichi'
        """
        actions = {
            'tsumo': [],
            'ron': [],
            'pon': [],
            'chi': [],
            'riichi': []
        }
        
        # Check for Tsumo (can win with current hand on player's turn)
        if self.can_win():
            actions['tsumo'] = [True]
        
        # Check for Ron, Pon, Chi (actions on other players' discards)
        if game_state.last_discarded_tile and game_state.last_discard_player != self.player_id:
            discarded_tile = game_state.last_discarded_tile
            
            # Check Ron
            if self.can_ron(discarded_tile):
                actions['ron'] = [True]
            
            # Check Pon and Chi using existing can_call method
            call_actions = self.can_call(game_state)
            # Convert Tile objects to strings for JSON serialization
            actions['pon'] = [[str(tile) for tile in tiles] for tiles in call_actions['pon']]
            actions['chi'] = [[str(tile) for tile in tiles] for tiles in call_actions['chi']]
        
        # Check Riichi options (closed tenpai) if not already declared
        if not getattr(self, 'riichi_declared', False):
            riichi_discards = self.get_riichi_discard_options(game_state)
            actions['riichi'] = [str(t) for t in riichi_discards]

        return actions
    
    def _can_form_sets(self, tile_counts: Dict[int, int], sets_needed: int) -> bool:
        """
        Recursively check if we can form the required number of sets
        from the remaining tiles
        """
        if sets_needed == 0:
            # Check if all tiles are used
            return all(count == 0 for count in tile_counts.values())
        
        # Find the first tile type that still has tiles
        for tile_value in sorted(tile_counts.keys()):
            if tile_counts.get(tile_value, 0) > 0:
                # Try forming a triplet first
                if tile_counts[tile_value] >= 3:
                    # Form a triplet
                    tile_counts[tile_value] -= 3
                    if self._can_form_sets(tile_counts, sets_needed - 1):
                        tile_counts[tile_value] += 3  # backtrack
                        return True
                    tile_counts[tile_value] += 3  # backtrack
                
                # Try forming a run (only if we have consecutive tiles)
                if (tile_value <= 7 and  # Can't start a run with 8 or 9
                    tile_counts.get(tile_value, 0) > 0 and
                    tile_counts.get(tile_value + 1, 0) > 0 and
                    tile_counts.get(tile_value + 2, 0) > 0):
                    
                    # Form a run
                    tile_counts[tile_value] -= 1
                    tile_counts[tile_value + 1] -= 1
                    tile_counts[tile_value + 2] -= 1
                    
                    if self._can_form_sets(tile_counts, sets_needed - 1):
                        # backtrack
                        tile_counts[tile_value] += 1
                        tile_counts[tile_value + 1] += 1
                        tile_counts[tile_value + 2] += 1
                        return True
                    
                    # backtrack
                    tile_counts[tile_value] += 1
                    tile_counts[tile_value + 1] += 1
                    tile_counts[tile_value + 2] += 1
                
                # If we can't form any set starting with this tile, return False
                return False
        
        # No tiles left but still need sets - shouldn't happen with valid input
        return False


class FullRiichi:
    """Full Riichi Mahjong engine (separate from SimpleJong).
    Implements:
    - Full tileset (manzu, pinzu, souzu, honors)
    - 14-tile hands (4 melds + 1 pair for a win)
    - Hanchan rounds (East 1-4, South 1-4); simplified dealer rotation
    - Initial yaku detection: Riichi, Tanyao, Yakuhai, Honitsu, Chinitsu
    - Simplified scoring: +2000 to winner (placeholder)
    """
    
    def __init__(self, players: List[Player]):
        if len(players) != 4:
            raise ValueError("FullRiichi requires exactly 4 players")
        
        self.players = players
        self.tiles: List[Tile] = []
        self.discarded_tiles: List[Tile] = []
        self.current_player_idx = 0
        self.game_over = False
        self.winner = None
        self.last_discarded_tile = None
        self.last_discard_player = None
        # Riichi/hanchan state
        self.round_wind: str = 'E'  # 'E' or 'S'
        self.hand_number: int = 1   # 1..4
        self.dealer_idx: int = 0    # East seat index
        self.honba: int = 0
        self.riichi_sticks: int = 0
        self.points: List[int] = [25000, 25000, 25000, 25000]
        # Initialize a fresh hand
        self._init_tileset()
        self._deal_new_hand()

    def _init_tileset(self):
        """Create full Riichi tileset (136 tiles)."""
        self.tiles = []
        # Numbered suits m/p/s: 1..9, 4 copies
        for suit in [Suit.MANZU, Suit.PINZU, Suit.SOUZU]:
            for tile_type in TileType:
                if 1 <= tile_type.value <= 9:
                    for _ in range(4):
                        self.tiles.append(Tile(suit, tile_type))
        # Honors (z): 1..7, 4 copies (map to TileType 1..7)
        for honor_val in range(1, 8):
            for _ in range(4):
                self.tiles.append(Tile(Suit.HONOR, TileType(honor_val)))
        random.shuffle(self.tiles)

    def _deal_new_hand(self):
        """Deal 13 tiles to each player and reset per-hand state."""
        # Reset per-hand state
        for p in self.players:
            p.hand = []
            p.called_sets = []
            p.riichi_declared = False
        self.discarded_tiles = []
        self.current_player_idx = self.dealer_idx
        self.game_over = False
        self.winner = None
        self.last_discarded_tile = None
        self.last_discard_player = None
        # Shuffle if tiles insufficient (start new wall)
        if len(self.tiles) < 52:  # safety: ensure enough tiles
            self._init_tileset()
        # Deal 13 each
        for player in self.players:
            for _ in range(13):
                tile = self.tiles.pop()
                player.add_tile(tile)

    def _get_seat_wind(self, player_idx: int) -> str:
        offset = (player_idx - self.dealer_idx) % 4
        return ['E', 'S', 'W', 'N'][offset]

    def _build_game_state(self, player_id: int) -> GameState:
        other_players_discarded = {}
        for i, _ in enumerate(self.players):
            if i != player_id:
                other_players_discarded[i] = []
        called_sets = {i: p.called_sets.copy() for i, p in enumerate(self.players)}
        seat_winds = {i: self._get_seat_wind(i) for i in range(4)}
        riichi_declared = {i: p.riichi_declared for i, p in enumerate(self.players)}
        return GameState(
            player_hand=self.players[player_id].hand.copy(),
            visible_tiles=self.discarded_tiles.copy(),
            remaining_tiles=len(self.tiles),
            player_id=player_id,
            other_players_discarded=other_players_discarded,
            called_sets=called_sets,
            last_discarded_tile=self.last_discarded_tile,
            last_discard_player=self.last_discard_player,
            can_call=self.last_discarded_tile is not None and self.last_discard_player != player_id,
            round_wind=self.round_wind,
            dealer_idx=self.dealer_idx,
            seat_winds=seat_winds,
            points=self.points.copy(),
            riichi_declared=riichi_declared,
        )
    
    def get_game_state(self, player_id: int) -> GameState:
        """Get extended game state for a specific player"""
        return self._build_game_state(player_id)
    
    def check_for_calls(self, discarded_tile: Tile, discard_player: int) -> Optional[Tuple[int, str, List[Tile]]]:
        """
        Check if any player can make a call on the discarded tile
        Returns (player_id, call_type, tiles) for the highest priority call, or None
        Priority: Pon > Chi (and pon can be called by any player, chi only by left player)
        """
        self.last_discarded_tile = discarded_tile
        self.last_discard_player = discard_player
        
        call_results = []
        
        # Check all other players for possible calls
        for i, player in enumerate(self.players):
            if i != discard_player:  # Can't call your own discard
                game_state = self.get_game_state(i)
                possible_calls = player.can_call(game_state)
                
                # Add pon calls (higher priority)
                for tiles in possible_calls['pon']:
                    call_results.append((i, 'pon', tiles))
                
                # Add chi calls (lower priority)
                for tiles in possible_calls['chi']:
                    call_results.append((i, 'chi', tiles))
        
        # If there are any calls, prioritize pon over chi
        # For now, just take the first pon if available, otherwise first chi
        pon_calls = [call for call in call_results if call[1] == 'pon']
        chi_calls = [call for call in call_results if call[1] == 'chi']
        
        if pon_calls:
            return pon_calls[0]
        elif chi_calls:
            return chi_calls[0]
        
        return None
    
    def make_call(self, player_id: int, call_type: str, tiles: List[Tile]):
        """Execute a call by a player"""
        player = self.players[player_id]
        player.make_call(call_type, tiles, self.last_discarded_tile, self.last_discard_player)
        
        # Remove the discarded tile from the discard pile since it was called
        if self.last_discarded_tile in self.discarded_tiles:
            self.discarded_tiles.remove(self.last_discarded_tile)
        
        # The calling player becomes the current player
        self.current_player_idx = player_id
        
        # Clear the last discard info
        self.last_discarded_tile = None
        self.last_discard_player = None
    
    def play_hand(self) -> Optional[int]:
        """
        Play one hand of the game
        Returns the winner's player_id, or None if no winner
        """
        while not self.game_over and self.tiles:
            current_player = self.players[self.current_player_idx]
            
            # 1. Draw a tile
            new_tile = self.tiles.pop()
            current_player.add_tile(new_tile)
            
            # 2. Get game state and let player decide what to do
            game_state = self.get_game_state(self.current_player_idx)
            action = current_player.play(game_state)
            
            # 3. Handle player's decision
            if isinstance(action, Tsumo):
                # Validate yaku then award
                if self._has_any_yaku(self.current_player_idx, is_ron=False):
                    self.winner = self.current_player_idx
                    self._award_win_points(self.current_player_idx, is_ron=False)
                    self.game_over = True
                    return self.winner
            elif isinstance(action, Ron):
                # Validate yaku then award
                if self.last_discarded_tile is not None and self._has_any_yaku(self.current_player_idx, is_ron=True):
                    self.winner = self.current_player_idx
                    self._award_win_points(self.current_player_idx, is_ron=True)
                    self.game_over = True
                    return self.winner
            elif isinstance(action, Discard):
                # Player discarded a tile
                current_player.remove_tile(action.tile)
                self.discarded_tiles.append(action.tile)
                
                # Check if another player can win with the discarded tile (Ron)
                for i, player in enumerate(self.players):
                    if i != self.current_player_idx:
                        if player.can_ron(action.tile) and self._has_any_yaku(i, is_ron=True):
                            self.winner = i
                            self._award_win_points(i, is_ron=True)
                            self.game_over = True
                            # The tile is "claimed" by the Ron winner, so add it to their hand
                            player.add_tile(action.tile)
                            return self.winner
                
                # Check for calls (pon/chi) after Ron check
                call_result = self.check_for_calls(action.tile, self.current_player_idx)
                if call_result:
                    caller_id, call_type, tiles = call_result
                    # For now, AI players make calls randomly if they can
                    # This will be refined later for human player input
                    if caller_id != 0:  # AI player
                        self.make_call(caller_id, call_type, tiles)
                        # Don't advance turn - the calling player continues
                        continue
                    # If human player can call, the web interface will handle it
            
            # Move to next player
            self.current_player_idx = (self.current_player_idx + 1) % 4
        
        self.game_over = True
        return None

    def play_hanchan(self):
        """Play a full hanchan (E1-E4, S1-S4). Simplified dealer rotation without honba.
        Returns final points list.
        """
        self.round_wind = 'E'
        self.hand_number = 1
        self.dealer_idx = 0
        while True:
            # Reset tiles and deal
            self._init_tileset()
            self._deal_new_hand()
            winner = self.play_hand()
            # Dealer repeats only if dealer wins (simplified; ignores draws/honba)
            dealer_won = (winner == self.dealer_idx)
            if not dealer_won:
                # rotate dealer
                self.dealer_idx = (self.dealer_idx + 1) % 4
                if self.hand_number == 4:
                    if self.round_wind == 'E':
                        self.round_wind = 'S'
                        self.hand_number = 1
                    else:
                        # Hanchan ends after South 4
                        break
                else:
                    self.hand_number += 1
            # If dealer won, repeat same hand number and round wind
        return self.points

    def _award_win_points(self, winner_idx: int, is_ron: bool):
        """Placeholder scoring: +2000 to winner. No deductions for now."""
        try:
            self.points[winner_idx] += 2000
        except Exception:
            pass

    def _has_any_yaku(self, player_idx: int, is_ron: bool) -> bool:
        """Check if the player has any of the initial yaku on the (structurally) winning hand.
        Yaku considered: Riichi, Tanyao, Yakuhai, Honitsu, Chinitsu.
        """
        tiles = self.players[player_idx].hand.copy()
        # Combine with called sets tiles
        for cs in self.players[player_idx].called_sets:
            tiles.extend(cs.tiles)
        # Riichi
        yaku_found = False
        if self.players[player_idx].riichi_declared:
            yaku_found = True
        # Tanyao: all tiles are simples (2..8) and no honors
        if all(t.suit != Suit.HONOR and 2 <= t.tile_type.value <= 8 for t in tiles):
            yaku_found = True
        # Yakuhai: triplet of round wind, seat wind, or any dragon
        if self._has_yakuhai(player_idx):
            yaku_found = True
        # Honitsu: tiles only from one suit plus honors (at least one from that suit)
        if self._is_honitsu(player_idx):
            yaku_found = True
        # Chinitsu: tiles only from one numbered suit (no honors)
        if self._is_chinitsu(player_idx):
            yaku_found = True
        return yaku_found

    def _has_yakuhai(self, player_idx: int) -> bool:
        tiles = self.players[player_idx].hand.copy()
        for cs in self.players[player_idx].called_sets:
            tiles.extend(cs.tiles)
        counts: Dict[Tuple[Suit, int], int] = {}
        for t in tiles:
            key = (t.suit, t.tile_type.value)
            counts[key] = counts.get(key, 0) + 1
        # Round wind
        round_map = {'E': 1, 'S': 2, 'W': 3, 'N': 4}
        round_val = round_map[self.round_wind]
        if counts.get((Suit.HONOR, round_val), 0) >= 3:
            return True
        # Seat wind
        seat_val = round_map[self._get_seat_wind(player_idx)]
        if counts.get((Suit.HONOR, seat_val), 0) >= 3:
            return True
        # Dragons: White(5), Green(6), Red(7)
        for d in [5, 6, 7]:
            if counts.get((Suit.HONOR, d), 0) >= 3:
                return True
        return False

    def _is_honitsu(self, player_idx: int) -> bool:
        tiles = self.players[player_idx].hand.copy()
        for cs in self.players[player_idx].called_sets:
            tiles.extend(cs.tiles)
        suits = set(t.suit for t in tiles)
        number_suits = {s for s in suits if s in (Suit.MANZU, Suit.PINZU, Suit.SOUZU)}
        # one number suit + optionally HONOR
        return (len(number_suits) == 1) and (Suit.HONOR in suits)

    def _is_chinitsu(self, player_idx: int) -> bool:
        tiles = self.players[player_idx].hand.copy()
        for cs in self.players[player_idx].called_sets:
            tiles.extend(cs.tiles)
        suits = set(t.suit for t in tiles)
        number_suits = {s for s in suits if s in (Suit.MANZU, Suit.PINZU, Suit.SOUZU)}
        return (len(number_suits) == 1) and (Suit.HONOR not in suits)
    
    def get_winner(self) -> Optional[int]:
        """Get the winner's player_id, or None if no winner"""
        return self.winner
    
    def is_game_over(self) -> bool:
        """Check if the game is over"""
        return self.game_over
    
    def get_remaining_tiles(self) -> int:
        """Get number of remaining tiles"""
        return len(self.tiles)


class MCTSNode:
    """Node in the Monte Carlo Tree Search tree"""
    
    def __init__(self, game_state: 'FullRiichi', player_id: int, parent=None, action=None):
        self.game_state = game_state
        self.player_id = player_id
        self.parent = parent
        self.action = action  # Action that led to this node
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried_actions = self._get_untried_actions()
        
    def _get_untried_actions(self) -> List[Action]:
        """Get all possible actions for the current player"""
        if self.game_state.is_game_over():
            return []
        
        current_player = self.game_state.players[self.player_id]
        game_state_for_player = self.game_state.get_game_state(self.player_id)
        
        actions = []
        
        # Check for Tsumo
        if current_player.can_win():
            actions.append(Tsumo())
        
        # Check for Ron
        if (self.game_state.last_discarded_tile and 
            self.game_state.last_discard_player != self.player_id):
            if current_player.can_ron(self.game_state.last_discarded_tile):
                actions.append(Ron())
        
        # Check for Pon/Chi calls
        if self.game_state.last_discarded_tile and self.game_state.last_discard_player != self.player_id:
            call_actions = current_player.can_call(game_state_for_player)
            for tiles in call_actions['pon']:
                actions.append(Pon(tiles))
            for tiles in call_actions['chi']:
                actions.append(Chi(tiles))
        
        # Add discard actions (discard each tile in hand)
        for tile in current_player.hand:
            actions.append(Discard(tile))
        
        return actions
    
    def is_terminal(self) -> bool:
        """Check if this is a terminal node (game over)"""
        return self.game_state.is_game_over()
    
    def is_fully_expanded(self) -> bool:
        """Check if all possible actions have been tried"""
        return len(self.untried_actions) == 0
    
    def select_child(self, exploration_constant: float = 1.414) -> 'MCTSNode':
        """Select a child node using UCB1 formula"""
        if not self.children:
            return None
        
        # UCB1 formula: argmax(vi + c * sqrt(ln(N) / ni))
        best_child = None
        best_score = float('-inf')
        
        for child in self.children:
            if child.visits == 0:
                return child
            
            # Calculate UCB1 score
            exploitation = child.value / child.visits
            exploration = exploration_constant * math.sqrt(math.log(self.visits) / child.visits)
            ucb_score = exploitation + exploration
            
            if ucb_score > best_score:
                best_score = ucb_score
                best_child = child
        
        return best_child
    
    def expand(self) -> 'MCTSNode':
        """Expand the tree by adding a new child node"""
        if not self.untried_actions:
            return None
        
        # Select a random untried action
        action = random.choice(self.untried_actions)
        self.untried_actions.remove(action)
        
        # Create a copy of the game state
        new_game_state = self._copy_game_state()
        
        # Apply the action
        self._apply_action(new_game_state, action, self.player_id)
        
        # Create child node
        next_player_id = (self.player_id - 1) % 4  # Counterclockwise order
        child = MCTSNode(new_game_state, next_player_id, parent=self, action=action)
        self.children.append(child)
        
        return child
    
    def _copy_game_state(self) -> 'FullRiichi':
        """Create a deep copy of the game state"""
        # Create a new FullRiichi instance with copied players
        copied_players = []
        for player in self.game_state.players:
            # Create a new player of the same type
            if isinstance(player, AIPlayer):
                new_player = AIPlayer(player.player_id, player.simulation_count, player.exploration_constant)
            else:
                new_player = Player(player.player_id)
            
            # Copy hand and called sets
            new_player.hand = player.hand.copy()
            new_player.called_sets = player.called_sets.copy()
            new_player.riichi_declared = player.riichi_declared
            copied_players.append(new_player)
        
        # Create new game instance
        new_game = FullRiichi(copied_players)
        
        # Copy game state
        new_game.tiles = self.game_state.tiles.copy()
        new_game.discarded_tiles = self.game_state.discarded_tiles.copy()
        new_game.current_player_idx = self.game_state.current_player_idx
        new_game.game_over = self.game_state.game_over
        new_game.winner = self.game_state.winner
        new_game.last_discarded_tile = self.game_state.last_discarded_tile
        new_game.last_discard_player = self.game_state.last_discard_player
        new_game.round_wind = self.game_state.round_wind
        new_game.hand_number = self.game_state.hand_number
        new_game.dealer_idx = self.game_state.dealer_idx
        new_game.honba = self.game_state.honba
        new_game.riichi_sticks = self.game_state.riichi_sticks
        new_game.points = self.game_state.points.copy()
        
        return new_game
    
    def _apply_action(self, game_state: 'FullRiichi', action: Action, player_id: int):
        """Apply an action to the game state"""
        player = game_state.players[player_id]
        
        if isinstance(action, Tsumo):
            game_state.winner = player_id
            game_state.game_over = True
        elif isinstance(action, Ron):
            game_state.winner = player_id
            game_state.game_over = True
            player.add_tile(game_state.last_discarded_tile)
        elif isinstance(action, Discard):
            player.remove_tile(action.tile)
            game_state.discarded_tiles.append(action.tile)
            game_state.last_discarded_tile = action.tile
            game_state.last_discard_player = player_id
        elif isinstance(action, Pon):
            player.make_call('pon', action.tiles, game_state.last_discarded_tile, game_state.last_discard_player)
            if game_state.last_discarded_tile in game_state.discarded_tiles:
                game_state.discarded_tiles.remove(game_state.last_discarded_tile)
            game_state.current_player_idx = player_id
            game_state.last_discarded_tile = None
            game_state.last_discard_player = None
        elif isinstance(action, Chi):
            player.make_call('chi', action.tiles, game_state.last_discarded_tile, game_state.last_discard_player)
            if game_state.last_discarded_tile in game_state.discarded_tiles:
                game_state.discarded_tiles.remove(game_state.last_discarded_tile)
            game_state.current_player_idx = player_id
            game_state.last_discarded_tile = None
            game_state.last_discard_player = None
    
    def simulate(self) -> float:
        """Simulate a random playout from this node"""
        current_state = self._copy_game_state()
        current_player_id = self.player_id
        
        # Play until game ends
        while not current_state.is_game_over() and current_state.tiles:
            current_player = current_state.players[current_player_id]
            game_state_for_player = current_state.get_game_state(current_player_id)
            
            # Get possible actions
            possible_actions = current_player.get_possible_actions(game_state_for_player)
            
            # Select random action
            action = self._select_random_action(current_player, possible_actions, current_state)
            
            # Apply action
            self._apply_action(current_state, action, current_player_id)
            
            # Move to next player
            current_player_id = (current_player_id - 1) % 4
        
        # Return reward based on game outcome
        return self._get_reward(current_state, self.player_id)
    
    def _select_random_action(self, player: 'Player', possible_actions: Dict[str, List], game_state: 'FullRiichi') -> Action:
        """Select a random action from possible actions"""
        # Priority: Tsumo > Ron > Pon > Chi > Discard
        
        if possible_actions.get('tsumo'):
            return Tsumo()
        
        if possible_actions.get('ron'):
            return Ron()
        
        # Occasionally declare Riichi if possible
        if possible_actions.get('riichi'):
            if random.random() < 0.2 and possible_actions['riichi']:
                # Choose a discard to riichi
                tstr = random.choice(possible_actions['riichi'])
                tile = Tile(Suit(tstr[-1]), TileType(int(tstr[:-1])))
                # mark riichi
                player.riichi_declared = True
                return Discard(tile)

        # 30% chance to make a call if possible
        if random.random() < 0.3:
            if possible_actions.get('pon'):
                tiles_str = random.choice(possible_actions['pon'])
                tiles = [Tile(Suit(t[-1]), TileType(int(t[:-1]))) for t in tiles_str]
                return Pon(tiles)
            
            if possible_actions.get('chi'):
                tiles_str = random.choice(possible_actions['chi'])
                tiles = [Tile(Suit(t[-1]), TileType(int(t[:-1]))) for t in tiles_str]
                return Chi(tiles)
        
        # Default: discard random tile
        return Discard(random.choice(player.hand))
    
    def _get_reward(self, game_state: 'FullRiichi', player_id: int) -> float:
        """Get reward for the player based on game outcome"""
        if not game_state.is_game_over():
            return 0.0  # Draw
        
        if game_state.winner == player_id:
            return 1.0  # Win
        else:
            return -1.0  # Loss
    
    def backpropagate(self, reward: float):
        """Backpropagate the reward up the tree"""
        node = self
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent
    
    def get_best_action(self) -> Action:
        """Get the best action based on visit counts"""
        if not self.children:
            return None
        
        best_child = max(self.children, key=lambda child: child.visits)
        return best_child.action


class PQNetwork:
    """Neural network that outputs both policy and value for Mahjong game states using TensorFlow/Keras"""
    
    def __init__(self, hidden_size: int = 128, embedding_dim: int = 4, max_turns: int = 50):
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for PQNetwork. Please install tensorflow.")
        
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.max_turns = max_turns
        
        # Build the model
        self.model = self._build_model()
    
    def _build_model(self):
        """Build the TensorFlow/Keras model with convolutional architecture"""
        
        # Input layers
        # Hand inputs: (batch_size, 14, 5) - 14 tiles, 5 features (4 embedding + 1 called flag)
        hand_inputs = keras.Input(shape=(14, 5), name='hand_inputs')
        
        # Discard pile inputs: (batch_size, max_turns, embedding_dim) for each player
        discard_inputs = []
        for i in range(4):
            discard_input = keras.Input(shape=(self.max_turns, self.embedding_dim), name=f'discard_input_{i}')
            discard_inputs.append(discard_input)
        
        # Additional game state features
        game_state_input = keras.Input(shape=(50,), name='game_state_input')
        
        # 1. Process hands with shared convolutional layers
        hand_conv1 = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(hand_inputs)
        hand_conv1 = layers.BatchNormalization()(hand_conv1)
        hand_conv2 = layers.Conv1D(128, kernel_size=3, padding='same', activation='relu')(hand_conv1)
        hand_conv2 = layers.BatchNormalization()(hand_conv2)
        hand_pool = layers.GlobalMaxPooling1D()(hand_conv2)
        
        # 2. Process discard piles with shared convolutional layers
        discard_features = []
        for i, discard_input in enumerate(discard_inputs):
            # Apply the same conv layers to each player's discard pile
            discard_conv1 = layers.Conv1D(32, kernel_size=3, padding='same', activation='relu')(discard_input)
            discard_conv1 = layers.BatchNormalization()(discard_conv1)
            discard_conv2 = layers.Conv1D(64, kernel_size=3, padding='same', activation='relu')(discard_conv1)
            discard_conv2 = layers.BatchNormalization()(discard_conv2)
            discard_pool = layers.GlobalMaxPooling1D()(discard_conv2)
            discard_features.append(discard_pool)
        
        # 3. Combine hand and discard features for each player
        player_features = []
        for i in range(4):
            if i == 0:  # Current player - use actual hand features
                player_hand = hand_pool
            else:  # Opponents - use placeholder (will be masked)
                player_hand = layers.Dense(128, activation='relu')(layers.Dense(128, activation='relu')(game_state_input))
            
            player_discard = discard_features[i]
            
            # Combine hand and discard features
            combined = layers.Concatenate()([player_hand, player_discard])
            player_combined = layers.Dense(256, activation='relu')(combined)
            player_combined = layers.Dropout(0.3)(player_combined)
            player_features.append(player_combined)
        
        # 4. Concatenate all player features
        all_players = layers.Concatenate()(player_features)
        
        # 5. Final hidden layers
        x = layers.Dense(self.hidden_size, activation='relu')(all_players)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(self.hidden_size // 2, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # 6. Add game state features
        x = layers.Concatenate()([x, game_state_input])
        x = layers.Dense(self.hidden_size // 2, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # 7. Output heads
        policy_head = layers.Dense(200, activation='softmax', name='policy')(x)
        value_head = layers.Dense(1, activation='tanh', name='value')(x)
        
        # Create model
        inputs = [hand_inputs] + discard_inputs + [game_state_input]
        model = keras.Model(inputs=inputs, outputs=[policy_head, value_head])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss={
                'policy': 'categorical_crossentropy',
                'value': 'mse'
            },
            metrics={
                'policy': 'accuracy',
                'value': 'mae'
            }
        )
        
        return model
    
    def evaluate(self, game_state: 'GameState') -> Tuple[np.ndarray, float]:
        """
        Evaluate a game state and return (policy, value)
        policy: probability distribution over all possible actions
        value: estimated state value (-1 to 1)
        """
        # Extract features
        features = self._extract_features(game_state)
        
        # Batch the features for prediction
        batched_features = []
        for feature in features:
            if len(feature.shape) == 2:
                # Add batch dimension
                batched_features.append(np.expand_dims(feature, axis=0))
            else:
                # Already 1D, add batch dimension
                batched_features.append(np.expand_dims(feature, axis=0))
        
        # Get predictions
        policy, value = self.model.predict(batched_features, verbose=0)
        
        return policy[0], float(value[0][0])
    
    def _extract_features(self, game_state: 'GameState') -> List[np.ndarray]:
        """Extract features from game state using the new convolutional architecture"""
        
        # 1. Hand features: (14, 5)
        hand_features = self._encode_hand_convolutional(game_state.player_hand, game_state.called_sets.get(game_state.player_id, []))
        
        # 2. Discard pile features: (max_turns, embedding_dim) for each player
        discard_features = []
        for player_id in range(4):
            player_discards = self._get_player_discards(game_state, player_id)
            discard_features.append(self._encode_discard_pile_convolutional(player_discards))
        
        # 3. Additional game state features
        game_state_features = self._extract_additional_features(game_state)
        
        return [hand_features] + discard_features + [game_state_features]
    
    def _encode_hand_convolutional(self, hand: List[Tile], called_sets: List['CalledSet']) -> np.ndarray:
        """Encode hand as (14, 5) tensor for convolutional processing"""
        hand_tensor = np.zeros((14, 5))
        
        # Create a set of called tiles for quick lookup
        called_tiles = set()
        for called_set in called_sets:
            for tile in called_set.tiles:
                called_tiles.add(tile)
        
        # Fill in the hand tensor
        for i, tile in enumerate(hand[:14]):  # Limit to 14 tiles
            # Get tile embedding (first 4 features)
            tile_embedding = self._get_tile_embedding(tile)
            hand_tensor[i, :4] = tile_embedding
            
            # Set called flag (5th feature)
            hand_tensor[i, 4] = 1.0 if tile in called_tiles else 0.0
        
        return hand_tensor
    
    def _encode_discard_pile_convolutional(self, discards: List[str]) -> np.ndarray:
        """Encode discard pile as (max_turns, embedding_dim) tensor"""
        # Initialize with zeros
        discard_tensor = np.zeros((self.max_turns, self.embedding_dim))
        
        # Fill in the discard tensor (most recent discards first)
        for i, tile_str in enumerate(discards[:self.max_turns]):
            try:
                # Parse tile string (e.g., "1p", "5s")
                tile_type = int(tile_str[:-1])
                suit = Suit(tile_str[-1])
                tile = Tile(suit, TileType(tile_type))
                
                # Get tile embedding
                tile_embedding = self._get_tile_embedding(tile)
                discard_tensor[i] = tile_embedding
            except:
                continue
        
        return discard_tensor
    
    def _get_player_discards(self, game_state: 'GameState', player_id: int) -> List[str]:
        """Get discards for a specific player"""
        if hasattr(game_state, 'player_discards') and player_id in game_state.player_discards:
            return game_state.player_discards[player_id]
        return []
    
    def _get_tile_embedding(self, tile: Tile) -> np.ndarray:
        """Get embedding for a single tile"""
        # For now, use a simple hash-based embedding
        # In a full implementation, you might want to learn these embeddings
        embedding = np.zeros(self.embedding_dim)
        tile_idx = self._get_tile_index(tile)
        
        # Simple hash-based embedding
        np.random.seed(tile_idx)  # For reproducible embeddings
        embedding = np.random.randn(self.embedding_dim) * 0.1
        np.random.seed()  # Reset seed
        
        return embedding
    
    def _get_tile_index(self, tile: Tile) -> int:
        """Get index for a tile in the embedding matrix"""
        # Create indices across suits: m(0..8)->0..8, p(0..8)->9..17, s(0..8)->18..26, honors(1..7)->27..33
        if tile.suit == Suit.MANZU:
            return (tile.tile_type.value - 1)
        if tile.suit == Suit.PINZU:
            return 9 + (tile.tile_type.value - 1)
        if tile.suit == Suit.SOUZU:
            return 18 + (tile.tile_type.value - 1)
        # HONOR
        honor_val = min(tile.tile_type.value, 7)
        return 27 + (honor_val - 1)
    
    def _extract_additional_features(self, game_state: 'GameState') -> np.ndarray:
        """Extract additional game state features"""
        features = []
        
        # Remaining tiles count (normalized to 136)
        features.append(game_state.remaining_tiles / 136.0)
        
        # Can call feature
        features.append(1.0 if game_state.can_call else 0.0)
        
        # Number of tiles in hand
        features.append(len(game_state.player_hand) / 14.0)  # Max 14 tiles
        
        # Number of called sets
        your_called_sets = game_state.called_sets.get(game_state.player_id, [])
        features.append(len(your_called_sets) / 4.0)  # Max 4 called sets
        
        # Number of opponents' called sets
        total_opponent_sets = sum(len(sets) for player_id, sets in game_state.called_sets.items() 
                                if player_id != game_state.player_id)
        features.append(total_opponent_sets / 12.0)  # Max 12 total opponent sets
        
        # Number of visible tiles
        features.append(len(game_state.visible_tiles) / 136.0)
        
        # Last discarded tile features
        if game_state.last_discarded_tile:
            last_tile_embedding = self._get_tile_embedding(game_state.last_discarded_tile)
            features.extend(last_tile_embedding)
        else:
            features.extend([0.0] * self.embedding_dim)
        
        # Last discard player (one-hot encoded)
        if game_state.last_discard_player is not None:
            player_features = [0.0] * 4
            player_features[game_state.last_discard_player] = 1.0
            features.extend(player_features)
        else:
            features.extend([0.0] * 4)
        
        # Pad to 50 features
        while len(features) < 50:
            features.append(0.0)
        
        return np.array(features[:50])
    
    def get_action_probabilities(self, game_state: 'GameState', possible_actions: Dict[str, List]) -> Dict[str, float]:
        """
        Get probability distribution over possible actions
        Returns dict mapping action strings to probabilities
        """
        policy, _ = self.evaluate(game_state)
        
        # Map policy to action probabilities
        action_probs = {}
        
        # Tsumo
        if possible_actions.get('tsumo'):
            action_probs['tsumo'] = policy[0] if len(policy) > 0 else 0.0
        
        # Ron
        if possible_actions.get('ron'):
            action_probs['ron'] = policy[1] if len(policy) > 1 else 0.0
        
        # Pon calls
        pon_idx = 2
        for tiles in possible_actions.get('pon', []):
            if pon_idx < len(policy):
                action_probs[f'pon_{"_".join(str(t) for t in tiles)}'] = policy[pon_idx]
            pon_idx += 1
        
        # Chi calls
        for tiles in possible_actions.get('chi', []):
            if pon_idx < len(policy):
                action_probs[f'chi_{"_".join(str(t) for t in tiles)}'] = policy[pon_idx]
            pon_idx += 1
        
        # Discards
        for tile in game_state.player_hand:
            if pon_idx < len(policy):
                action_probs[f'discard_{str(tile)}'] = policy[pon_idx]
            pon_idx += 1
        
        # Normalize probabilities
        total_prob = sum(action_probs.values())
        if total_prob > 0:
            for action in action_probs:
                action_probs[action] /= total_prob
        
        return action_probs
    
    def save_model(self, filepath: str):
        """Save the model to a file"""
        if not filepath.endswith('.keras'):
            filepath += '.keras'
        self.model.save(filepath)
    
    def load_model(self, filepath: str):
        """Load the model from a file"""
        if not filepath.endswith('.keras'):
            filepath += '.keras'
        self.model = keras.models.load_model(filepath)
    
    def train(self, training_data: List[Tuple['GameState', np.ndarray, float]], epochs: int = 10, batch_size: int = 32):
        """
        Train the model on provided data
        training_data: List of (game_state, target_policy, target_value) tuples
        """
        if not training_data:
            return
        
        # Prepare training data
        hand_inputs = []
        discard_inputs = [[] for _ in range(4)]
        game_state_inputs = []
        y_policy = []
        y_value = []
        
        for game_state, target_policy, target_value in training_data:
            features = self._extract_features(game_state)
            
            hand_inputs.append(features[0])
            for i in range(4):
                discard_inputs[i].append(features[i + 1])
            game_state_inputs.append(features[-1])
            
            y_policy.append(target_policy)
            y_value.append([target_value])
        
        # Convert to numpy arrays
        hand_inputs = np.array(hand_inputs)
        discard_inputs = [np.array(discards) for discards in discard_inputs]
        game_state_inputs = np.array(game_state_inputs)
        y_policy = np.array(y_policy)
        y_value = np.array(y_value)
        
        # Train the model
        self.model.fit(
            [hand_inputs] + discard_inputs + [game_state_inputs],
            {'policy': y_policy, 'value': y_value},
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )


class AIPlayer(Player):
    """AI player using MCTS with neural network value estimation"""
    
    def __init__(self, player_id: int, simulation_count: int = 1000, exploration_constant: float = 1.414):
        super().__init__(player_id)
        self.simulation_count = simulation_count
        self.exploration_constant = exploration_constant
        
        # Initialize PQNetwork for policy and value estimation (if available)
        self.pq_network: Optional[PQNetwork] = None
        if TENSORFLOW_AVAILABLE:
            try:
                self.pq_network = PQNetwork(hidden_size=64, embedding_dim=4, max_turns=20)
            except Exception:
                self.pq_network = None
        self.current_game = None
    
    def play(self, game_state: GameState) -> Action:
        """Use MCTS to select the best action"""
        if not self.hand:
            return Tsumo()
        
        # Check for immediate wins
        if self.can_win():
            return Tsumo()
        
        # Check for Ron
        if (game_state.last_discarded_tile and 
            game_state.last_discard_player != self.player_id):
            if self.can_ron(game_state.last_discarded_tile):
                return Ron()
        
        # Use MCTS to find best action
        best_action = self._mcts_search(game_state)
        
        if best_action is None:
            # Fallback to random action
            return Discard(random.choice(self.hand))
        
        return best_action
    
    def _mcts_search(self, game_state: GameState) -> Action:
        """Perform MCTS search to find the best action"""
        # Create a copy of the current game state for MCTS
        game_copy = self._create_game_copy(game_state)
        
        # Create root node
        root = MCTSNode(game_copy, self.player_id)
        
        # Run simulations
        for _ in range(self.simulation_count):
            # Selection
            node = root
            while node.is_fully_expanded() and not node.is_terminal():
                node = node.select_child(self.exploration_constant)
                if node is None:
                    break
            
            # Expansion
            if node and not node.is_terminal():
                node = node.expand()
            
            # Simulation
            if node:
                reward = node.simulate()
                # Blend with PQNetwork value estimate if available
                if not node.is_terminal() and self.pq_network is not None:
                    try:
                        _, value_estimate = self.pq_network.evaluate(
                            node.game_state.get_game_state(self.player_id)
                        )
                        reward = 0.5 * reward + 0.5 * float(value_estimate)
                    except Exception:
                        pass
                
                # Backpropagation
                node.backpropagate(reward)
        
        # Return best action
        return root.get_best_action()
    
    def _create_game_copy(self, game_state: GameState) -> 'FullRiichi':
        """Create a copy of the game state for MCTS"""
        # Create new players
        copied_players = []
        for i in range(4):
            if i == self.player_id:
                new_player = AIPlayer(i, self.simulation_count, self.exploration_constant)
            else:
                new_player = Player(i)
            
            # Copy hand and called sets from the current game state
            if i == self.player_id:
                new_player.hand = game_state.player_hand.copy()
                new_player.called_sets = game_state.called_sets.get(i, []).copy()
                new_player.riichi_declared = game_state.riichi_declared.get(i, False)
            else:
                # For other players, we don't have full information, so we'll use empty hands
                new_player.hand = []
                new_player.called_sets = game_state.called_sets.get(i, []).copy()
                new_player.riichi_declared = game_state.riichi_declared.get(i, False)
            
            copied_players.append(new_player)
        
        # Create new game instance
        new_game = FullRiichi(copied_players)
        
        # Copy game state
        new_game.tiles = []  # We don't have full information about remaining tiles
        new_game.discarded_tiles = game_state.visible_tiles.copy()
        new_game.current_player_idx = self.player_id
        new_game.game_over = False
        new_game.winner = None
        new_game.last_discarded_tile = game_state.last_discarded_tile
        new_game.last_discard_player = game_state.last_discard_player
        new_game.round_wind = game_state.round_wind
        new_game.dealer_idx = game_state.dealer_idx
        new_game.points = game_state.points.copy()
        
        return new_game
    
    def set_game_state(self, game_state: 'FullRiichi'):
        """Set the current game state for MCTS"""
        self.current_game = game_state