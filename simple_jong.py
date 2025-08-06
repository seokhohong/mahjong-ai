import random
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum


class TileType(Enum):
    """Enum for the 9 types of Pinzu tiles (1-9)"""
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
    tile_type: TileType
    
    def __str__(self):
        return f"{self.tile_type.value}p"
    
    def __eq__(self, other):
        if isinstance(other, Tile):
            return self.tile_type == other.tile_type
        return False
    
    def __hash__(self):
        return hash(self.tile_type)


@dataclass
class GameState:
    """Container for game state information available to a player"""
    player_hand: List[Tile]
    visible_tiles: List[Tile]  # Tiles on the table
    remaining_tiles: int
    player_id: int
    other_players_discarded: Dict[int, List[Tile]]  # Other players' discarded tiles


class Player:
    """Base player class"""
    
    def __init__(self, player_id: int):
        self.player_id = player_id
        self.hand: List[Tile] = []
    
    def play(self, game_state: GameState) -> Optional[Tile]:
        """
        Player's turn - returns a tile to discard from hand
        For now, just returns a random tile from hand
        """
        if not self.hand:
            return None
        return random.choice(self.hand)
    
    def add_tile(self, tile: Tile):
        """Add a tile to player's hand"""
        self.hand.append(tile)
    
    def remove_tile(self, tile: Tile):
        """Remove a tile from player's hand"""
        if tile in self.hand:
            self.hand.remove(tile)
    
    def can_win(self) -> bool:
        """
        Check if the player can win with current hand
        Need 3 sets of 3 tiles (triplets like 333, or runs like 234)
        """
        if len(self.hand) != 9:  # Need exactly 9 tiles to win
            return False
        
        # Convert hand to a count dictionary for easier manipulation
        tile_counts = {}
        for tile in self.hand:
            tile_counts[tile.tile_type.value] = tile_counts.get(tile.tile_type.value, 0) + 1
        
        # Try all possible ways to form 3 sets
        return self._can_form_sets(tile_counts, 3)

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
            if tile_counts[tile_value] > 0:
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


class SimpleJong:
    """Simplified Mahjong game with Pinzu tiles only"""
    
    def __init__(self, players: List[Player]):
        if len(players) != 4:
            raise ValueError("SimpleJong requires exactly 4 players")
        
        self.players = players
        self.tiles: List[Tile] = []
        self.discarded_tiles: List[Tile] = []
        self.current_player_idx = 0
        self.game_over = False
        self.winner = None
        
        # Initialize tiles: 6 copies of each tile type (1-9)
        for tile_type in TileType:
            for _ in range(6):
                self.tiles.append(Tile(tile_type))
        
        # Shuffle tiles
        random.shuffle(self.tiles)
        
        # Deal 8 tiles to each player
        for player in self.players:
            for _ in range(8):
                if self.tiles:
                    tile = self.tiles.pop()
                    player.add_tile(tile)
    
    def get_game_state(self, player_id: int) -> GameState:
        """Get game state for a specific player"""
        player = self.players[player_id]
        
        # Get other players' discarded tiles
        other_players_discarded = {}
        for i, p in enumerate(self.players):
            if i != player_id:
                other_players_discarded[i] = [] # Simplified for now
        
        return GameState(
            player_hand=player.hand.copy(),
            visible_tiles=self.discarded_tiles.copy(),
            remaining_tiles=len(self.tiles),
            player_id=player_id,
            other_players_discarded=other_players_discarded
        )
    
    def play_round(self) -> Optional[int]:
        """
        Play one round of the game
        Returns the winner's player_id, or None if no winner
        """
        while not self.game_over and self.tiles:
            current_player = self.players[self.current_player_idx]
            
            # 1. Draw a tile
            new_tile = self.tiles.pop()
            current_player.add_tile(new_tile)
            
            # 2. Check for win
            if current_player.can_win():
                self.winner = self.current_player_idx
                self.game_over = True
                return self.winner
            
            # 3. Discard a tile
            game_state = self.get_game_state(self.current_player_idx)
            discarded_tile = current_player.play(game_state)
            
            if discarded_tile:
                current_player.remove_tile(discarded_tile)
                self.discarded_tiles.append(discarded_tile)
                
                # Check if another player can win with the discarded tile
                for i, player in enumerate(self.players):
                    if i != self.current_player_idx:
                        player.add_tile(discarded_tile)
                        if player.can_win():
                            self.winner = i
                            self.game_over = True
                            # The tile is "claimed" so it's not on the table
                            # but in the winner's hand.
                            return self.winner 
                        player.remove_tile(discarded_tile) # backtrack
            
            # Move to next player
            self.current_player_idx = (self.current_player_idx + 1) % 4
        
        self.game_over = True
        return None
    
    def get_winner(self) -> Optional[int]:
        """Get the winner's player_id, or None if no winner"""
        return self.winner
    
    def is_game_over(self) -> bool:
        """Check if the game is over"""
        return self.game_over
    
    def get_remaining_tiles(self) -> int:
        """Get number of remaining tiles"""
        return len(self.tiles)