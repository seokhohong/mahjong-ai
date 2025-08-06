#!/usr/bin/env python3
"""
Demonstration script showing the Ron functionality in SimpleJong
"""

import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.game import SimpleJong, Player, Tile, TileType, Ron, Tsumo, Discard

def demo_ron():
    """Demonstrate Ron functionality"""
    print("=== SimpleJong Ron Demonstration ===\n")
    
    # Create players
    players = [Player(i) for i in range(4)]
    
    # Create game
    game = SimpleJong(players)
    
    # Set up a specific scenario where player 1 can declare Ron
    # Player 1: 333, 444, 55 (needs 5 to complete 555)
    players[1].hand = [
        Tile(TileType.THREE), Tile(TileType.THREE), Tile(TileType.THREE),
        Tile(TileType.FOUR), Tile(TileType.FOUR), Tile(TileType.FOUR),
        Tile(TileType.FIVE), Tile(TileType.FIVE)
    ]
    
    print(f"Player 1's hand: {[str(tile) for tile in players[1].hand]}")
    print("Player 1 needs a 5 tile to complete the hand: 333, 444, 555\n")
    
    # Simulate player 0 discarding a 5 tile
    five_tile = Tile(TileType.FIVE)
    print(f"Player 0 discards: {five_tile}")
    
    # Check if player 1 can declare Ron
    if players[1].can_ron(five_tile):
        print("✓ Player 1 can declare Ron!")
        
        # Simulate the Ron declaration
        game.winner = 1
        game.game_over = True
        players[1].add_tile(five_tile)  # Add the claimed tile to winner's hand
        
        print(f"Player 1 declares Ron and wins!")
        print(f"Final hand: {[str(tile) for tile in players[1].hand]}")
        print("Winning hand breakdown:")
        print("  - 333 (triplet)")
        print("  - 444 (triplet)") 
        print("  - 555 (triplet)")
    else:
        print("✗ Player 1 cannot declare Ron with this tile")
    
    print(f"\nGame winner: Player {game.get_winner()}")
    print(f"Game over: {game.is_game_over()}")

if __name__ == "__main__":
    demo_ron() 