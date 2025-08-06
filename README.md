# SimpleJong - Simplified Mahjong AI Project

A simplified mahjong game implementation designed for AI/RL research. This version focuses on Pinzu tiles only with simplified rules to make it easier to implement and test AI strategies.

## Game Rules

- **Tiles**: Only Pinzu tiles (1-9), 6 copies of each type
- **Players**: 4 players
- **Starting hand**: 8 tiles per player
- **Objective**: Complete 3 sets of 3 tiles to win
- **Valid sets**:
  - Sets of 3 identical tiles (e.g., 333, 444, 555)
  - Sequences of 3 consecutive tiles (e.g., 123, 234, 345, 456, 567, 678, 789)
- **Gameplay**: Players take turns discarding one tile and drawing one tile
- **Winning**: First player to complete 3 sets wins
- **Draw**: If no tiles remain, the game ends in a draw

## Project Structure

```
mahjong-ai/
├── simple_jong.py      # Main game implementation
├── test_simple_jong.py # Test and demonstration script
└── README.md          # This file
```

## Classes

### `SimpleJong`
The main game class that manages the game state and flow.

**Constructor:**
```python
SimpleJong(players: List[Player])
```

**Key methods:**
- `play_round()`: Play one complete round, returns winner's player_id or None
- `get_game_state(player_id)`: Get game state for a specific player
- `get_winner()`: Get the winner's player_id
- `is_game_over()`: Check if the game is over
- `get_remaining_tiles()`: Get number of remaining tiles

### `Player`
Base player class that can be extended for AI implementations.

**Key methods:**
- `play(game_state)`: Player's turn - returns a tile to discard
- `can_win()`: Check if the player can win with current hand
- `add_tile(tile)`: Add a tile to player's hand
- `remove_tile(tile)`: Remove a tile from player's hand

### `GameState`
Data container for game state information available to a player.

**Fields:**
- `player_hand`: List of tiles in player's hand
- `visible_tiles`: List of tiles on the table
- `remaining_tiles`: Number of remaining tiles in deck
- `player_id`: Player's ID
- `other_players_discarded`: Dictionary of other players' discarded tiles

### `Tile`
Represents a single tile.

**Fields:**
- `tile_type`: TileType enum value (1-9)

## Usage Example

```python
from simple_jong import SimpleJong, Player

# Create 4 players
players = [Player(i) for i in range(4)]

# Create and play a game
game = SimpleJong(players)
winner = game.play_round()

if winner is not None:
    print(f"Player {winner} wins!")
else:
    print("Game ended in a draw")
```

## Running Tests

To run the test script:

```bash
python test_simple_jong.py
```

This will:
1. Test the winning hand detection logic with various scenarios
2. Play a complete game with 4 random players
3. Display the results and verify the winning hand

## Future Enhancements

This simplified version is designed as a foundation for:
- Reinforcement learning experiments
- AI strategy development
- Game theory research
- Multi-agent systems

Potential extensions:
- Chi and Pon mechanics
- More complex scoring systems
- Different tile types (Manzu, Souzu)
- Wind and dragon tiles
- More sophisticated AI players 