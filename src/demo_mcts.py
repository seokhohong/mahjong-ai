#!/usr/bin/env python3
"""
Demo script for MCTS AI implementation
"""

import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.game import (
    SimpleJong, AIPlayer, Player, Tile, TileType, Suit,
    Tsumo, Ron, Discard, Pon, Chi
)


def demo_mcts_ai():
    """Demo the MCTS AI implementation"""
    print("=== MCTS AI Mahjong Demo ===\n")
    
    # Create 4 AI players with different simulation counts
    players = [
        AIPlayer(0, simulation_count=500),  # Human-like player
        AIPlayer(1, simulation_count=100),  # Faster player
        AIPlayer(2, simulation_count=100),  # Faster player
        AIPlayer(3, simulation_count=100),  # Faster player
    ]
    
    # Create game
    game = SimpleJong(players)
    
    print(f"Game started with {len(game.tiles)} tiles remaining")
    print(f"Players start with {len(players[0].hand)} tiles each\n")
    
    # Play the game
    round_count = 0
    while not game.is_game_over() and game.tiles and round_count < 50:
        round_count += 1
        current_player = game.players[game.current_player_idx]
        
        print(f"--- Round {round_count} ---")
        print(f"Player {game.current_player_idx}'s turn")
        print(f"Hand size: {len(current_player.hand)} tiles")
        print(f"Remaining tiles: {len(game.tiles)}")
        
        # Draw a tile
        if game.tiles:
            new_tile = game.tiles.pop()
            current_player.add_tile(new_tile)
            print(f"Drew tile: {new_tile}")
        
        # Get game state and let player decide
        game_state = game.get_turn_snapshot(game.current_player_idx)
        action = current_player.play(),, game_state
        
        # Handle the action
        if isinstance(action, Tsumo):
            print(f"🎉 Player {game.current_player_idx} declares TSUMO!")
            game.winner = game.current_player_idx
            game.game_over = True
            break
        elif isinstance(action, Ron):
            print(f"🎉 Player {game.current_player_idx} declares RON!")
            game.winner = game.current_player_idx
            game.game_over = True
            break
        elif isinstance(action, Discard):
            current_player.remove_tile(action.tile)
            try:
                game.player_discards[current_player_id].append(action.tile)
            except Exception:
                pass
            print(f"Discarded: {action.tile}")
            
            # Check for Ron
            for i, player in enumerate(game.players):
                if i != game.current_player_idx:
                    if player.can_ron(action.tile):
                        print(f"🎉 Player {i} declares RON on {action.tile}!")
                        game.winner = i
                        game.game_over = True
                        player.add_tile(action.tile)
                        break
            
            if game.game_over:
                break
            
            # Check for calls
            call_result = game.check_for_calls(action.tile, game.current_player_idx)
            if call_result:
                caller_id, call_type, tiles = call_result
                print(f"Player {caller_id} calls {call_type.upper()} with tiles: {[str(t) for t in tiles]}")
                game.make_call(caller_id, call_type, tiles)
                continue
        
        # Move to next player
        game.current_player_idx = (game.current_player_idx - 1) % 4
        print()
    
    # Game over
    if game.is_game_over():
        if game.winner is not None:
            print(f"\n🏆 Player {game.winner} wins!")
        else:
            print("\n🤝 Game ended in a draw")
    else:
        print(f"\n⏰ Game stopped after {round_count} rounds")
    
    print(f"\nFinal stats:")
    print(f"- Rounds played: {round_count}")
    print(f"- Tiles remaining: {len(game.tiles)}")
    try:
        total_discards = sum(len(v) for v in game.player_discards.values())
    except Exception:
        total_discards = 0
    print(f"- Discarded tiles: {total_discards}")
    
    # Show final hands
    for i, player in enumerate(game.players):
        print(f"- Player {i} hand: {len(player.hand)} tiles")
        if player.called_sets:
            print(f"  Called sets: {len(player.called_sets)}")


def demo_mcts_vs_random():
    """Demo MCTS AI vs random players"""
    print("=== MCTS AI vs Random Players Demo ===\n")
    
    # Create 1 MCTS AI player and 3 random players
    players = [
        AIPlayer(0, simulation_count=200),  # MCTS AI
        Player(1),  # Random player
        Player(2),  # Random player
        Player(3),  # Random player
    ]
    
    # Create game
    game = SimpleJong(players)
    
    print("MCTS AI (Player 0) vs Random Players (Players 1-3)")
    print(f"Game started with {len(game.tiles)} tiles remaining\n")
    
    # Play the game
    round_count = 0
    while not game.is_game_over() and game.tiles and round_count < 30:
        round_count += 1
        current_player = game.players[game.current_player_idx]
        
        print(f"--- Round {round_count} ---")
        player_type = "MCTS AI" if isinstance(current_player, AIPlayer) else "Random"
        print(f"Player {game.current_player_idx} ({player_type})'s turn")
        
        # Draw a tile
        if game.tiles:
            new_tile = game.tiles.pop()
            current_player.add_tile(new_tile)
        
        # Get game state and let player decide
        game_state = game.get_turn_snapshot(game.current_player_idx)
        action = current_player.play(),, game_state
        
        # Handle the action
        if isinstance(action, Tsumo):
            print(f"🎉 Player {game.current_player_idx} ({player_type}) declares TSUMO!")
            game.winner = game.current_player_idx
            game.game_over = True
            break
        elif isinstance(action, Ron):
            print(f"🎉 Player {game.current_player_idx} ({player_type}) declares RON!")
            game.winner = game.current_player_idx
            game.game_over = True
            break
        elif isinstance(action, Discard):
            current_player.remove_tile(action.tile)
            try:
                game.player_discards[current_player_idx].append(action.tile)
            except Exception:
                pass
            print(f"Discarded: {action.tile}")
            
            # Check for Ron
            for i, player in enumerate(game.players):
                if i != game.current_player_idx:
                    if player.can_ron(action.tile):
                        player_type_ron = "MCTS AI" if isinstance(player, AIPlayer) else "Random"
                        print(f"🎉 Player {i} ({player_type_ron}) declares RON on {action.tile}!")
                        game.winner = i
                        game.game_over = True
                        player.add_tile(action.tile)
                        break
            
            if game.game_over:
                break
        
        # Move to next player
        game.current_player_idx = (game.current_player_idx - 1) % 4
    
    # Game over
    if game.is_game_over():
        if game.winner is not None:
            winner_type = "MCTS AI" if isinstance(game.players[game.winner], AIPlayer) else "Random"
            print(f"\n🏆 Player {game.winner} ({winner_type}) wins!")
        else:
            print("\n🤝 Game ended in a draw")
    else:
        print(f"\n⏰ Game stopped after {round_count} rounds")
    
    print(f"\nFinal stats:")
    print(f"- Rounds played: {round_count}")
    print(f"- Tiles remaining: {len(game.tiles)}")


if __name__ == "__main__":
    print("Mahjong MCTS AI Demo")
    print("=" * 30)
    
    # Run demos
    demo_mcts_ai()
    print("\n" + "=" * 50 + "\n")
    demo_mcts_vs_random()
