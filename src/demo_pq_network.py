#!/usr/bin/env python3
"""
Demo script for PQNetwork implementation
"""

import sys
import os
import numpy as np

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.game import (
    PQNetwork, GameState, Tile, TileType, Suit, CalledSet, TENSORFLOW_AVAILABLE
)


def demo_pq_network():
    """Demo the PQNetwork implementation"""
    print("=== PQNetwork Mahjong AI Demo ===\n")
    
    if not TENSORFLOW_AVAILABLE:
        print("TensorFlow not available. Please install tensorflow to use PQNetwork.")
        print("Run: pip install tensorflow")
        return
    
    # Create a PQNetwork
    print("Creating PQNetwork...")
    pq_network = PQNetwork(hidden_size=64, embedding_dim=4, max_turns=20)
    print(f"Network created with convolutional architecture")
    print(f"Hidden size: {pq_network.hidden_size}, Embedding dim: {pq_network.embedding_dim}, Max turns: {pq_network.max_turns}\n")
    
    # Create a sample game state
    print("Creating sample game state...")
    game_state = GameState(
        player_hand=[
            Tile(Suit.PINZU, TileType.ONE),
            Tile(Suit.PINZU, TileType.TWO),
            Tile(Suit.PINZU, TileType.THREE),
            Tile(Suit.SOUZU, TileType.FOUR),
            Tile(Suit.SOUZU, TileType.FIVE),
            Tile(Suit.SOUZU, TileType.SIX),
            Tile(Suit.PINZU, TileType.SEVEN),
            Tile(Suit.PINZU, TileType.EIGHT),
            Tile(Suit.PINZU, TileType.NINE),
            Tile(Suit.SOUZU, TileType.ONE),
            Tile(Suit.SOUZU, TileType.TWO),
        ],
        visible_tiles=[
            Tile(Suit.PINZU, TileType.ONE),
            Tile(Suit.SOUZU, TileType.THREE),
        ],
        remaining_tiles=45,
        player_id=0,
        other_players_discarded={
            1: [Tile(Suit.PINZU, TileType.FIVE)],
            2: [Tile(Suit.SOUZU, TileType.SEVEN)],
            3: [Tile(Suit.PINZU, TileType.NINE)]
        },
        called_sets={
            0: [
                CalledSet(
                    tiles=[Tile(Suit.PINZU, TileType.ONE), Tile(Suit.PINZU, TileType.ONE), Tile(Suit.PINZU, TileType.ONE)],
                    call_type="pon",
                    called_tile=Tile(Suit.PINZU, TileType.ONE),
                    caller_position=0,
                    source_position=1
                )
            ],
            1: [],
            2: [],
            3: []
        },
        can_call=False
    )
    
    # Add player_discards to game_state for the demo
    game_state.player_discards = {
        0: ["1p", "2s", "3p"],
        1: ["4s", "5p"],
        2: ["6s", "7p", "8s"],
        3: ["9p"]
    }
    
    print(f"Game state created with {len(game_state.player_hand)} tiles in hand")
    print(f"Visible tiles: {len(game_state.visible_tiles)}")
    print(f"Remaining tiles: {game_state.remaining_tiles}")
    print(f"Called sets: {len(game_state.called_sets[0])}")
    print(f"Player discards: {[len(discards) for discards in game_state.player_discards.values()]}\n")
    
    # Test feature extraction
    print("Testing feature extraction...")
    features = pq_network._extract_features(game_state)
    print(f"Number of feature arrays: {len(features)}")
    print(f"Hand features shape: {features[0].shape}")
    print(f"Discard features shapes: {[f.shape for f in features[1:5]]}")
    print(f"Game state features shape: {features[5].shape}\n")
    
    # Evaluate the game state
    print("Evaluating game state...")
    policy, value = pq_network.evaluate(game_state)
    
    print(f"State value: {value:.4f}")
    print(f"Policy shape: {policy.shape}")
    print(f"Policy sum: {np.sum(policy):.4f}")
    print(f"Max policy value: {np.max(policy):.4f}")
    print(f"Min policy value: {np.min(policy):.4f}\n")
    
    # Get action probabilities
    print("Getting action probabilities...")
    possible_actions = {
        'tsumo': [],
        'ron': [],
        'pon': [],
        'chi': []
    }
    
    action_probs = pq_network.get_action_probabilities(game_state, possible_actions)
    
    print("Action probabilities:")
    for action, prob in sorted(action_probs.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {action}: {prob:.4f}")
    
    print(f"\nTotal actions: {len(action_probs)}")
    print(f"Probability sum: {sum(action_probs.values()):.4f}\n")
    
    # Test training (with dummy data)
    print("Testing training with dummy data...")
    dummy_training_data = []
    
    for i in range(5):  # Reduced for demo
        # Create dummy target policy (one-hot encoded)
        target_policy = np.zeros(200)
        target_policy[i % 200] = 1.0
        
        # Create dummy target value
        target_value = np.random.uniform(-1, 1)
        
        dummy_training_data.append((game_state, target_policy, target_value))
    
    print(f"Created {len(dummy_training_data)} training samples")
    print("Training for 1 epoch...")
    
    try:
        pq_network.train(dummy_training_data, epochs=1, batch_size=2)
        print("Training completed successfully!")
    except Exception as e:
        print(f"Training failed: {e}")
    
    print("\nDemo completed!")


if __name__ == "__main__":
    demo_pq_network()
