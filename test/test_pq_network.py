import unittest
import sys
import os
import numpy as np

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.game import (
    PQNetwork, GameState, Tile, TileType, Suit, CalledSet, TENSORFLOW_AVAILABLE
)


class TestPQNetwork(unittest.TestCase):
    """Test cases for PQNetwork implementation"""
    
    def setUp(self):
        """Set up test fixtures"""
        if not TENSORFLOW_AVAILABLE:
            self.skipTest("TensorFlow not available")
        self.pq_network = PQNetwork(hidden_size=64, embedding_dim=4, max_turns=20)
    
    def test_pq_network_creation(self):
        """Test that PQNetwork can be created"""
        if not TENSORFLOW_AVAILABLE:
            self.skipTest("TensorFlow not available")
        
        self.assertEqual(self.pq_network.hidden_size, 64)
        self.assertEqual(self.pq_network.embedding_dim, 4)
        self.assertEqual(self.pq_network.max_turns, 20)
        self.assertIsNotNone(self.pq_network.model)
    
    def test_tile_embedding(self):
        """Test tile embedding functionality"""
        if not TENSORFLOW_AVAILABLE:
            self.skipTest("TensorFlow not available")
        
        tile = Tile(Suit.PINZU, TileType.ONE)
        embedding = self.pq_network._get_tile_embedding(tile)
        self.assertEqual(embedding.shape, (4,))
        self.assertTrue(np.all(embedding != 0))  # Should not be all zeros
    
    def test_tile_index_calculation(self):
        """Test tile index calculation"""
        if not TENSORFLOW_AVAILABLE:
            self.skipTest("TensorFlow not available")
        
        # Test PINZU tiles
        for i, tile_type in enumerate(TileType):
            tile = Tile(Suit.PINZU, tile_type)
            expected_idx = i * 2
            self.assertEqual(self.pq_network._get_tile_index(tile), expected_idx)
        
        # Test SOUZU tiles
        for i, tile_type in enumerate(TileType):
            tile = Tile(Suit.SOUZU, tile_type)
            expected_idx = i * 2 + 1
            self.assertEqual(self.pq_network._get_tile_index(tile), expected_idx)
    
    def test_hand_encoding_convolutional(self):
        """Test hand encoding with convolutional approach"""
        if not TENSORFLOW_AVAILABLE:
            self.skipTest("TensorFlow not available")
        
        hand = [
            Tile(Suit.PINZU, TileType.ONE),
            Tile(Suit.PINZU, TileType.TWO),
            Tile(Suit.SOUZU, TileType.THREE)
        ]
        
        called_sets = [
            CalledSet(
                tiles=[Tile(Suit.PINZU, TileType.ONE), Tile(Suit.PINZU, TileType.ONE), Tile(Suit.PINZU, TileType.ONE)],
                call_type="pon",
                called_tile=Tile(Suit.PINZU, TileType.ONE),
                caller_position=0,
                source_position=1
            )
        ]
        
        hand_tensor = self.pq_network._encode_hand_convolutional(hand, called_sets)
        self.assertEqual(hand_tensor.shape, (12, 5))  # 12 tiles, 5 features
        
        # Check that the first tile has the called flag set
        self.assertEqual(hand_tensor[0, 4], 1.0)  # Called flag for first tile
        self.assertEqual(hand_tensor[1, 4], 0.0)  # Not called for second tile
    
    def test_discard_pile_encoding_convolutional(self):
        """Test discard pile encoding with convolutional approach"""
        if not TENSORFLOW_AVAILABLE:
            self.skipTest("TensorFlow not available")
        
        discards = ["1p", "2s", "3p", "4s"]
        discard_tensor = self.pq_network._encode_discard_pile_convolutional(discards)
        self.assertEqual(discard_tensor.shape, (20, 4))  # max_turns, embedding_dim
        
        # Check that the first few positions have embeddings
        self.assertTrue(np.any(discard_tensor[0] != 0))
        self.assertTrue(np.any(discard_tensor[1] != 0))
    
    def test_feature_extraction(self):
        """Test feature extraction from game state"""
        if not TENSORFLOW_AVAILABLE:
            self.skipTest("TensorFlow not available")
        
        game_state = GameState(
            player_hand=[Tile(Suit.PINZU, TileType.ONE), Tile(Suit.PINZU, TileType.TWO)],
            visible_tiles=[],
            remaining_tiles=50,
            player_id=0,
            other_players_discarded={},
            called_sets={},
            can_call=False
        )
        
        features = self.pq_network._extract_features(game_state)
        
        # Should return a list of arrays: [hand_features, discard_features[4], game_state_features]
        self.assertEqual(len(features), 6)  # 1 hand + 4 discards + 1 game state
        
        # Check shapes
        self.assertEqual(features[0].shape, (12, 5))  # Hand features
        for i in range(1, 5):
            self.assertEqual(features[i].shape, (20, 4))  # Discard features
        self.assertEqual(features[5].shape, (50,))  # Game state features
    
    def test_evaluate(self):
        """Test the evaluate method"""
        if not TENSORFLOW_AVAILABLE:
            self.skipTest("TensorFlow not available")
        
        game_state = GameState(
            player_hand=[Tile(Suit.PINZU, TileType.ONE), Tile(Suit.PINZU, TileType.TWO)],
            visible_tiles=[],
            remaining_tiles=50,
            player_id=0,
            other_players_discarded={},
            called_sets={},
            can_call=False
        )
        
        policy, value = self.pq_network.evaluate(game_state)
        
        # Check multi-head policy shapes and properties
        self.assertIn('action', policy)
        self.assertIn('tile1', policy)
        self.assertIn('tile2', policy)
        self.assertEqual(policy['action'].shape, (5,))
        self.assertEqual(policy['tile1'].shape, (18,))
        self.assertEqual(policy['tile2'].shape, (18,))
        self.assertTrue(np.isclose(np.sum(policy['action']), 1.0))
        self.assertTrue(np.isclose(np.sum(policy['tile1']), 1.0))
        self.assertTrue(np.isclose(np.sum(policy['tile2']), 1.0))
        self.assertTrue(np.all(policy['action'] >= 0))
        self.assertTrue(np.all(policy['tile1'] >= 0))
        self.assertTrue(np.all(policy['tile2'] >= 0))
        
        # Check value range
        self.assertTrue(-1.0 <= value <= 1.0)  # Value in [-1, 1] range
    
    def test_action_probabilities(self):
        """Test action probability calculation"""
        if not TENSORFLOW_AVAILABLE:
            self.skipTest("TensorFlow not available")
        
        game_state = GameState(
            player_hand=[Tile(Suit.PINZU, TileType.ONE), Tile(Suit.PINZU, TileType.TWO)],
            visible_tiles=[],
            remaining_tiles=50,
            player_id=0,
            other_players_discarded={},
            called_sets={},
            can_call=False
        )
        
        possible_actions = {
            'tsumo': [],
            'ron': [],
            'pon': [],
            'chi': []
        }
        
        action_probs = self.pq_network.get_action_probabilities(game_state, possible_actions)
        
        # Should have probabilities for discards
        self.assertTrue(len(action_probs) > 0)
        
        # Probabilities should sum to 1
        total_prob = sum(action_probs.values())
        self.assertAlmostEqual(total_prob, 1.0, places=5)
    
    def test_model_save_load(self):
        """Test model save and load functionality"""
        if not TENSORFLOW_AVAILABLE:
            self.skipTest("TensorFlow not available")
        
        # Create a temporary file path
        temp_model_path = "temp_model"
        
        try:
            # Save the model
            self.pq_network.save_model(temp_model_path)
            
            # Create a new network and load the model
            new_network = PQNetwork(hidden_size=64, embedding_dim=4, max_turns=20)
            new_network.load_model(temp_model_path)
            
            # Test that the loaded model works
            game_state = GameState(
                player_hand=[Tile(Suit.PINZU, TileType.ONE)],
                visible_tiles=[],
                remaining_tiles=50,
                player_id=0,
                other_players_discarded={},
                called_sets={},
                can_call=False
            )
            
            policy, value = new_network.evaluate(game_state)
            # Multi-head outputs preserved after load
            self.assertIn('action', policy)
            self.assertIn('tile1', policy)
            self.assertIn('tile2', policy)
            self.assertEqual(policy['action'].shape, (5,))
            self.assertEqual(policy['tile1'].shape, (18,))
            self.assertEqual(policy['tile2'].shape, (18,))
            self.assertTrue(-1.0 <= value <= 1.0)
            
        finally:
            # Clean up
            import shutil
            if os.path.exists(temp_model_path + '.keras'):
                os.remove(temp_model_path + '.keras')
    
    def test_get_player_discards(self):
        """Test getting player discards"""
        if not TENSORFLOW_AVAILABLE:
            self.skipTest("TensorFlow not available")
        
        game_state = GameState(
            player_hand=[],
            visible_tiles=[],
            remaining_tiles=50,
            player_id=0,
            other_players_discarded={},
            called_sets={},
            can_call=False
        )
        
        # Test with no player_discards attribute
        discards = self.pq_network._get_player_discards(game_state, 0)
        self.assertEqual(discards, [])
        
        # Test with player_discards attribute
        game_state.player_discards = {0: ["1p", "2s"], 1: ["3p"]}
        discards = self.pq_network._get_player_discards(game_state, 0)
        self.assertEqual(discards, ["1p", "2s"])
        
        discards = self.pq_network._get_player_discards(game_state, 1)
        self.assertEqual(discards, ["3p"])


if __name__ == '__main__':
    unittest.main()
