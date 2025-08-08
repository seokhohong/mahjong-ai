import unittest
import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.game import (
    SimpleJong, AIPlayer, Player, Tile, TileType, Suit, 
    Tsumo, Ron, Discard, Pon, Chi, GameState
)


class TestMCTSAI(unittest.TestCase):
    """Test cases for MCTS AI implementation"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create 4 AI players
        self.players = [
            AIPlayer(0, simulation_count=1),
            AIPlayer(1, simulation_count=1),
            AIPlayer(2, simulation_count=1),
            AIPlayer(3, simulation_count=1)
        ]
        
        # Create game
        self.game = SimpleJong(self.players)
    
    def test_ai_player_creation(self):
        """Test that AIPlayer can be created"""
        ai_player = AIPlayer(0)
        self.assertEqual(ai_player.player_id, 0)
        self.assertEqual(ai_player.simulation_count, 1000)
        self.assertEqual(ai_player.exploration_constant, 1.414)
        # PQNetwork may be None if TensorFlow is not available
        self.assertTrue(hasattr(ai_player, 'pq_network'))
    
    def test_ai_player_immediate_win(self):
        """Test that AI player declares win when possible"""
        # Set up a winning hand for player 0 (12 tiles = 4 sets of 3)
        ai_player = self.players[0]
        ai_player.hand = [
            # Set 1: Triplet of 1p
            Tile(Suit.PINZU, TileType.ONE),
            Tile(Suit.PINZU, TileType.ONE),
            Tile(Suit.PINZU, TileType.ONE),
            # Set 2: Run 2s-3s-4s
            Tile(Suit.SOUZU, TileType.TWO),
            Tile(Suit.SOUZU, TileType.THREE),
            Tile(Suit.SOUZU, TileType.FOUR),
            # Set 3: Triplet of 5p
            Tile(Suit.PINZU, TileType.FIVE),
            Tile(Suit.PINZU, TileType.FIVE),
            Tile(Suit.PINZU, TileType.FIVE),
            # Set 4: Run 6s-7s-8s
            Tile(Suit.SOUZU, TileType.SIX),
            Tile(Suit.SOUZU, TileType.SEVEN),
            Tile(Suit.SOUZU, TileType.EIGHT),
        ]
        
        game_state = self.game.get_game_state(0)
        action = ai_player.play(game_state)
        
        self.assertIsInstance(action, Tsumo)
    
    def test_ai_player_ron(self):
        """Test that AI player declares Ron when possible"""
        ai_player = self.players[0]
        
        # Set up a hand that can win with a specific tile (11 tiles)
        ai_player.hand = [
            # Set 1: Triplet of 1p
            Tile(Suit.PINZU, TileType.ONE),
            Tile(Suit.PINZU, TileType.ONE),
            Tile(Suit.PINZU, TileType.ONE),
            # Set 2: Run 2s-3s-4s
            Tile(Suit.SOUZU, TileType.TWO),
            Tile(Suit.SOUZU, TileType.THREE),
            Tile(Suit.SOUZU, TileType.FOUR),
            # Set 3: Triplet of 5p
            Tile(Suit.PINZU, TileType.FIVE),
            Tile(Suit.PINZU, TileType.FIVE),
            Tile(Suit.PINZU, TileType.FIVE),
            # Need one more tile to complete the 4th set
            Tile(Suit.SOUZU, TileType.SIX),
            Tile(Suit.SOUZU, TileType.SEVEN),
        ]
        
        # Set up game state with a discarded tile that can complete the hand
        game_state = GameState(
            player_hand=ai_player.hand.copy(),
            visible_tiles=[Tile(Suit.SOUZU, TileType.EIGHT)],
            remaining_tiles=50,
            player_id=0,
            other_players_discarded={},
            called_sets={},
            last_discarded_tile=Tile(Suit.SOUZU, TileType.EIGHT),
            last_discard_player=1,
            can_call=True
        )
        
        action = ai_player.play(game_state)
        self.assertIsInstance(action, Ron)
    
    def test_ai_player_discard(self):
        """Test that AI player discards when no win is possible"""
        ai_player = self.players[0]
        
        # Set up a hand that can't win (11 tiles, not forming 4 sets)
        ai_player.hand = [
            Tile(Suit.PINZU, TileType.ONE),
            Tile(Suit.PINZU, TileType.TWO),
            Tile(Suit.SOUZU, TileType.THREE),
            Tile(Suit.SOUZU, TileType.FOUR),
            Tile(Suit.PINZU, TileType.FIVE),
            Tile(Suit.PINZU, TileType.SIX),
            Tile(Suit.SOUZU, TileType.SEVEN),
            Tile(Suit.SOUZU, TileType.EIGHT),
            Tile(Suit.PINZU, TileType.NINE),
            Tile(Suit.SOUZU, TileType.ONE),
            Tile(Suit.PINZU, TileType.TWO),
        ]
        
        game_state = self.game.get_game_state(0)
        action = ai_player.play(game_state)
        
        self.assertIsInstance(action, Discard)
        # Check that the discarded tile was actually in the hand
        self.assertIn(action.tile, ai_player.hand)
    
    def test_mcts_node_creation(self):
        """Test that MCTS nodes can be created"""
        from core.game import MCTSNode
        
        node = MCTSNode(self.game, 0)
        self.assertEqual(node.player_id, 0)
        self.assertEqual(node.visits, 0)
        self.assertEqual(node.value, 0.0)
        self.assertIsNotNone(node.untried_actions)
    
    def test_pq_network_optional(self):
        """AIPlayer should expose pq_network attribute, which may be None without TF"""
        ai_player = AIPlayer(0)
        # No strict assertions on value; just ensure attribute exists
        _ = ai_player.pq_network


if __name__ == '__main__':
    unittest.main()
