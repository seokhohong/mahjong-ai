import unittest
import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.game import (
    SimpleJong, Player, MCTSNode, SimpleHeuristicsPlayer,
    Tile, TileType, Suit, Discard, Ron, Pon, Chi, Tsumo,
)


class TestMCTSAI(unittest.TestCase):
    """Tests for MCTSNode and heuristic player behavior"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.players = [Player(0), Player(1), Player(2), Player(3)]
        self.game = SimpleJong(self.players)
    
    def test_mcts_node_creation(self):
        node = MCTSNode(self.game, 0)
        self.assertEqual(node.player_id, 0)
        self.assertEqual(node.visits, 0)
        self.assertEqual(node.value, 0.0)
        self.assertIsInstance(node.untried_actions, list)
    
    def test_expand_prefers_ron(self):
        # Player 0 discarded 3p; player 1 can ron
        g = SimpleJong([Player(0), Player(1), Player(2), Player(3)])
        g._player_hands[0] = [Tile(Suit.PINZU, TileType.THREE)] + [Tile(Suit.SOUZU, TileType.ONE)] * 10
        base_s = [
            Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.TWO), Tile(Suit.SOUZU, TileType.THREE),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE), Tile(Suit.SOUZU, TileType.SIX),
            Tile(Suit.SOUZU, TileType.SEVEN), Tile(Suit.SOUZU, TileType.EIGHT), Tile(Suit.SOUZU, TileType.NINE),
        ]
        g._player_hands[1] = base_s + [Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR)]
        g.tiles = []
        g.current_player_idx = 0
        g.step(0, Discard(Tile(Suit.PINZU, TileType.THREE)))
        node = MCTSNode(g, player_id=1)
        child = node.expand()
        self.assertIsInstance(child.action, Ron)
    
    def test_expand_prefers_pon_then_chi(self):
        # Prepare last discard 5s; actor 2 can pon
        g = SimpleJong([Player(0), Player(1), Player(2), Player(3)])
        g._player_hands[0] = [Tile(Suit.SOUZU, TileType.FIVE)] + [Tile(Suit.SOUZU, TileType.ONE)] * 10
        # Hand for player 2 avoids forming four melds even after adding 5s (prevents Ron)
        g._player_hands[2] = [
            Tile(Suit.SOUZU, TileType.FIVE), Tile(Suit.SOUZU, TileType.FIVE),
            Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.THREE),
            Tile(Suit.SOUZU, TileType.SIX), Tile(Suit.SOUZU, TileType.EIGHT), Tile(Suit.SOUZU, TileType.NINE),
            Tile(Suit.PINZU, TileType.ONE), Tile(Suit.PINZU, TileType.FOUR), Tile(Suit.PINZU, TileType.SEVEN), Tile(Suit.PINZU, TileType.NINE)
        ]
        g.tiles = []
        g.current_player_idx = 0
        g.step(0, Discard(Tile(Suit.SOUZU, TileType.FIVE)))
        node = MCTSNode(g, player_id=2)
        child = node.expand()
        self.assertIsInstance(child.action, Pon)
        # Chi case: last 3p, left player has 2p and 4p
        g2 = SimpleJong([Player(0), Player(1), Player(2), Player(3)])
        g2._player_hands[0] = [Tile(Suit.PINZU, TileType.THREE)] + [Tile(Suit.SOUZU, TileType.ONE)] * 10
        # Choose 9 souzu tiles that cannot be partitioned into three melds to avoid accidental Ron
        non_partitionable_souzu = [
            Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.TWO),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE),
            Tile(Suit.SOUZU, TileType.SEVEN), Tile(Suit.SOUZU, TileType.SEVEN), Tile(Suit.SOUZU, TileType.EIGHT), Tile(Suit.SOUZU, TileType.NINE)
        ]
        g2._player_hands[1] = [Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR)] + non_partitionable_souzu
        g2.tiles = []
        g2.current_player_idx = 0
        g2.step(0, Discard(Tile(Suit.PINZU, TileType.THREE)))
        node2 = MCTSNode(g2, player_id=1)
        child2 = node2.expand()
        # If ron possible it would have triggered; ensure chi selected
        self.assertIsInstance(child2.action, Chi)
    
    def test_discard_heuristic_outermost_on_tie(self):
        # Actor 0 in action phase: hand 1p,2p,5p,7p,7p plus filler to 11
        g = SimpleJong([Player(0), Player(1), Player(2), Player(3)])
        special = [Tile(Suit.PINZU, TileType.ONE), Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FIVE), Tile(Suit.PINZU, TileType.SEVEN), Tile(Suit.PINZU, TileType.SEVEN)]
        filler = [Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE), Tile(Suit.SOUZU, TileType.SIX), Tile(Suit.SOUZU, TileType.SEVEN), Tile(Suit.SOUZU, TileType.EIGHT), Tile(Suit.SOUZU, TileType.NINE)]
        g._player_hands[0] = special + filler
        g.current_player_idx = 0
        g.tiles = []
        node = MCTSNode(g, player_id=0)
        # Ask heuristic directly
        legal = g.legal_moves(0)
        hp = SimpleHeuristicsPlayer(0)
        choice = hp.select_action(g, 0, legal)
        self.assertIsInstance(choice, Discard)
        # Expect 1p discarded over 2p due to outermost tiebreak
        self.assertEqual(str(choice.tile), '1p')
    
    # Exception handling tests retained/adapted below

    def test_mctsnode_expand_no_legal_moves_raises(self):
        # Create a game with default current_player_idx = 0 and actor set to 1 (no legal moves for actor 1)
        players = [Player(0), Player(1), Player(2), Player(3)]
        game = SimpleJong(players)
        game.tiles = [Tile(Suit.PINZU, TileType.ONE)]  # ensure simulate loops if needed
        game.current_player_idx = 0
        node = MCTSNode(game, player_id=1)
        with self.assertRaises(MCTSNode.NoLegalMoves):
            node.expand()

    def test_mctsnode_expand_no_action_from_player_raises(self):
        # Player that never returns an action
        class NullChoicePlayer(SimpleHeuristicsPlayer):
            def select_action(self, game, actor_id, legal_moves):
                return None
        players = [Player(0), Player(1), Player(2), Player(3)]
        game = SimpleJong(players)
        game.tiles = [Tile(Suit.PINZU, TileType.ONE)]
        game.current_player_idx = 0
        # Ensure actor is current so legal moves exist
        node = MCTSNode(game, player_id=0, player=NullChoicePlayer(0))
        with self.assertRaises(MCTSNode.NoActionFromPlayer):
            node.expand()

    def test_mctsnode_simulate_no_legal_moves_raises(self):
        players = [Player(0), Player(1), Player(2), Player(3)]
        game = SimpleJong(players)
        # Ensure there are tiles so simulate enters loop, but pick actor with no legal moves
        game.tiles = [Tile(Suit.PINZU, TileType.ONE), Tile(Suit.SOUZU, TileType.TWO)]
        game.current_player_idx = 0
        node = MCTSNode(game, player_id=1)
        with self.assertRaises(MCTSNode.NoLegalMoves):
            node.simulate()

    def test_mctsnode_simulate_no_action_from_player_raises(self):
        class NullChoicePlayer(SimpleHeuristicsPlayer):
            def select_action(self, game, actor_id, legal_moves):
                return None
        players = [Player(0), Player(1), Player(2), Player(3)]
        game = SimpleJong(players)
        game.tiles = [Tile(Suit.PINZU, TileType.ONE)]
        game.current_player_idx = 0
        node = MCTSNode(game, player_id=0, player=NullChoicePlayer(0))
        with self.assertRaises(MCTSNode.NoActionFromPlayer):
            node.simulate()


if __name__ == '__main__':
    unittest.main()
