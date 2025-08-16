#!/usr/bin/env python3
import unittest
import sys
import os
import numpy as np

# Add project root (so 'src.*' absolute imports work) and also 'src' for legacy direct imports
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
SRC_DIR = os.path.join(ROOT_DIR, 'src')
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from core.game import SimpleJong, HeuristicPlayer


class TestACNetworkAndPlayer(unittest.TestCase):
    def test_ac_network_shapes_and_inference(self):
        from core.learn_ac.ac_network import ACNetwork  # type: ignore

        net = ACNetwork(hidden_size=32, embedding_dim=4, max_turns=20)

        # Build minimal game and perspective for player 0
        players = []
        # Use a no-op player for other seats
        for i in range(4):
            players.append(HeuristicPlayer(i))
        game = SimpleJong(players, tile_copies=4)
        gs = game.get_game_perspective(game.current_player_idx)

        policy, value = net.evaluate(gs)
        self.assertTrue(np.all(np.isfinite(policy)))
        self.assertTrue(np.isfinite(value))

    def test_ac_player_can_play_one_game(self):
        from core.learn_ac.ac_network import ACNetwork  # type: ignore
        from core.learn_ac.ac_player import ACPlayer  # type: ignore

        net = ACNetwork(hidden_size=32, embedding_dim=4, max_turns=20, temperature=0.8)
        players = [ACPlayer(i, net, temperature=0.8) for i in range(4)]
        game = SimpleJong(players, tile_copies=4)
        game.play_round()
        self.assertTrue(game.is_game_over())
        # winners are valid ids
        for w in game.get_winners():
            self.assertIn(w, [0, 1, 2, 3])

    def test_ac_player_reaction_hierarchy(self):
        from core.learn_ac.ac_network import ACNetwork  # type: ignore
        from core.learn_ac.ac_player import ACPlayer  # type: ignore

        class _DummyNet:
            # Force high probabilities for Ron over others in reaction
            def evaluate(self, gs):
                main = np.array([0.1, 0.1, 0.9, 0.0, 0.0, 0.0], dtype=np.float32)  # prefer Ron
                tile = np.full((18,), 1.0/18, dtype=np.float32)
                chi = np.array([0.2, 0.2, 0.6], dtype=np.float32)
                return { 'main': main, 'tile': tile, 'chi_range': chi }, 0.0

        # Build a game where player 1 can ron
        from core.game import Tile, Suit, TileType, Discard, Ron
        players = [None, None, None, None]
        players = [ACPlayer(i, _DummyNet(), temperature=0.0) for i in range(4)]
        game = SimpleJong(players, tile_copies=4)
        # Setup a ron situation: give p1 11 tiles that can win on 1p discard from p0
        # We leverage the engine to reach a reaction state by simulating one discard
        # Step until p0 discards a tile; then ask p1 for reaction
        actor = game.current_player_idx
        gs0 = game.get_game_perspective(actor)
        # Force a discard move to create a reaction window
        move = Discard(gs0.player_hand[0])
        game.step(actor, move)
        # Now player to the right is reacting; ensure ACPlayer chooses Ron if available
        reactor = game.current_player_idx
        gs_react = game.get_game_perspective(reactor)
        opts = gs_react.get_call_options()
        # If ron is available, AC should pick it
        if gs_react.can_ron():
            mv = players[reactor].choose_reaction(gs_react, opts)
            self.assertIsInstance(mv, Ron)

    def test_legality_mask_matches_legal_moves_initial_state(self):
        # Initialize a fresh game and compare unique flat-action count to legality mask ones
        from core.learn_ac.policy_utils import legal_flat_mask, flat_index_for_action  # type: ignore
        from core.learn.pure_policy_dataset import serialize_state, serialize_action  # type: ignore
        players = [HeuristicPlayer(i) for i in range(4)]
        game = SimpleJong(players, tile_copies=4)
        gs = game.get_game_perspective(game.current_player_idx)
        legal = gs.legal_moves()
        mask = legal_flat_mask(gs)
        # Map legal moves to flat indices and compare unique count (since multiple identical discards map to one index)
        sd = serialize_state(gs)
        flat_indices = set()
        for mv in legal:
            ad = serialize_action(mv)
            idx = flat_index_for_action(sd, ad)
            if 0 <= idx < 25:
                flat_indices.add(int(idx))
        self.assertEqual(int(mask.sum()), len(flat_indices))


if __name__ == '__main__':
    unittest.main(verbosity=2)


