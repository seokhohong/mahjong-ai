#!/usr/bin/env python3
import os
import sys
import unittest
import numpy as np

# Make src importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.game import SimpleJong, Player, Tile, TileType, Suit, Discard
from core.learn.pure_policy import PurePolicyNetwork
from core.learn.pure_policy_player import PurePolicyPlayer
from core.learn.pure_policy_dataset import serialize_state, extract_indexed_state, get_action_index_map
from core.constants import MAX_CALLED_SETS_PER_PLAYER as _MCSP  # type: ignore


class TestTrainedPolicyNetwork(unittest.TestCase):
    # Accept comma-separated list via env MODEL_PATH or MODEL_PATHS; fallback to common location
    MODEL_PATHS = [
        os.path.join('models', 'copy_gen1.pt'),
        os.path.join('models', 'pure_policy_brief.pt')
    ]

    def _collect_models(self):
        paths: list[str] = []
        for p in self.MODEL_PATHS:
            p = p.strip()
            # allow omitting extension
            if not p.endswith('.pt') and os.path.exists(p + '.pt'):
                p = p + '.pt'
            if os.path.exists(p):
                paths.append(p)
        return paths

    def setUp(self):
        self._models = self._collect_models()
        if not self._models:
            self.skipTest("No trained model paths provided or found. Set MODEL_PATHS or MODEL_PATH env var.")

    def _predict_from_game_state(self, net: PurePolicyNetwork, g: SimpleJong, pid: int) -> np.ndarray:
        gs = g.get_game_perspective(pid)
        sd = serialize_state(gs)
        idx = extract_indexed_state(sd)
        hands = idx['hand_idx'][None, :]
        discs = idx['disc_idx'][None, :, :]
        cs = idx.get('called_sets_idx')
        if cs is None:
            cs = np.zeros((4, _MCSP, 3), dtype=np.int32)
        called = cs[None, :, :, :]
        gss = idx['game_state'][None, :]
        return net.model.predict([hands, discs, called, gss], verbose=0)[0]

    def test_tsumo_selected_on_trivial_tsumo_state(self):
        for model_path in self._models:
            with self.subTest(model=model_path):
                net = PurePolicyNetwork()
                net.load_model(model_path)

                # Player that asserts the selected move is Tsumo using the same inference path as gameplay
                from core.game import Tsumo  # local import to avoid polluting module scope
                class AssertTsumoPlayer(PurePolicyPlayer):
                    def play(self, game_state):
                        legal = game_state.legal_moves()
                        chosen = self._select_best_legal(game_state, legal)
                        assert isinstance(chosen, Tsumo), f"model={model_path} expected Tsumo, got {type(chosen).__name__ if chosen is not None else 'None'}"
                        return chosen

                # Create a SimpleJong game and set player 0 hand to a near-complete 11-tile hand; next draw completes it
                g = SimpleJong([AssertTsumoPlayer(0, net), Player(1), Player(2), Player(3)])
                g._player_hands[0] = [
                    Tile(Suit.PINZU, TileType.ONE), Tile(Suit.PINZU, TileType.ONE), Tile(Suit.PINZU, TileType.ONE),
                    Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.TWO),
                    Tile(Suit.SOUZU, TileType.THREE), Tile(Suit.SOUZU, TileType.THREE), Tile(Suit.SOUZU, TileType.THREE),
                    Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FOUR)
                ]
                # Ensure the next draw is the winning tile to make tsumo trivially legal
                g.tiles = [Tile(Suit.SOUZU, TileType.FOUR)]
                g.current_player_idx = 0

                # Run exactly one turn: draw the tile and have player 0 act; assertion occurs inside the player
                g.play_turn()

    def test_ron_selected_on_trivial_ron_state(self):
        for model_path in self._models:
            with self.subTest(model=model_path):
                net = PurePolicyNetwork()
                net.load_model(model_path)

                # Player that asserts the selected reaction is Ron using the same inference path as gameplay
                from core.game import Ron  # local import to avoid polluting module scope
                class AssertRonPlayer(PurePolicyPlayer):
                    def choose_reaction(self, game_state, options):
                        legal = [Ron()] if game_state.can_ron() else []
                        chosen = self._select_best_legal(game_state, legal)
                        assert isinstance(chosen, Ron), f"model={model_path} expected Ron, got {type(chosen).__name__ if chosen is not None else 'None'}"
                        return chosen

                # Create a SimpleJong game where player 1 can Ron on 3p discarded by player 0
                g = SimpleJong([Player(0), AssertRonPlayer(1, net), Player(2), Player(3)])
                # Player 1: nearly complete; wins with 3p
                g._player_hands[1] = [
                    Tile(Suit.PINZU, TileType.ONE), Tile(Suit.PINZU, TileType.ONE), Tile(Suit.PINZU, TileType.ONE),
                    Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.TWO),
                    Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FOUR),
                    Tile(Suit.SOUZU, TileType.THREE), Tile(Suit.SOUZU, TileType.THREE),
                ]
                # Player 0 will discard 3p using isolation heuristic
                g._player_hands[0] = [Tile(Suit.PINZU, TileType.THREE)] + [Tile(Suit.SOUZU, TileType.ONE)] * 10
                # No draw to keep state stable; immediate discard by player 0
                g.tiles = []
                g.current_player_idx = 0

                # Run exactly one turn: player 0 discards 3p, reaction is solicited; assertion occurs inside player 1
                g.play_turn()


    # this test is failing because the network is not predicting legal moves right now
    def test_predicts_legal(self):
        """Ensure that whenever the network is consulted (action or reaction),
        the argmax of its predicted policy corresponds to a legal move index.
        """
        for model_path in self._models:
            with self.subTest(model=model_path):
                net = PurePolicyNetwork()
                net.load_model(model_path)

                # Define an audited player that asserts legality of the network argmax
                class AuditedPurePolicyPlayer(PurePolicyPlayer):
                    def _assert_argmax_legal(self, gs: 'SimpleJong.GamePerspective') -> None:  # type: ignore[name-defined]
                        import numpy as _np  # local import to avoid polluting module
                        probs = self.predict_policy_probs(gs)
                        mask = self._game.legality_mask(self.player_id)  # type: ignore[attr-defined]
                        idx = int(_np.argmax(probs))
                        assert bool(mask[idx]), f"model={getattr(self, '_model_path', '?')} argmax illegal idx={idx}"

                    def play(self, game_state):  # action phase
                        self._assert_argmax_legal(game_state)
                        return super().play(game_state)

                    def choose_reaction(self, game_state, options):  # reaction phase
                        self._assert_argmax_legal(game_state)
                        return super().choose_reaction(game_state, options)

                # Seat 0 uses the audited pure policy player; others are baseline
                p0 = AuditedPurePolicyPlayer(0, net)
                setattr(p0, '_model_path', model_path)
                g = SimpleJong([
                    p0,
                    Player(1),
                    Player(2),
                    Player(3),
                ])
                # Provide back-reference to game for legality mask access
                try:
                    setattr(p0, '_game', g)
                except Exception:
                    pass

                # Play several rounds to exercise both action and reaction consultations
                for _ in range(2):
                    g.play_round()
                    p0 = AuditedPurePolicyPlayer(0, net)
                    setattr(p0, '_model_path', model_path)
                    g = SimpleJong([
                        p0,
                        Player(1),
                        Player(2),
                        Player(3),
                    ])
                    try:
                        setattr(p0, '_game', g)
                    except Exception:
                        pass

if __name__ == '__main__':
    unittest.main(verbosity=2)


