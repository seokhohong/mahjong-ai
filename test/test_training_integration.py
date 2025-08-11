#!/usr/bin/env python3
import os
import sys
import unittest
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.game import SimpleJong, Player  # type: ignore
from core.learn.pure_policy import PurePolicyNetwork  # type: ignore
from core.learn.pure_policy_player import PurePolicyPlayer  # type: ignore
from core.learn.pure_policy_dataset import Recorder, serialize_state, encode_action_flat_index, get_action_index_map, _tile_index_from_str  # type: ignore
from core.learn.rule_copy_network import RuleCopyNetwork  # type: ignore
from core.learn.rule_copy_player import RuleCopyPlayer  # type: ignore


class RecordingPurePolicyPlayer(PurePolicyPlayer):
    """Extend PurePolicyPlayer to record actions via provided Recorder.

    The same instance will be attached to all four seats.
    """
    def __init__(self, player_id: int, network: PurePolicyNetwork, recorder: Recorder):
        super().__init__(player_id, network)
        self._recorder = recorder

    def play(self, game_state):  # type: ignore[override]
        move = super().play(game_state)
        if self._recorder is not None:
            # Record the move with the raw GamePerspective for later verification
            self._recorder.record(game_state, self.player_id, move, self.predict_policy_probs(game_state), self._game.legality_mask(self.player_id) if getattr(self, "_game", None) is not None else None)
        return move

    def choose_reaction(self, game_state, options):  # type: ignore[override]
        move = super().choose_reaction(game_state, options)
        if self._recorder is not None:
            self._recorder.record(game_state, self.player_id, move, self.predict_policy_probs(game_state), self._game.legality_mask(self.player_id) if getattr(self, "_game", None) is not None else None)
        return move


class TestTrainingIntegration(unittest.TestCase):
    def test_feature_consistency(self):
        # Tiny network for speed
        import random
        random.seed(0)
        net = PurePolicyNetwork(hidden_size=16, embedding_dim=4)

        # Shared recorder and a single player instance used across all seats
        rec = Recorder()
        players = [RecordingPurePolicyPlayer(i, net, rec) for i in range(4)]

        g = SimpleJong(players)
        g.play_round()

        self.assertGreater(len(rec.events), 0, "should record some events")

        # For each recorded (GamePerspective, action), verify parity
        for idx, compound in enumerate(zip(rec.events, rec.event_probs)):
            tup, event_prob = compound
            actor_id, gp, action_obj = tup
            from core.learn.pure_policy_dataset import serialize_action as _ser_act  # type: ignore
            # Use the player's probability head on the raw GamePerspective. All players should have the same net
            probs = players[0].predict_policy_probs(gp)
            self.assertTrue(np.allclose(probs, players[1].predict_policy_probs(gp)))
            self.assertTrue(np.allclose(probs, event_prob))

    def test_copy_accuracy(self):
        # Locate trained rule-copy model
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        model_path = os.path.join(repo_root, 'models', 'rule_copy_5k.pt')
        if not os.path.exists(model_path):
            self.skipTest(f"Missing model checkpoint: {model_path}")

        # Load network once
        net = RuleCopyNetwork(hidden_size=128, embedding_dim=4)
        net.load_model(model_path)

        rng = np.random.RandomState(123)
        total_decisions = 0
        copied_decisions = 0

        # Simulate 10 games; at each actor turn, compare base Player vs RuleCopyPlayer decisions
        # This test is fragile since it replicates SimpleJong's logic rather than using it
        for g_idx in range(10):
            # Fresh game with four baseline players (we'll decide actions externally)
            players = [Player(i) for i in range(4)]
            game = SimpleJong(players)

            # Play until terminal, advancing the environment using base decisions
            while not game.game_over and (game.tiles or game.last_discarded_tile is not None):
                # Resolve any outstanding reactions first
                if game._resolve_outstanding_reactions_if_any():
                    break

                # Start of turn: draw if needed
                game._draw_for_current_if_needed()

                # Current actor and perspective (action phase)
                actor_id = game.current_player_idx
                gs = game.get_game_perspective(actor_id)

                # Construct deciders attached to the live game
                base_decider = Player(actor_id)
                base_decider._game = game  # type: ignore[attr-defined]
                copy_decider = RuleCopyPlayer(actor_id, net)
                copy_decider._game = game  # type: ignore[attr-defined]

                # Base and copy decisions for the same state/position
                base_move = base_decider.play(gs)
                copy_move = copy_decider.play(gs)

                # Count match on action type and payload (tile) for action phase
                total_decisions += 1
                same = type(base_move) is type(copy_move)
                try:
                    # For discards, also require same tile
                    from core.game import Discard as _Discard  # type: ignore
                    if isinstance(base_move, _Discard) and isinstance(copy_move, _Discard):
                        same = same and (base_move.tile == copy_move.tile)
                except Exception:
                    pass
                if same:
                    copied_decisions += 1

                # Step the environment using the base player's decision
                game.step(actor_id, base_move)
                if game.game_over:
                    break

                # If a discard was made, resolve reactions immediately
                if game.last_discarded_tile is not None and game.last_discard_player is not None:
                    if game._resolve_reactions_after_discard():
                        break
                    # If a call transferred the turn, continue to next loop without advancing
                    if game._skip_draw_for_current:
                        continue

                # Advance to next actor otherwise
                game.current_player_idx = (game.current_player_idx + 1) % 4

                # End game if wall empty and no pending discard
                if not game.tiles and game.last_discarded_tile is None:
                    game.game_over = True
                    break

        accuracy = (copied_decisions / max(1, total_decisions))
        self.assertGreaterEqual(accuracy, 0.80, f"RuleCopyPlayer match rate too low: {accuracy*100:.2f}% (need >= 80%)")


if __name__ == '__main__':
    unittest.main(verbosity=2)


