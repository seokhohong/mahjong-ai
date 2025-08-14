#!/usr/bin/env python3
import os
import sys
import json
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
        # Use engine's single-turn API to avoid replicating logic
        for g_idx in range(10):
            # Fresh game with four baseline players (we'll decide actions externally)
            players = [Player(i) for i in range(4)]
            game = SimpleJong(players)

            # Play until terminal; at each new actionable state, compare decisions
            while not game.game_over and (game.tiles or game.last_discarded_tile is not None):
                # Snapshot state before engine acts
                actor_id = game.current_player_idx
                gs = game.get_game_perspective(actor_id)

                base_decider = Player(actor_id)
                base_decider._game = game  # type: ignore[attr-defined]
                copy_decider = RuleCopyPlayer(actor_id, net)
                copy_decider._game = game  # type: ignore[attr-defined]

                base_move = base_decider.play(gs)
                copy_move = copy_decider.play(gs)

                total_decisions += 1
                same = type(base_move) is type(copy_move)
                try:
                    from core.game import Discard as _Discard  # type: ignore
                    if isinstance(base_move, _Discard) and isinstance(copy_move, _Discard):
                        same = same and (base_move.tile == copy_move.tile)
                except Exception:
                    pass
                if same:
                    copied_decisions += 1
                else:
                    try:
                        from core.game import Tsumo as _Tsumo, Ron as _Ron, Discard as _Discard, Pon as _Pon, Chi as _Chi, PassCall as _Pass  # type: ignore
                        from core.learn.pure_policy_dataset import serialize_action as _ser_act  # type: ignore
                        def _action_to_str(a):
                            if isinstance(a, _Discard):
                                return f"Discard({a.tile})"
                            if isinstance(a, _Tsumo):
                                return "Tsumo()"
                            if isinstance(a, _Ron):
                                return "Ron()"
                            if isinstance(a, _Pon):
                                return "Pon([" + ", ".join(str(t) for t in a.tiles) + "])"
                            if isinstance(a, _Chi):
                                return "Chi([" + ", ".join(str(t) for t in a.tiles) + "])"
                            if isinstance(a, _Pass):
                                return "Pass()"
                            return repr(a)

                        sd = serialize_state(gs)
                        ldt = sd.get('last_discarded_tile')
                        base_idx = encode_action_flat_index(_ser_act(base_move), ldt)
                        copy_idx = encode_action_flat_index(_ser_act(copy_move), ldt)
                        print("\n=== Divergence detected ===")
                        print(f"game={g_idx} decision_idx={total_decisions} actor={actor_id}")
                        print(f"base_move={_action_to_str(base_move)} (idx={base_idx})  copy_move={_action_to_str(copy_move)} (idx={copy_idx})")
                        phase_name = getattr(gs.state, '__name__', str(gs.state))
                        extras = {
                            'phase': phase_name,
                            'is_current_turn': bool(getattr(gs, 'is_current_turn', False)),
                            'player_id': int(getattr(gs, 'player_id', -1)),
                            'player_hand': [str(t) for t in getattr(gs, 'player_hand', [])],
                            'newly_drawn_tile': str(getattr(gs, 'newly_drawn_tile', None)) if getattr(gs, 'newly_drawn_tile', None) is not None else None,
                            'last_discarded_tile': str(getattr(gs, 'last_discarded_tile', None)) if getattr(gs, 'last_discarded_tile', None) is not None else None,
                            'last_discard_player': getattr(gs, 'last_discard_player', None),
                            'called_sets': {pid: [
                                {
                                    'call_type': getattr(cs, 'call_type', 'unknown'),
                                    'tiles': [str(t) for t in getattr(cs, 'tiles', [])],
                                } for cs in sets
                            ] for pid, sets in getattr(gs, 'called_sets', {}).items()},
                            'other_players_discarded': {pid: [str(t) for t in tiles] for pid, tiles in getattr(gs, 'other_players_discarded', {}).items()},
                        }
                        print("Extras:")
                        print(json.dumps(extras, indent=2, sort_keys=True))
                        print("GamePerspective (serialized):")
                        print(json.dumps(sd, indent=2, sort_keys=True))
                    except Exception as e:
                        print(f"[warn] failed to print divergence details: {e}")

                # Now let the engine perform the actual turn, which may alter current_player_idx
                game.play_turn()
                if game.is_game_over():
                    break

        accuracy = (copied_decisions / max(1, total_decisions))
        self.assertGreaterEqual(accuracy, 0.90, f"RuleCopyPlayer match rate too low: {accuracy*100:.2f}% (need >= 90%)")


if __name__ == '__main__':
    unittest.main(verbosity=2)


