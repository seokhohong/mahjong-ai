#!/usr/bin/env python3
import unittest
import sys
import os
import json
import numpy as np

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.learn.pure_policy_dataset import generate_pure_policy_dataset  # type: ignore
from core.constants import MAX_TURNS  # type: ignore
from core.learn.pure_policy_dataset import extract_indexed_state  # type: ignore


class TestPurePolicyDataset(unittest.TestCase):
    def test_generate_small_dataset(self):
        # Prepare output path under training_data
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        out_dir = os.path.join(project_root, 'training_data')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'test_pure_policy_dataset.npz')
        if os.path.exists(out_path):
            os.remove(out_path)

        # Generate a very small dataset
        path = generate_pure_policy_dataset(num_games=2, seed=123, out_path=out_path)
        self.assertTrue(os.path.exists(path))

        # Load and validate contents
        npz = np.load(path, allow_pickle=True)
        for key in ['states', 'y_flat', 'rewards', 'game_ids', 'step_ids', 'action_labels']:
            self.assertIn(key, npz.files)

        states = npz['states']  # object array of dicts: {'hand_idx','disc_idx','game_state'}
        y_flat = npz['y_flat']
        rewards = npz['rewards']
        game_ids = npz['game_ids']
        step_ids = npz['step_ids']
        action_labels = npz['action_labels']

        # Basic sanity checks
        self.assertGreater(states.shape[0], 0)
        self.assertGreater(y_flat.shape[1], 60)
        self.assertEqual(states.shape[0], rewards.shape[0])
        self.assertEqual(states.shape[0], game_ids.shape[0])
        self.assertEqual(states.shape[0], step_ids.shape[0])
        self.assertGreater(action_labels.shape[0], 60)

        # Inspect the first state's structure
        s0 = states[0].item() if hasattr(states[0], 'item') else states[0]
        self.assertIn('hand_idx', s0)
        self.assertIn('disc_idx', s0)
        self.assertIn('game_state', s0)
        self.assertEqual(s0['hand_idx'].shape, (12,))
        self.assertEqual(s0['disc_idx'].shape, (4, MAX_TURNS))
        self.assertEqual(s0['game_state'].shape, (50,))
        self.assertIn('called_sets_idx', s0)
        self.assertEqual(s0['called_sets_idx'].shape, (4, 4, 3))

    def test_discard_rotation_per_perspective(self):
        # Construct minimal serialized states for two different perspectives
        sd_template = {
            'remaining_tiles': 72,
            'can_call': False,
            'player_hand': ['1p'] * 12,
            'called_sets': {0: [], 1: [], 2: [], 3: []},
            'last_discarded_tile': None,
            'last_discard_player': None,
            'player_discards': {
                0: ['1p'],
                1: ['2p'],
                2: ['3p'],
                3: ['4p'],
            },
        }

        sd_p0 = dict(sd_template)
        sd_p0['player_id'] = 0
        s0 = extract_indexed_state(sd_p0)

        sd_p2 = dict(sd_template)
        sd_p2['player_id'] = 2
        s2 = extract_indexed_state(sd_p2)

        # For player 0 perspective, disc_idx rows are [P0,P1,P2,P3]
        # For player 2 perspective, disc_idx rows are [P2,P3,P0,P1]
        # Therefore compare first entries accordingly
        self.assertEqual(s0['disc_idx'][0, 0], s2['disc_idx'][2, 0])  # P0
        self.assertEqual(s0['disc_idx'][1, 0], s2['disc_idx'][3, 0])  # P1
        self.assertEqual(s0['disc_idx'][2, 0], s2['disc_idx'][0, 0])  # P2
        self.assertEqual(s0['disc_idx'][3, 0], s2['disc_idx'][1, 0])  # P3

    def test_called_sets_encoding(self):
        # Player 0 has a pon of 3p; Player 2 has a pon of 5s. Check rotation and per-row mapping.
        sd_base = {
            'remaining_tiles': 72,
            'can_call': False,
            'player_hand': ['1p'] * 12,
            'called_sets': {
                0: [{
                    'tiles': ['3p', '3p', '3p'],
                    'call_type': 'pon',
                    'called_tile': '3p',
                    'caller_position': 0,
                    'source_position': 1,
                }],
                1: [],
                2: [{
                    'tiles': ['5s', '5s', '5s'],
                    'call_type': 'pon',
                    'called_tile': '5s',
                    'caller_position': 2,
                    'source_position': 3,
                }],
                3: [],
            },
            'last_discarded_tile': None,
            'last_discard_player': None,
            'player_discards': {0: [], 1: [], 2: [], 3: []},
        }

        sd_p0 = dict(sd_base)
        sd_p0['player_id'] = 0
        s0 = extract_indexed_state(sd_p0)
        cs0 = s0['called_sets_idx']

        sd_p2 = dict(sd_base)
        sd_p2['player_id'] = 2
        s2 = extract_indexed_state(sd_p2)
        cs2 = s2['called_sets_idx']

        # For p0 perspective: row mapping [P0,P1,P2,P3]
        # For p2 perspective: row mapping [P2,P3,P0,P1]
        # Verify p0's first row equals p2's third row (P0)
        self.assertTrue((cs0[0] == cs2[2]).all())
        # Verify p2's first row equals p0's third row (P2)
        self.assertTrue((cs2[0] == cs0[2]).all())
        # Rows without called sets should be all zeros
        self.assertTrue((cs0[1] == 0).all())
        self.assertTrue((cs0[3] == 0).all())
        self.assertTrue((cs2[1] == 0).all())

    def test_player_invariance_full_feature_set(self):
        # Build rotationally identical states: relative per-seat discards/called sets and last_discard_player rel=1
        base_discards_rel = {
            0: ['1p', '2p'],
            1: ['3p'],
            2: ['4p', '5p', '6p'],
            3: [],
        }
        base_called_rel = {
            0: [{
                'tiles': ['2s', '2s', '2s'],
                'call_type': 'pon', 'called_tile': '2s', 'caller_position': 0, 'source_position': 1,
            }],
            1: [],
            2: [{
                'tiles': ['3s', '4s', '5s'],
                'call_type': 'chi', 'called_tile': '4s', 'caller_position': 2, 'source_position': 3,
            }],
            3: [],
        }
        def build_sd(pid: int) -> dict:
            sd = {
                'remaining_tiles': 60,
                'can_call': True,
                'player_id': pid,
                'player_hand': ['9s'] * 12,
                'called_sets': {},
                'last_discarded_tile': '5p',
                'last_discard_player': (pid + 1) % 4,
                'player_discards': {},
            }
            # rotate discards and called sets
            cs_map = {}
            pd_map = {}
            for rel in range(4):
                abs_seat = (pid + rel) % 4
                cs_map[abs_seat] = base_called_rel[rel]
                pd_map[abs_seat] = base_discards_rel[rel]
            sd['called_sets'] = cs_map
            sd['player_discards'] = pd_map
            return sd

        extracted = [extract_indexed_state(build_sd(pid)) for pid in range(4)]
        # Compare all features across perspectives (player-invariant)
        ref = extracted[0]
        for e in extracted[1:]:
            self.assertTrue(np.array_equal(ref['hand_idx'], e['hand_idx']))
            self.assertTrue(np.array_equal(ref['disc_idx'], e['disc_idx']))
            self.assertTrue(np.array_equal(ref['called_sets_idx'], e['called_sets_idx']))
            self.assertTrue(np.array_equal(ref['game_state'], e['game_state']))


    def test_reward_assignment_for_immediate_ron_scenario(self):
        # Build a trivial, deterministic scenario:
        # - Player 1 discards 3p
        # - Player 2 can ron immediately on 3p
        from core.game import (
            SimpleJong,
            Player,
            GamePerspective,
            Tile,
            TileType,
            Suit,
            Discard,
            Ron,
            PassCall,
        )
        from core.learn.pure_policy_dataset import (
            _assign_rewards,
            serialize_action,
            serialize_state,
        )
        from core.learn.pure_policy import PurePolicyRecorder

        class DiscardThreeP(Player):
            def play(self, game_state: GamePerspective):
                return Discard(Tile(Suit.PINZU, TileType.THREE))

        class AlwaysRonIfPossible(Player):
            def play(self, game_state: GamePerspective):
                # Never tsumo for this test
                return Discard(game_state.player_hand[0])

            def choose_reaction(self, game_state: GamePerspective, options):
                if game_state.can_ron():
                    return Ron()
                return PassCall()

        class AlwaysPass(Player):
            def play(self, game_state: GamePerspective):
                return Discard(game_state.player_hand[0])

            def choose_reaction(self, game_state: GamePerspective, options):
                return PassCall()

        # Players: 0 pass, 1 discards 3p, 2 rons if possible, 3 passes
        base_players = [
            AlwaysPass(0),
            DiscardThreeP(1),
            AlwaysRonIfPossible(2),
            AlwaysPass(3),
        ]
        recorders = [PurePolicyRecorder(p) for p in base_players]
        game = SimpleJong(recorders)

        # Configure hands so player 1 has 3p to discard, and player 2 can ron on 3p
        base_s = [
            Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.TWO), Tile(Suit.SOUZU, TileType.THREE),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE), Tile(Suit.SOUZU, TileType.SIX),
            Tile(Suit.SOUZU, TileType.SEVEN), Tile(Suit.SOUZU, TileType.EIGHT), Tile(Suit.SOUZU, TileType.NINE),
        ]

        game._player_hands[0] = [Tile(Suit.SOUZU, TileType.ONE)] * 11
        game._player_hands[1] = [Tile(Suit.PINZU, TileType.THREE)] + [Tile(Suit.SOUZU, TileType.ONE)] * 10
        game._player_hands[2] = base_s + [Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR)]
        game._player_hands[3] = [Tile(Suit.SOUZU, TileType.ONE)] * 11

        # Allow a single draw, and start at player 1 so the first action is the discard of 3p
        game.tiles = [Tile(Suit.SOUZU, TileType.NINE)]
        game.current_player_idx = 1

        winner = game.play_round()
        self.assertEqual(winner, 2)
        self.assertEqual(game.get_winners(), [2])
        self.assertEqual(game.get_loser(), 1)

        # Compute per-player rewards as dataset generation does
        rewards = _assign_rewards(4, game.get_winners(), game.get_loser())

        # Collect serialized (state, action) pairs from recorders
        serialized = []
        for pid, rec in enumerate(recorders):
            for state, action, _ in rec.records:
                serialized.append((pid, serialize_state(state), serialize_action(action)))

        # Expect at least one discard by player 1 and one ron by player 2
        has_losing_discard = any(
            pid == 1 and entry['type'] == 'discard' and rewards[pid] == -1.0
            for pid, _, entry in serialized
        )
        has_winning_ron = any(
            pid == 2 and entry['type'] == 'ron' and rewards[pid] == 1.0
            for pid, _, entry in serialized
        )

        self.assertTrue(has_losing_discard, "Expected a recorded discard with -1.0 reward for the discarder")
        self.assertTrue(has_winning_ron, "Expected a recorded ron with +1.0 reward for the winner")


if __name__ == '__main__':
    unittest.main(verbosity=2)


