#!/usr/bin/env python3
import unittest
import sys
import os
import json
import numpy as np

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.learn.pure_policy_dataset import generate_pure_policy_dataset, RecordingPlayer  # type: ignore
from core.constants import MAX_TURNS  # type: ignore
from core.learn.pure_policy_dataset import extract_indexed_state  # type: ignore


class TestPurePolicyDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Prepare a shared dataset of 50 games for tests that need broader coverage
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        out_dir = os.path.join(project_root, 'training_data')
        os.makedirs(out_dir, exist_ok=True)
        cls.large_out_path = os.path.join(out_dir, 'test_pure_policy_dataset_50.npz')
        if os.path.exists(cls.large_out_path):
            os.remove(cls.large_out_path)
        cls.large_path = generate_pure_policy_dataset(num_games=50, seed=4321, out_path=cls.large_out_path)
        cls.large_data = np.load(cls.large_path, allow_pickle=True)

    def test_generate_small_dataset(self):
        # Reuse the 50-game dataset produced in setUpClass for consistency
        npz = self.large_data
        for key in ['states', 'y_flat', 'rewards', 'game_ids', 'step_ids', 'action_labels', 'legal_masks']:
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
        legal_masks = npz['legal_masks']
        self.assertEqual(legal_masks.shape[0], states.shape[0])
        self.assertEqual(legal_masks.shape[1], y_flat.shape[1])

        # Rewards should span negative, zero, and positive categories
        import numpy as _np
        has_neg = _np.any(rewards < 0.0)
        has_zero = _np.any(rewards == 0.0)
        has_pos = _np.any(rewards > 0.0)
        self.assertTrue(has_neg, 'Expected at least one negative reward')
        self.assertTrue(has_zero, 'Expected at least one zero reward')
        self.assertTrue(has_pos, 'Expected at least one positive reward')

        # Additional sanity: require >= 2 legal actions on every step
        # Engine guarantees PassCall in reaction phase and multiple discards in action phase
        import numpy as _np
        sums = _np.sum(legal_masks, axis=1).astype(int)
        self.assertTrue(_np.all(sums >= 2), f"Found steps with fewer than 2 legal actions (min={int(sums.min())})")

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


    def test_rewards_consistent_for_winning_player_per_game(self):
        # Use the shared 50-game dataset from setUpClass
        rewards = self.large_data['rewards']  # (N,)
        game_ids = self.large_data['game_ids']  # (N,)

        # For each game: if any reward==1 appears, then all samples in that game
        # with positive reward should be exactly 1 (i.e., consistent winner reward)
        unique_games = np.unique(game_ids)
        for gid in unique_games:
            mask = (game_ids == gid)
            r = rewards[mask]
            if np.any(r == 1.0):
                # All positive rewards should be 1.0
                pos = r[r > 0.0]
                self.assertTrue(np.all(pos == 1.0), f"Game {gid}: found non-1 positive reward values {np.unique(pos)}")

    def test_rewards_positive_for_win_actions_when_present(self):
        # If any row is labeled as 'ron' or 'tsumo', ensure its reward > 0
        y_flat = self.large_data['y_flat']  # (N, num_actions)
        rewards = self.large_data['rewards']  # (N,)
        action_labels = self.large_data['action_labels']
        # Ensure python list of labels
        labels = [str(x) for x in (action_labels.tolist() if hasattr(action_labels, 'tolist') else action_labels)]
        try:
            ron_idx = labels.index('ron')
        except ValueError:
            ron_idx = -1
        try:
            tsumo_idx = labels.index('tsumo')
        except ValueError:
            tsumo_idx = -1
        mask = np.zeros((y_flat.shape[0],), dtype=bool)
        if ron_idx >= 0:
            mask |= (y_flat[:, ron_idx] == 1.0)
        if tsumo_idx >= 0:
            mask |= (y_flat[:, tsumo_idx] == 1.0)
        self.assertTrue(np.any(mask), "No 'ron' or 'tsumo' actions recorded in dataset")

    def test_reward_assignment_for_immediate_ron_scenario(self):
        # Build a trivial, deterministic scenario:
        # - Player 1 discards 3p
        # - Player 2 can ron immediately on 3p
        from core.game import (
            SimpleJong,
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
            Recorder,
            RecordingPlayer,
        )

        # Wrap behaviors with a recording player so actions (including Ron) are explicitly captured
        class RecDiscardThreeP(RecordingPlayer):
            def play(self, game_state: GamePerspective):  # type: ignore[override]
                legal_mask = self._game.legality_mask(self.player_id)
                action = Discard(Tile(Suit.PINZU, TileType.THREE))
                self._rec.record(game_state, self.player_id, action, None, legal_mask)
                return action

        class RecAlwaysRonIfPossible(RecordingPlayer):
            def play(self, game_state: GamePerspective):  # type: ignore[override]
                legal_mask = self._game.legality_mask(self.player_id)
                action = Discard(game_state.player_hand[0])
                self._rec.record(game_state, self.player_id, action, None, legal_mask)
                return action

            def choose_reaction(self, game_state: GamePerspective, options):  # type: ignore[override]
                legal_mask = self._game.legality_mask(self.player_id)
                if game_state.can_ron():
                    reaction = Ron()
                else:
                    reaction = PassCall()
                self._rec.record(game_state, self.player_id, reaction, None, legal_mask)
                return reaction

        class RecAlwaysPass(RecordingPlayer):
            def play(self, game_state: GamePerspective):  # type: ignore[override]
                legal_mask = self._game.legality_mask(self.player_id)
                action = Discard(game_state.player_hand[0])
                self._rec.record(game_state, self.player_id, action, None, legal_mask)
                return action

            def choose_reaction(self, game_state: GamePerspective, options):  # type: ignore[override]
                legal_mask = self._game.legality_mask(self.player_id)
                reaction = PassCall()
                self._rec.record(game_state, self.player_id, reaction, None, legal_mask)
                return reaction

        rec = Recorder()
        players = [
            RecAlwaysPass(0, rec),
            RecDiscardThreeP(1, rec),
            RecAlwaysRonIfPossible(2, rec),
            RecAlwaysPass(3, rec),
        ]
        game = SimpleJong(players)

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

        # Ensure Ron was explicitly recorded as an action for player 2
        ron_actions = [
            (actor_id, serialize_action(action_obj))
            for (actor_id, gp, action_obj) in rec.events
            if actor_id == 2
        ]
        self.assertTrue(any(ad.get('type') == 'ron' for _, ad in ron_actions), "Expected explicit 'ron' action for player 2")

        # All moves by player 1 should have reward -1; all by player 2 should have reward +1
        for actor_id, gp, action_obj in rec.events:
            if actor_id == 1:
                self.assertEqual(rewards[actor_id], -1.0)
            if actor_id == 2:
                self.assertEqual(rewards[actor_id], 1.0)

    def test_simulate_game_collecting(self):
        # Fix random seed for reproducibility so the game is unlikely to end too early
        import random
        random.seed(12345)
        from core.learn.pure_policy_dataset import _simulate_game_collecting
        res = _simulate_game_collecting()
        log, winners, loser, masks = res
        # Ensure we collected a reasonable number of steps
        self.assertGreaterEqual(len(log), 5, f"Collected only {len(log)} steps; update the seed if flaky")

    def test_serialize(self):
        # Ensure serialize_action always yields known action types
        from core.learn.pure_policy_dataset import serialize_action  # type: ignore
        from core.game import Tsumo, Ron, Discard, Pon, Chi, PassCall, Tile, Suit, TileType  # type: ignore

        actions = [
            Tsumo(),
            Ron(),
            Discard(Tile(Suit.PINZU, TileType.THREE)),
            Pon([Tile(Suit.PINZU, TileType.FIVE), Tile(Suit.PINZU, TileType.FIVE)]),
            Chi([Tile(Suit.SOUZU, TileType.TWO), Tile(Suit.SOUZU, TileType.FOUR)]),
            PassCall(),
        ]

        for a in actions:
            ad = serialize_action(a)
            self.assertIn('type', ad)
            self.assertNotEqual(ad.get('type'), 'unknown', f"Serialized unknown for {type(a).__name__}")
            # Spot checks for expected payload keys
            if ad['type'] == 'discard':
                self.assertIn('tile', ad)
                self.assertIsInstance(ad['tile'], str)
            if ad['type'] in ('pon', 'chi'):
                self.assertIn('tiles', ad)
                self.assertIsInstance(ad['tiles'], list)
                self.assertTrue(all(isinstance(t, str) for t in ad['tiles']))

    def test_recorder(self):
        # Validate that recorder captures only known, serializable actions
        from core.learn.pure_policy_dataset import Recorder, serialize_action  # type: ignore
        from core.game import SimpleJong, Player, Discard  # type: ignore

        rec = Recorder()
        players = [RecordingPlayer(i, rec) for i in range(4)]
        game = SimpleJong(players)
        game.play_round()

        # Ensure we recorded some events
        self.assertGreater(len(rec.events), 0)

        # Check that all serialized actions are recognized (not 'unknown')
        unknowns = []
        for actor_id, gp, action_obj in rec.events:  # type: ignore[attr-defined]
            ad = serialize_action(action_obj)
            if ad.get('type') == 'unknown':
                unknowns.append((actor_id, type(action_obj).__name__, str(action_obj)))
        self.assertEqual(len(unknowns), 0, f"Found unknown serialized actions: {unknowns}")


if __name__ == '__main__':
    unittest.main(verbosity=2)


