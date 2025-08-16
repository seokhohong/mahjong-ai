#!/usr/bin/env python3
import unittest
import sys
import os
import numpy as np

from core.learn_ac import RecordingHeuristicACPlayer

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from core.game import SimpleJong, Ron, Tsumo, Tile, Suit, TileType, Player
from core.constants import TILE_COPIES_DEFAULT
from core.learn_ac.ac_constants import chi_variant_index


class TestRecordingACPlayer(unittest.TestCase):
    def setUp(self):
        import random
        random.seed(123)
        # Use heuristic recording players which do not require torch
        self.players = [RecordingHeuristicACPlayer(i) for i in range(4)]
        # Simulate 10 games to populate buffers
        for _ in range(10):
            game = SimpleJong(self.players, tile_copies=TILE_COPIES_DEFAULT)
            game.play_round()

    def test_records_experiences(self):
        # Each player should have recorded some experiences
        total = 0
        for p in self.players:
            self.assertGreater(len(p.experience), 0)
            total += len(p.experience)
            # Sanity: states and actions align
            self.assertEqual(len(p.experience.states), len(p.experience.actions))
            self.assertEqual(len(p.experience.states), len(p.experience.rewards))
        self.assertGreater(total, 0)

    def test_chi_variant_index(self):
        # Build minimal game state to assign last_discarded_tile
        players = [Player(0), Player(1), Player(2), Player(3)]
        game = SimpleJong(players)
        # Configure last discard = 3p
        game.last_discarded_tile = Tile(Suit.PINZU, TileType.THREE)
        # left/low variant: tiles [1p,2p]
        left_tiles = [Tile(Suit.PINZU, TileType.ONE), Tile(Suit.PINZU, TileType.TWO)]
        self.assertEqual(chi_variant_index(game.last_discarded_tile, left_tiles), 0)
        # mid variant: tiles [2p,4p]
        mid_tiles = [Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR)]
        self.assertEqual(chi_variant_index(game.last_discarded_tile, mid_tiles), 1)
        # right/high variant: tiles [4p,5p]
        right_tiles = [Tile(Suit.PINZU, TileType.FOUR), Tile(Suit.PINZU, TileType.FIVE)]
        self.assertEqual(chi_variant_index(game.last_discarded_tile, right_tiles), 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)


