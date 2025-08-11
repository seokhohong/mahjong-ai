#!/usr/bin/env python3
import os
import sys
import unittest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.game import (  # type: ignore
    Player,
    GamePerspective,
    Tile,
    TileType,
    Suit,
    CalledSet,
    Action,
    Discard,
)


class TestBasePlayer(unittest.TestCase):
    def test_heuristics(self):
        # Recreate the snapshot from the divergence log
        # player_id=1, action phase, current turn, no last discard/drawer
        player_id = 1

        # Hand: [5s, 1s, 6p, 1s, 8p, 3s]
        hand = [
            Tile(Suit.SOUZU, TileType.FIVE),
            Tile(Suit.SOUZU, TileType.ONE),
            Tile(Suit.PINZU, TileType.SIX),
            Tile(Suit.SOUZU, TileType.ONE),
            Tile(Suit.PINZU, TileType.EIGHT),
            Tile(Suit.SOUZU, TileType.THREE),
        ]

        # Called sets per snapshot (values do not affect discard heuristic here)
        called_sets = {
            0: [],
            1: [
                CalledSet(
                    tiles=[Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.ONE), Tile(Suit.PINZU, TileType.THREE)],
                    call_type='chi',
                    called_tile=Tile(Suit.PINZU, TileType.TWO),
                    caller_position=1,
                    source_position=0,
                ),
                CalledSet(
                    tiles=[Tile(Suit.PINZU, TileType.FOUR), Tile(Suit.PINZU, TileType.FOUR), Tile(Suit.PINZU, TileType.FOUR)],
                    call_type='pon',
                    called_tile=Tile(Suit.PINZU, TileType.FOUR),
                    caller_position=1,
                    source_position=0,
                ),
            ],
            2: [
                CalledSet(
                    tiles=[Tile(Suit.SOUZU, TileType.SEVEN), Tile(Suit.SOUZU, TileType.EIGHT), Tile(Suit.SOUZU, TileType.NINE)],
                    call_type='chi',
                    called_tile=Tile(Suit.SOUZU, TileType.SEVEN),
                    caller_position=2,
                    source_position=1,
                )
            ],
            3: [
                CalledSet(
                    tiles=[Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.ONE), Tile(Suit.PINZU, TileType.THREE)],
                    call_type='chi',
                    called_tile=Tile(Suit.PINZU, TileType.TWO),
                    caller_position=3,
                    source_position=2,
                )
            ],
        }

        other_discards = {0: [], 2: [], 3: []}

        gs = GamePerspective(
            player_hand=hand,
            remaining_tiles=50,
            player_id=player_id,
            other_players_discarded=other_discards,
            called_sets=called_sets,
            last_discarded_tile=None,
            last_discard_player=None,
            can_call=False,
            state=Action,
            newly_drawn_tile=None,
            is_current_turn=True,
        )

        base = Player(player_id)
        move = base.play(gs)

        # After fixing neighbor counting to include identical tiles as neighbors (but not itself),
        # the pair of 1s is no longer the most isolated. With current tie-breakers, this
        # hand leads to discarding 8p.
        self.assertIsInstance(move, Discard)
        self.assertEqual(str(move.tile), '8p')

    def test_heuristics_start_heavy_ones_discards_five(self):
        # Start-of-game like: no called sets, current player's action
        player_id = 0

        # Hand: 10x 1s, 1x 3s, 1x 5s (all Souzu) => should discard 5s as most isolated
        hand = [Tile(Suit.SOUZU, TileType.ONE) for _ in range(10)]
        hand += [Tile(Suit.SOUZU, TileType.THREE), Tile(Suit.SOUZU, TileType.FIVE)]

        called_sets = {0: [], 1: [], 2: [], 3: []}
        other_discards = {1: [], 2: [], 3: []}

        gs = GamePerspective(
            player_hand=hand,
            remaining_tiles=50,
            player_id=player_id,
            other_players_discarded=other_discards,
            called_sets=called_sets,
            last_discarded_tile=None,
            last_discard_player=None,
            can_call=False,
            state=Action,
            newly_drawn_tile=None,
            is_current_turn=True,
        )

        base = Player(player_id)
        move = base.play(gs)
        self.assertIsInstance(move, Discard)
        self.assertEqual(str(move.tile), '5s')


if __name__ == '__main__':
    unittest.main(verbosity=2)


