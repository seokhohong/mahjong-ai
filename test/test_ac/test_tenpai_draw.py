#!/usr/bin/env python3
import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from core.game import SimpleJong, Tile, Suit, TileType, Player


class DummyPlayer(Player):
    def play(self, game_state):  # type: ignore[override]
        # Always discard first tile; no tsumo
        from core.game import Discard
        return Discard(game_state.player_hand[0])


class TestTenpaiDraw(unittest.TestCase):
    def test_tenpai_assignment_on_draw(self):
        players = [DummyPlayer(i) for i in range(4)]
        g = SimpleJong(players, tile_copies=1)
        # Construct hands near completion for some players (manual override)
        # p0: 12 tiles that become complete with any 1p -> tenpai
        g._player_hands[0] = [Tile(Suit.PINZU, TileType.ONE), Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.THREE),
                              Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.TWO), Tile(Suit.SOUZU, TileType.THREE),
                              Tile(Suit.PINZU, TileType.FOUR), Tile(Suit.PINZU, TileType.FIVE), Tile(Suit.PINZU, TileType.SIX),
                              Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE), Tile(Suit.SOUZU, TileType.SIX)]
        # p1: 12 tiles not in tenpai (random mix)
        g._player_hands[1] = [Tile(Suit.PINZU, TileType.ONE), Tile(Suit.SOUZU, TileType.TWO), Tile(Suit.PINZU, TileType.THREE),
                              Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.PINZU, TileType.FIVE), Tile(Suit.SOUZU, TileType.SIX),
                              Tile(Suit.PINZU, TileType.SEVEN), Tile(Suit.SOUZU, TileType.EIGHT), Tile(Suit.PINZU, TileType.NINE),
                              Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.PINZU, TileType.TWO), Tile(Suit.SOUZU, TileType.THREE)]
        # p2, p3: arbitrary
        g._player_hands[2] = list(g._player_hands[0])
        g._player_hands[3] = list(g._player_hands[1])

        # Empty the wall to force draw
        g.tiles = []
        g.last_discarded_tile = None
        g.check_game_over()

        self.assertTrue(g.is_game_over())
        winners = g.get_winners()
        # At least one of p0/p2 should be tenpai
        self.assertTrue(0 in winners or 2 in winners)
        # At least one of p1/p3 should be non-tenpai
        self.assertTrue(not (1 in winners and 3 in winners))


if __name__ == '__main__':
    unittest.main(verbosity=2)


