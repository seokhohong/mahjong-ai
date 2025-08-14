#!/usr/bin/env python3
import sys
import os
import argparse
import random

# Ensure src is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from core.game import SimpleJong, Player, Tile, Suit, TileType  # type: ignore
from core.learn.pure_policy_dataset import Recorder, RecordingPlayer, serialize_action  # type: ignore



def main():
    players = [Player(0), Player(1), Player(2), Player(3)]
    game = SimpleJong(players)
    # Discard 3p by player 0; players 1 and 2 can ron
    game.last_discarded_tile = Tile(Suit.PINZU, TileType.THREE)
    game.last_discard_player = 0
    base_s = [
        Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.TWO), Tile(Suit.SOUZU, TileType.THREE),
        Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE), Tile(Suit.SOUZU, TileType.SIX),
        Tile(Suit.SOUZU, TileType.SEVEN), Tile(Suit.SOUZU, TileType.EIGHT), Tile(Suit.SOUZU, TileType.NINE),
    ]
    game._player_hands[1] = base_s + [Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR)]
    game._player_hands[2] = base_s + [Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR)]
    game._player_hands[3] = [Tile(Suit.SOUZU, TileType.ONE)] * 11
    game.tiles = []
    game.play_round()
    winners = set(game.get_winners())


if __name__ == '__main__':
    sys.exit(main())
