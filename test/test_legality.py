import unittest
import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.game import (
    SimpleJong, Player, Tile, TileType, Suit,
    Discard, Tsumo, Ron, Pon, Chi, CalledSet, PassCall
)


class TestStepLegality(unittest.TestCase):
    def setUp(self):
        self.players = [Player(i) for i in range(4)]
        self.game = SimpleJong(self.players)

    def test_illegal_ron_without_discard(self):
        # No outstanding discard; Ron should be illegal
        self.assertFalse(self.game.is_legal(1, Ron()))
        with self.assertRaises(SimpleJong.IllegalMoveException):
            self.game.step(1, Ron())
        self.assertFalse(self.game.is_game_over())

    def test_illegal_discard_by_non_current_player(self):
        # Current player is 0 by default
        tile = self.game.hand(1)[0]
        self.assertFalse(self.game.is_legal(1, Discard(tile)))
        with self.assertRaises(SimpleJong.IllegalMoveException):
            self.game.step(1, Discard(tile))
        self.assertIsNone(self.game.last_discarded_tile)

    def test_illegal_discard_tile_not_in_hand(self):
        # Pick a tile not in player 0's hand
        current_hand = self.game.hand(0)
        # Construct a tile guaranteed not in hand by flipping suit/number if possible
        candidate = Tile(Suit.PINZU, TileType.ONE)
        if candidate in current_hand:
            candidate = Tile(Suit.SOUZU, TileType.NINE)
            if candidate in current_hand:
                # Find any tile not in hand
                all_tiles = [Tile(s, TileType(v)) for s in (Suit.PINZU, Suit.SOUZU) for v in range(1, 10)]
                for t in all_tiles:
                    if t not in current_hand:
                        candidate = t
                        break
        self.assertFalse(self.game.is_legal(0, Discard(candidate)))
        with self.assertRaises(SimpleJong.IllegalMoveException):
            self.game.step(0, Discard(candidate))
        self.assertIsNone(self.game.last_discarded_tile)

    def test_legal_discard_by_current_player(self):
        tile = self.game.hand(0)[0]
        self.assertTrue(self.game.is_legal(0, Discard(tile)))
        applied = self.game.step(0, Discard(tile))
        self.assertTrue(applied)
        self.assertEqual(self.game.last_discarded_tile, tile)
        self.assertEqual(self.game.last_discard_player, 0)

    def test_legal_ron_after_discard(self):
        # Configure a situation where player 1 can ron on 3p discarded by player 0
        game = SimpleJong([Player(0), Player(1), Player(2), Player(3)])
        # Set hands: player 0 has 3p to discard
        game._player_hands[0] = [Tile(Suit.PINZU, TileType.THREE)] + [Tile(Suit.SOUZU, TileType.ONE)] * 10
        # Player 1 holds 11 tiles: 123s,456s,789s and 2p,4p
        base_s = [
            Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.TWO), Tile(Suit.SOUZU, TileType.THREE),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE), Tile(Suit.SOUZU, TileType.SIX),
            Tile(Suit.SOUZU, TileType.SEVEN), Tile(Suit.SOUZU, TileType.EIGHT), Tile(Suit.SOUZU, TileType.NINE),
        ]
        game._player_hands[1] = base_s + [Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR)]
        game.tiles = []
        game.current_player_idx = 0
        # Discard 3p by player 0
        discard_tile = Tile(Suit.PINZU, TileType.THREE)
        self.assertTrue(game.is_legal(0, Discard(discard_tile)))
        self.assertTrue(game.step(0, Discard(discard_tile)))
        # Player 1 rons
        self.assertTrue(game.is_legal(1, Ron()))
        applied = game.step(1, Ron())
        self.assertTrue(applied)
        self.assertTrue(game.is_game_over())
        self.assertEqual(game.get_winners(), [1])
        self.assertEqual(game.get_loser(), 0)

    def test_double_ron_on_same_discard(self):
        # Player 0 discards 3p
        g = SimpleJong([Player(0), Player(1), Player(2), Player(3)])
        g._player_hands[0] = [Tile(Suit.PINZU, TileType.THREE)] + [Tile(Suit.SOUZU, TileType.ONE)] * 10
        # Prepare player 1 and 2 so both can ron on 3p (11 tiles each)
        base_s = [
            Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.TWO), Tile(Suit.SOUZU, TileType.THREE),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE), Tile(Suit.SOUZU, TileType.SIX),
            Tile(Suit.SOUZU, TileType.SEVEN), Tile(Suit.SOUZU, TileType.EIGHT), Tile(Suit.SOUZU, TileType.NINE),
        ]
        g._player_hands[1] = base_s + [Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR)]
        g._player_hands[2] = base_s + [Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR)]
        g.tiles = []
        g.current_player_idx = 0
        # Discard 3p by player 0
        g.step(0, Discard(Tile(Suit.PINZU, TileType.THREE)))
        # Resolve reactions; both 1 and 2 should win, loser is 0
        ended = g._resolve_reactions_after_discard()
        self.assertTrue(ended)
        self.assertTrue(g.is_game_over())
        winners = set(g.get_winners())
        self.assertEqual(winners, {1, 2})
        self.assertEqual(g.get_loser(), 0)

    def test_illegal_chi_by_non_left_player(self):
        game = SimpleJong([Player(0), Player(1), Player(2), Player(3)])
        # Player 0 discards 3p, left is player 1
        game._player_hands[0] = [Tile(Suit.PINZU, TileType.THREE)] + [Tile(Suit.SOUZU, TileType.ONE)] * 10
        game._player_hands[2] = [Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR)] + [Tile(Suit.SOUZU, TileType.ONE)] * 9
        game.tiles = []
        game.current_player_idx = 0
        self.assertTrue(game.step(0, Discard(Tile(Suit.PINZU, TileType.THREE))))
        # Player 2 is not left of discarder; chi should be illegal
        self.assertFalse(game.is_legal(2, Chi([Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR)])))
        with self.assertRaises(SimpleJong.IllegalMoveException):
            game.step(2, Chi([Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR)]))

    def test_legal_chi_by_left_player(self):
        game = SimpleJong([Player(0), Player(1), Player(2), Player(3)])
        game._player_hands[0] = [Tile(Suit.PINZU, TileType.THREE)] + [Tile(Suit.SOUZU, TileType.ONE)] * 10
        # Ensure left player can chi but cannot ron by using non-partitionable souzu filler
        non_partitionable_souzu = [
            Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.TWO),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE),
            Tile(Suit.SOUZU, TileType.SEVEN), Tile(Suit.SOUZU, TileType.SEVEN), Tile(Suit.SOUZU, TileType.EIGHT), Tile(Suit.SOUZU, TileType.NINE)
        ]
        game._player_hands[1] = [Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR)] + non_partitionable_souzu
        game.tiles = []
        game.current_player_idx = 0
        self.assertTrue(game.step(0, Discard(Tile(Suit.PINZU, TileType.THREE))))
        self.assertTrue(game.is_legal(1, Chi([Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR)])))
        applied = game.step(1, Chi([Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR)]))
        self.assertTrue(applied)
        self.assertIsNone(game.last_discarded_tile)
        self.assertEqual(game.current_player_idx, 1)

    def test_legal_pon_by_any_player(self):
        game = SimpleJong([Player(0), Player(1), Player(2), Player(3)])
        # Player 0 discards 5s, player 2 holds two 5s
        game._player_hands[0] = [Tile(Suit.SOUZU, TileType.FIVE)] + [Tile(Suit.SOUZU, TileType.ONE)] * 10
        # Ensure player 2 can pon but cannot ron by using non-partitionable souzu filler
        non_partitionable_souzu = [
            Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.TWO),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.SIX),
            Tile(Suit.SOUZU, TileType.SEVEN), Tile(Suit.SOUZU, TileType.SEVEN), Tile(Suit.SOUZU, TileType.EIGHT), Tile(Suit.SOUZU, TileType.NINE)
        ]
        game._player_hands[2] = [Tile(Suit.SOUZU, TileType.FIVE), Tile(Suit.SOUZU, TileType.FIVE)] + non_partitionable_souzu
        game.tiles = []
        game.current_player_idx = 0
        self.assertTrue(game.step(0, Discard(Tile(Suit.SOUZU, TileType.FIVE))))
        self.assertTrue(game.is_legal(2, Pon([Tile(Suit.SOUZU, TileType.FIVE), Tile(Suit.SOUZU, TileType.FIVE)])))
        applied = game.step(2, Pon([Tile(Suit.SOUZU, TileType.FIVE), Tile(Suit.SOUZU, TileType.FIVE)]))
        self.assertTrue(applied)
        # Both 5s should be consumed from player 2's hand
        remaining_fives = [t for t in game.hand(2) if t.suit == Suit.SOUZU and t.tile_type == TileType.FIVE]
        self.assertEqual(len(remaining_fives), 0)
        self.assertIsNone(game.last_discarded_tile)
        self.assertEqual(game.current_player_idx, 2)

    def test_illegal_fourth_chi_when_ron_available(self):
        g = SimpleJong([Player(0), Player(1), Player(2), Player(3)])
        # Give player 1 three called sets (111p, 222p, 333s)
        cs1 = CalledSet(tiles=[Tile(Suit.PINZU, TileType.ONE)]*3, call_type='pon', called_tile=Tile(Suit.PINZU, TileType.ONE), caller_position=1, source_position=0)
        cs2 = CalledSet(tiles=[Tile(Suit.PINZU, TileType.TWO)]*3, call_type='pon', called_tile=Tile(Suit.PINZU, TileType.TWO), caller_position=1, source_position=0)
        cs3 = CalledSet(tiles=[Tile(Suit.SOUZU, TileType.THREE)]*3, call_type='pon', called_tile=Tile(Suit.SOUZU, TileType.THREE), caller_position=1, source_position=0)
        g._player_called_sets[1] = [cs1, cs2, cs3]
        # Player 1 concealed hand has two tiles that could Chi with 3p (2p and 4p)
        g._player_hands[1] = [Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR)]
        # Player 0 discards 3p; player 1 is left of discarder
        g._player_hands[0] = [Tile(Suit.PINZU, TileType.THREE)] + [Tile(Suit.SOUZU, TileType.ONE)]*10
        g.tiles = []
        g.current_player_idx = 0
        self.assertTrue(g.step(0, Discard(Tile(Suit.PINZU, TileType.THREE))))
        # Legal moves for player 1 should include Ron; Chi should not be legal when Ron is available
        moves_p1 = g.legal_moves(1)
        self.assertTrue(any(isinstance(m, Ron) for m in moves_p1))
        self.assertFalse(any(isinstance(m, Chi) for m in moves_p1))

    def test_legal_moves_action_phase_for_current_player(self):
        # At start, current player is 0 with 11 tiles; tsumo should be absent, discards present
        moves = self.game.legal_moves(0)
        discard_moves = [m for m in moves if isinstance(m, Discard)]
        tsumo_moves = [m for m in moves if isinstance(m, Tsumo)]
        self.assertEqual(len(discard_moves), len(self.game.hand(0)))
        self.assertEqual(len(tsumo_moves), 0)

    def test_legal_moves_action_phase_others_have_none(self):
        # Non-current players should have no legal moves during action phase
        for pid in [1, 2, 3]:
            self.assertEqual(self.game.legal_moves(pid), [])

    def test_legal_moves_reaction_phase_includes_ron(self):
        # Setup: player 0 discards 3p; player 1 can ron on it
        game = SimpleJong([Player(0), Player(1), Player(2), Player(3)])
        game._player_hands[0] = [Tile(Suit.PINZU, TileType.THREE)] + [Tile(Suit.SOUZU, TileType.ONE)] * 10
        base_s = [
            Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.TWO), Tile(Suit.SOUZU, TileType.THREE),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE), Tile(Suit.SOUZU, TileType.SIX),
            Tile(Suit.SOUZU, TileType.SEVEN), Tile(Suit.SOUZU, TileType.EIGHT), Tile(Suit.SOUZU, TileType.NINE),
        ]
        game._player_hands[1] = base_s + [Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR)]
        game.tiles = []
        game.current_player_idx = 0
        self.assertTrue(game.step(0, Discard(Tile(Suit.PINZU, TileType.THREE))))
        moves_p1 = game.legal_moves(1)
        self.assertTrue(any(isinstance(m, Ron) for m in moves_p1))
        # At least two options: Ron and PassCall
        self.assertGreaterEqual(len(moves_p1), 2)
        self.assertTrue(any(isinstance(m, PassCall) for m in moves_p1))
        # Discarder cannot react
        self.assertEqual(game.legal_moves(0), [])

    def test_legal_moves_reaction_phase_chi_left_only(self):
        game = SimpleJong([Player(0), Player(1), Player(2), Player(3)])
        game._player_hands[0] = [Tile(Suit.PINZU, TileType.THREE)] + [Tile(Suit.SOUZU, TileType.ONE)] * 10
        # Use souzu filler that cannot be partitioned into three melds to avoid accidental Ron
        non_partitionable_souzu = [
            Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.TWO),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE),
            Tile(Suit.SOUZU, TileType.SEVEN), Tile(Suit.SOUZU, TileType.SEVEN), Tile(Suit.SOUZU, TileType.EIGHT), Tile(Suit.SOUZU, TileType.NINE)
        ]
        game._player_hands[1] = [Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR)] + non_partitionable_souzu
        game.tiles = []
        game.current_player_idx = 0
        self.assertTrue(game.step(0, Discard(Tile(Suit.PINZU, TileType.THREE))))
        moves_left = game.legal_moves(1)
        self.assertTrue(any(isinstance(m, Chi) and len(m.tiles) == 2 for m in moves_left))
        # PassCall should always be available in reaction phase
        self.assertTrue(any(isinstance(m, PassCall) for m in moves_left))
        self.assertGreaterEqual(len(moves_left), 2)
        moves_not_left = game.legal_moves(2)
        self.assertFalse(any(isinstance(m, Chi) for m in moves_not_left))

    def test_legal_moves_reaction_phase_pon_any_player(self):
        game = SimpleJong([Player(0), Player(1), Player(2), Player(3)])
        game._player_hands[0] = [Tile(Suit.SOUZU, TileType.FIVE)] + [Tile(Suit.SOUZU, TileType.ONE)] * 10
        # Ensure player 3 is not in Ron state on 5s; filler cannot form three melds by itself
        non_partitionable_souzu = [
            Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.TWO),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.SIX),
            Tile(Suit.SOUZU, TileType.SEVEN), Tile(Suit.SOUZU, TileType.SEVEN), Tile(Suit.SOUZU, TileType.EIGHT), Tile(Suit.SOUZU, TileType.NINE)
        ]
        game._player_hands[3] = [Tile(Suit.SOUZU, TileType.FIVE), Tile(Suit.SOUZU, TileType.FIVE)] + non_partitionable_souzu
        game.tiles = []
        game.current_player_idx = 0
        self.assertTrue(game.step(0, Discard(Tile(Suit.SOUZU, TileType.FIVE))))
        moves = game.legal_moves(3)
        self.assertTrue(any(isinstance(m, Pon) and len(m.tiles) == 2 for m in moves))
        self.assertTrue(any(isinstance(m, PassCall) for m in moves))
        self.assertGreaterEqual(len(moves), 2)

    def test_simultaneous_legal_reactions(self):
        # Player 0 discards 3p; player 1 can Ron, player 2 can Pon. Both should see legal Ron/Pon and Pass.
        g = SimpleJong([Player(0), Player(1), Player(2), Player(3)])
        # Discarder (0) has 3p to discard
        g._player_hands[0] = [Tile(Suit.PINZU, TileType.THREE)] + [Tile(Suit.SOUZU, TileType.ONE)] * 10
        # Player 1 has a structural 11 that Rons on 3p (2p,4p present)
        base_s = [
            Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.TWO), Tile(Suit.SOUZU, TileType.THREE),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE), Tile(Suit.SOUZU, TileType.SIX),
            Tile(Suit.SOUZU, TileType.SEVEN), Tile(Suit.SOUZU, TileType.EIGHT), Tile(Suit.SOUZU, TileType.NINE),
        ]
        g._player_hands[1] = base_s + [Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR)]
        # Player 2 can Pon on 3p but cannot Ron (use non-partitionable souzu filler)
        non_partitionable_souzu = [
            Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.TWO),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE),
            Tile(Suit.SOUZU, TileType.SEVEN), Tile(Suit.SOUZU, TileType.SEVEN), Tile(Suit.SOUZU, TileType.EIGHT), Tile(Suit.SOUZU, TileType.NINE)
        ]
        g._player_hands[2] = [Tile(Suit.PINZU, TileType.THREE), Tile(Suit.PINZU, TileType.THREE)] + non_partitionable_souzu
        g.tiles = []
        g.current_player_idx = 0
        # Discard 3p
        self.assertTrue(g.step(0, Discard(Tile(Suit.PINZU, TileType.THREE))))
        # Player 1 legal moves include Ron and Pass
        moves_p1 = g.legal_moves(1)
        self.assertTrue(any(isinstance(m, Ron) for m in moves_p1))
        self.assertTrue(any(isinstance(m, PassCall) for m in moves_p1))
        # Player 2 legal moves include Pon and Pass (even though Ron priority will preempt Pon execution)
        moves_p2 = g.legal_moves(2)
        self.assertTrue(any(isinstance(m, Pon) for m in moves_p2))
        self.assertTrue(any(isinstance(m, PassCall) for m in moves_p2))

    def test_reaction_phase_always_has_at_least_two_options(self):
        game = SimpleJong([Player(0), Player(1), Player(2), Player(3)])
        # Force a discard to enter reaction phase
        tile = game.hand(0)[0]
        game.step(0, Discard(tile))
        for pid in [1, 2, 3]:
            moves = game.legal_moves(pid)
            # With new logic: if there is any legal reaction (Ron/Pon/Chi), Pass must also be present.
            if any(isinstance(m, (Ron, Pon, Chi)) for m in moves):
                self.assertTrue(any(isinstance(m, PassCall) for m in moves))
                if any(isinstance(m, Ron) for m in moves):
                    self.assertGreaterEqual(len(moves), 2)
            else:
                # No legal reactions: player has no moves
                self.assertEqual(len(moves), 0)

    def test_legality_mask_matches_legal_moves(self):
        # Construct a small scenario with a pending discard and check legality_mask
        game = SimpleJong([Player(0), Player(1), Player(2), Player(3)])
        game._player_hands[0] = [Tile(Suit.SOUZU, TileType.FIVE)] + [Tile(Suit.SOUZU, TileType.ONE)] * 10
        game._player_hands[1] = [Tile(Suit.SOUZU, TileType.FIVE), Tile(Suit.SOUZU, TileType.FIVE)] + [Tile(Suit.SOUZU, TileType.ONE)] * 9
        game.tiles = []
        game.current_player_idx = 0
        self.assertTrue(game.step(0, Discard(Tile(Suit.SOUZU, TileType.FIVE))))

        # For actor 1, Pon should be legal
        moves = game.legal_moves(1)
        from core.learn.pure_policy_dataset import get_num_actions, serialize_action, encode_action_flat_index  # type: ignore
        mask = game.legality_mask(1)
        self.assertEqual(len(mask), get_num_actions())
        gs = game.get_game_perspective(1)
        ldt = str(gs.last_discarded_tile) if gs.last_discarded_tile is not None else None
        # Every legal move must map to a True in the mask
        for m in moves:
            ad = serialize_action(m)
            idx = encode_action_flat_index(ad, ldt)
            self.assertTrue(mask[int(idx)], f"Expected mask[{idx}] True for move {type(m).__name__}")


if __name__ == '__main__':
    unittest.main(verbosity=2)


