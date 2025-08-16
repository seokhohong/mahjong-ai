import unittest
import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from medium_core.game import (
    MediumJong, Player, Tile, TileType, Suit, Honor,
    Discard, Tsumo, Ron, Pon, Chi, CalledSet, PassCall, Riichi
)


class TestMediumLegality(unittest.TestCase):
    def setUp(self):
        self.players = [Player(i) for i in range(4)]
        self.game = MediumJong(self.players)

    def test_illegal_ron_without_discard(self):
        self.assertFalse(self.game.is_legal(1, Ron()))
        with self.assertRaises(MediumJong.IllegalMoveException):
            self.game.step(1, Ron())
        self.assertFalse(self.game.is_game_over())

    def test_illegal_discard_by_non_current_player(self):
        tile = self.game.hand(1)[0]
        self.assertFalse(self.game.is_legal(1, Discard(tile)))
        with self.assertRaises(MediumJong.IllegalMoveException):
            self.game.step(1, Discard(tile))
        self.assertIsNone(self.game.last_discarded_tile)

    def test_illegal_discard_tile_not_in_hand(self):
        current_hand = self.game.hand(0)
        candidate = Tile(Suit.PINZU, TileType.ONE)
        if candidate in current_hand:
            candidate = Tile(Suit.SOUZU, TileType.NINE)
            if candidate in current_hand:
                # Find any tile not in hand
                all_tiles = [Tile(s, TileType(v)) for s in (Suit.PINZU, Suit.SOUZU, Suit.MANZU) for v in range(1, 10)]
                for t in all_tiles:
                    if t not in current_hand:
                        candidate = t
                        break
        self.assertFalse(self.game.is_legal(0, Discard(candidate)))
        with self.assertRaises(MediumJong.IllegalMoveException):
            self.game.step(0, Discard(candidate))
        self.assertIsNone(self.game.last_discarded_tile)

    def test_legal_discard_by_current_player(self):
        tile = self.game.hand(0)[0]
        self.assertTrue(self.game.is_legal(0, Discard(tile)))
        applied = self.game.step(0, Discard(tile))
        self.assertTrue(applied)
        self.assertEqual(self.game.last_discarded_tile, tile)
        self.assertEqual(self.game.last_discard_player, 0)

    def test_double_ron_on_same_discard(self):
        g = MediumJong([Player(0), Player(1), Player(2), Player(3)])
        # Discarder has 3p
        g._player_hands[0] = [Tile(Suit.PINZU, TileType.THREE)] + g._player_hands[0][1:]
        # Players 1 and 2 can ron on 3p: base 9 souzu + pair 7m7m + 2p,4p (13 tiles)
        base_s = [
            Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.TWO), Tile(Suit.SOUZU, TileType.THREE),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE), Tile(Suit.SOUZU, TileType.SIX),
            Tile(Suit.SOUZU, TileType.SEVEN), Tile(Suit.SOUZU, TileType.EIGHT), Tile(Suit.SOUZU, TileType.NINE),
        ]
        pair = [Tile(Suit.MANZU, TileType.SEVEN), Tile(Suit.MANZU, TileType.SEVEN)]
        g._player_hands[1] = base_s + pair + [Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR)]
        g._player_hands[2] = base_s + pair + [Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR)]
        g.tiles = []
        g.current_player_idx = 0
        self.assertTrue(g.step(0, Discard(Tile(Suit.PINZU, TileType.THREE))))
        # Let engine resolve reactions: both should ron
        g._resolve_reactions()
        self.assertTrue(g.is_game_over())
        winners = set(g.get_winners())
        self.assertEqual(winners, {1, 2})
        self.assertEqual(g.get_loser(), 0)

    def test_illegal_chi_by_non_left_player(self):
        game = MediumJong([Player(0), Player(1), Player(2), Player(3)])
        game._player_hands[0] = [Tile(Suit.PINZU, TileType.THREE)] + game._player_hands[0][1:]
        # Player 2 has 2p and 4p
        game._player_hands[2][:2] = [Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR)]
        game.tiles = []
        game.current_player_idx = 0
        self.assertTrue(game.step(0, Discard(Tile(Suit.PINZU, TileType.THREE))))
        self.assertFalse(game.is_legal(2, Chi([Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR)])))
        with self.assertRaises(MediumJong.IllegalMoveException):
            game.step(2, Chi([Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR)]))

    def test_chi_not_legal_when_ron_available(self):
        g = MediumJong([Player(0), Player(1), Player(2), Player(3)])
        # Player 1 has called sets to limit hand size and 2p,4p with a ron-ready structure
        cs1 = CalledSet(tiles=[Tile(Suit.PINZU, TileType.ONE)]*3, call_type='pon', called_tile=Tile(Suit.PINZU, TileType.ONE), caller_position=1, source_position=0)
        cs2 = CalledSet(tiles=[Tile(Suit.PINZU, TileType.TWO)]*3, call_type='pon', called_tile=Tile(Suit.PINZU, TileType.TWO), caller_position=1, source_position=0)
        cs3 = CalledSet(tiles=[Tile(Suit.SOUZU, TileType.THREE)]*3, call_type='pon', called_tile=Tile(Suit.SOUZU, TileType.THREE), caller_position=1, source_position=0)
        g._player_called_sets[1] = [cs1, cs2, cs3]
        g._player_hands[1] = [Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR)]
        g._player_hands[0] = [Tile(Suit.PINZU, TileType.THREE)] + g._player_hands[0][1:]
        g.tiles = []
        g.current_player_idx = 0
        self.assertTrue(g.step(0, Discard(Tile(Suit.PINZU, TileType.THREE))))
        moves_p1 = g.legal_moves(1)
        self.assertTrue(any(isinstance(m, Ron) for m in moves_p1))
        self.assertTrue(any(isinstance(m, PassCall) for m in moves_p1))
        self.assertFalse(any(isinstance(m, Chi) for m in moves_p1))

    def test_legal_chi_by_left_player(self):
        game = MediumJong([Player(0), Player(1), Player(2), Player(3)])
        game._player_hands[0] = [Tile(Suit.PINZU, TileType.THREE)] + game._player_hands[0][1:]
        non_part = [
            Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.TWO),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE),
            Tile(Suit.SOUZU, TileType.SEVEN), Tile(Suit.SOUZU, TileType.SEVEN), Tile(Suit.SOUZU, TileType.EIGHT), Tile(Suit.SOUZU, TileType.NINE)
        ]
        game._player_hands[1] = [Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR)] + non_part
        game.tiles = []
        game.current_player_idx = 0
        self.assertTrue(game.step(0, Discard(Tile(Suit.PINZU, TileType.THREE))))
        self.assertTrue(game.is_legal(1, Chi([Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR)])))
        self.assertTrue(game.step(1, Chi([Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR)])))
        self.assertIsNone(game.last_discarded_tile)
        self.assertEqual(game.current_player_idx, 1)

    def test_legal_pon_by_any_player(self):
        game = MediumJong([Player(0), Player(1), Player(2), Player(3)])
        game._player_hands[0] = [Tile(Suit.SOUZU, TileType.FIVE)] + game._player_hands[0][1:]
        non_part = [
            Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.TWO),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.SIX),
            Tile(Suit.SOUZU, TileType.SEVEN), Tile(Suit.SOUZU, TileType.SEVEN), Tile(Suit.SOUZU, TileType.EIGHT), Tile(Suit.SOUZU, TileType.NINE)
        ]
        game._player_hands[2] = [Tile(Suit.SOUZU, TileType.FIVE), Tile(Suit.SOUZU, TileType.FIVE)] + non_part
        game.tiles = []
        game.current_player_idx = 0
        self.assertTrue(game.step(0, Discard(Tile(Suit.SOUZU, TileType.FIVE))))
        self.assertTrue(game.is_legal(2, Pon([Tile(Suit.SOUZU, TileType.FIVE), Tile(Suit.SOUZU, TileType.FIVE)])))
        self.assertTrue(game.step(2, Pon([Tile(Suit.SOUZU, TileType.FIVE), Tile(Suit.SOUZU, TileType.FIVE)])))
        remaining_fives = [t for t in game.hand(2) if t.suit == Suit.SOUZU and t.tile_type == TileType.FIVE]
        self.assertEqual(len(remaining_fives), 0)
        self.assertIsNone(game.last_discarded_tile)
        self.assertEqual(game.current_player_idx, 2)

    def test_legal_moves_action_phase_for_current_player(self):
        moves = self.game.legal_moves(0)
        discard_moves = [m for m in moves if isinstance(m, Discard)]
        tsumo_moves = [m for m in moves if isinstance(m, Tsumo)]
        self.assertEqual(len(discard_moves), len(self.game.hand(0)))
        self.assertEqual(len(tsumo_moves), 0)

    def test_legal_moves_action_phase_others_have_none(self):
        for pid in [1, 2, 3]:
            self.assertEqual(self.game.legal_moves(pid), [])

    def test_legal_moves_reaction_phase_includes_ron_and_pass(self):
        game = MediumJong([Player(0), Player(1), Player(2), Player(3)])
        game._player_hands[0] = [Tile(Suit.PINZU, TileType.THREE)] + game._player_hands[0][1:]
        base_s = [
            Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.TWO), Tile(Suit.SOUZU, TileType.THREE),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE), Tile(Suit.SOUZU, TileType.SIX),
            Tile(Suit.SOUZU, TileType.SEVEN), Tile(Suit.SOUZU, TileType.EIGHT), Tile(Suit.SOUZU, TileType.NINE),
        ]
        # Ensure player 1 has 13 tiles pre-ron: add a pair
        pair = [Tile(Suit.MANZU, TileType.SEVEN), Tile(Suit.MANZU, TileType.SEVEN)]
        game._player_hands[1] = base_s + pair + [Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR)]
        game.tiles = []
        game.current_player_idx = 0
        self.assertTrue(game.step(0, Discard(Tile(Suit.PINZU, TileType.THREE))))
        moves_p1 = game.legal_moves(1)
        self.assertTrue(any(isinstance(m, Ron) for m in moves_p1))
        self.assertTrue(any(isinstance(m, PassCall) for m in moves_p1))
        self.assertEqual(game.legal_moves(0), [])

    def test_riichi_locks_discards_to_newly_drawn(self):
        g = MediumJong([Player(0), Player(1), Player(2), Player(3)])
        # Player 0 closed tenpai: 234m, 345p, 678m, pair 77p, wait 4-5s on 6s
        p0 = [
            Tile(Suit.MANZU, TileType.TWO), Tile(Suit.MANZU, TileType.THREE), Tile(Suit.MANZU, TileType.FOUR),
            Tile(Suit.PINZU, TileType.THREE), Tile(Suit.PINZU, TileType.FOUR), Tile(Suit.PINZU, TileType.FIVE),
            Tile(Suit.MANZU, TileType.SIX), Tile(Suit.MANZU, TileType.SEVEN), Tile(Suit.MANZU, TileType.EIGHT),
            Tile(Suit.PINZU, TileType.SEVEN), Tile(Suit.PINZU, TileType.SEVEN),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE),
        ]
        g._player_hands[0] = p0
        g.current_player_idx = 0
        g.last_discarded_tile = None
        # Prepare a deterministic wall: first draw for P0 (non-winning), then draws for P1..P3, then next draw for P0 (non-winning)
        first_draw = Tile(Suit.MANZU, TileType.ONE)
        next_draw = Tile(Suit.MANZU, TileType.NINE)
        g.tiles = [next_draw, Tile(Suit.PINZU, TileType.ONE), Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.MANZU, TileType.TWO), first_draw]
        # Player 0 declares riichi
        # Two different safe riichi tiles (both should keep tenpai)
        lm_before = g.legal_moves(0)
        riichi_moves = [m for m in lm_before if isinstance(m, Riichi)]
        # Ensure parameterized riichi exists for multiple tiles
        self.assertGreaterEqual(len(riichi_moves), 1)
        # Choose any one riichi to proceed
        self.assertTrue(g.step(0, riichi_moves[0]))
        # Draw for Player 0 and verify only discard of newly drawn tile is legal
        g._draw_for_current_if_needed()
        self.assertIsNotNone(g.last_drawn_tile)
        lm0 = g.legal_moves(0)
        discards = [m for m in lm0 if isinstance(m, Discard)]
        self.assertEqual(len(discards), 1)
        self.assertEqual(discards[0].tile, g.last_drawn_tile)
        # Discard it
        self.assertTrue(g.step(0, discards[0]))
        # Resolve reactions (none) and advance through players 1..3 with safe discards
        g._resolve_reactions()
        g.current_player_idx = 1
        for pid in [1, 2, 3]:
            g._draw_for_current_if_needed()
            safe_tile = Tile(Suit.HONORS, Honor.EAST)
            g._player_hands[pid][0] = safe_tile
            self.assertTrue(g.step(pid, Discard(safe_tile)))
            g._resolve_reactions()
            g.current_player_idx = (pid + 1) % 4
        # Back to player 0: draw a non-winning tile and verify only that discard is legal
        self.assertEqual(g.current_player_idx, 0)
        g._draw_for_current_if_needed()
        self.assertIsNotNone(g.last_drawn_tile)
        lm0b = g.legal_moves(0)
        self.assertTrue(all(isinstance(m, (Discard,)) for m in lm0b if not isinstance(m, Tsumo)))
        discards2 = [m for m in lm0b if isinstance(m, Discard)]
        self.assertEqual(len(discards2), 1)
        self.assertEqual(discards2[0].tile, g.last_drawn_tile)

    def test_riichi_multiple_discard_options_in_tenpai(self):
        # Construct a hand where discarding 2m or 8m both keep tenpai
        g = MediumJong([Player(0), Player(1), Player(2), Player(3)])
        hand = [
            Tile(Suit.MANZU, TileType.TWO),  # candidate A
            Tile(Suit.MANZU, TileType.THREE), Tile(Suit.MANZU, TileType.FOUR),  # 23(4)
            Tile(Suit.PINZU, TileType.THREE), Tile(Suit.PINZU, TileType.FOUR), Tile(Suit.PINZU, TileType.FIVE),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE), Tile(Suit.SOUZU, TileType.SIX),
            Tile(Suit.MANZU, TileType.SIX), Tile(Suit.MANZU, TileType.SEVEN), Tile(Suit.MANZU, TileType.EIGHT),  # 678m; candidate B is 8m pair builder
            Tile(Suit.PINZU, TileType.SEVEN),
        ]
        g._player_hands[0] = hand
        g.current_player_idx = 0
        # List riichi options
        lm = g.legal_moves(0)
        riichi_moves = [m for m in lm if isinstance(m, Riichi)]
        # Expect at least two distinct riichi tiles among options
        riichi_tiles = {(m.tile.suit.value, int(m.tile.tile_type.value)) for m in riichi_moves}
        self.assertGreaterEqual(len(riichi_tiles), 2)

    def test_furiten_blocks_ron_but_allows_tsumo(self):
        # Player 0 has a hand waiting on 3p; they have previously discarded 3p -> furiten
        g = MediumJong([Player(0), Player(1), Player(2), Player(3)])
        base_s = [
            Tile(Suit.SOUZU, TileType.ONE), Tile(Suit.SOUZU, TileType.TWO), Tile(Suit.SOUZU, TileType.THREE),
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE), Tile(Suit.SOUZU, TileType.SIX),
            Tile(Suit.SOUZU, TileType.SEVEN), Tile(Suit.SOUZU, TileType.EIGHT), Tile(Suit.SOUZU, TileType.NINE),
        ]
        hand = base_s + [Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR), Tile(Suit.MANZU, TileType.SEVEN), Tile(Suit.MANZU, TileType.SEVEN)]
        g._player_hands[0] = hand
        # Discard history: player 0 had discarded 3p earlier
        g.player_discards[0] = [Tile(Suit.PINZU, TileType.THREE)]
        # Player 1 discards 3p which would complete player 0's hand
        g.current_player_idx = 1
        g.last_discarded_tile = None
        g.last_discard_player = None
        g._player_hands[1][0] = Tile(Suit.PINZU, TileType.THREE)
        g.tiles = []
        self.assertTrue(g.step(1, Discard(Tile(Suit.PINZU, TileType.THREE))))
        # Player 0 is furiten, so ron must be illegal
        self.assertFalse(g.is_legal(0, Ron()))
        # Clear reaction and let player 0 draw the winning tile to tsumo
        g._resolve_reactions()
        g.current_player_idx = 0
        g.tiles = [Tile(Suit.PINZU, TileType.THREE)]
        g.play_turn()
        self.assertTrue(g.is_game_over())
        self.assertEqual(g.get_winners(), [0])

    def test_riichi_allows_ankan_on_drawn_fourth_tile(self):
        g = MediumJong([Player(0), Player(1), Player(2), Player(3)])
        # Player 0 closed tenpai with three 5m in hand; waits on 6s to complete 456s
        p0 = [
            Tile(Suit.MANZU, TileType.TWO), Tile(Suit.MANZU, TileType.THREE), Tile(Suit.MANZU, TileType.FOUR),  # 234m
            Tile(Suit.PINZU, TileType.THREE), Tile(Suit.PINZU, TileType.FOUR), Tile(Suit.PINZU, TileType.FIVE),  # 345p
            Tile(Suit.MANZU, TileType.FIVE), Tile(Suit.MANZU, TileType.FIVE), Tile(Suit.MANZU, TileType.FIVE),  # 555m (concealed triplet)
            Tile(Suit.PINZU, TileType.SEVEN), Tile(Suit.PINZU, TileType.SEVEN),  # pair 77p
            Tile(Suit.SOUZU, TileType.FOUR), Tile(Suit.SOUZU, TileType.FIVE),   # 45s wait on 6s
        ]
        g._player_hands[0] = p0
        g.current_player_idx = 0
        g.last_discarded_tile = None
        # Wall: draw the fourth 5m to enable ankan after riichi
        draw_fourth_5m = Tile(Suit.MANZU, TileType.FIVE)
        g.tiles = [draw_fourth_5m]
        # Declare riichi
        self.assertTrue(g.is_legal(0, Riichi()))
        self.assertTrue(g.step(0, Riichi()))
        # Draw the fourth 5m
        g._draw_for_current_if_needed()
        self.assertEqual(g.last_drawn_tile, draw_fourth_5m)
        # Legal moves should include KanAnkan on 5m and only discard of the drawn tile
        lm = g.legal_moves(0)
        from medium_core.game import KanAnkan  # local import to reference class
        self.assertTrue(any(isinstance(m, KanAnkan) and m.tile == draw_fourth_5m for m in lm))
        discards = [m for m in lm if isinstance(m, Discard)]
        self.assertEqual(len(discards), 1)
        self.assertEqual(discards[0].tile, draw_fourth_5m)
        # Perform the ankan
        kan_move = next(m for m in lm if isinstance(m, KanAnkan))
        self.assertTrue(g.step(0, kan_move))
        # Called set recorded and rinshan draw occurred
        csets = g.called_sets(0)
        self.assertTrue(any(cs.call_type == 'kan_ankan' for cs in csets))
        self.assertIsNotNone(g.last_drawn_tile)
        # Now simulate other players discarding tiles that would allow Chi/Pon, and ensure those are illegal due to Riichi
        # Make player 1 discard 2p (chi candidate with 3p,4p), and player 2 discard 5m (pon candidate with remaining 5m after kan)
        # Clear pending discard then step each player
        g.last_discarded_tile = None
        g.last_discard_player = None
        g.current_player_idx = 1
        # Ensure player 0 has 2p and 4p in hand to present a chi option if allowed
        if not any(t.suit == Suit.PINZU and t.tile_type == TileType.TWO for t in g.hand(0)):
            g._player_hands[0].append(Tile(Suit.PINZU, TileType.TWO))
        if not any(t.suit == Suit.PINZU and t.tile_type == TileType.FOUR for t in g.hand(0)):
            g._player_hands[0].append(Tile(Suit.PINZU, TileType.FOUR))
        g._player_hands[1][0] = Tile(Suit.PINZU, TileType.THREE)
        self.assertTrue(g.step(1, Discard(Tile(Suit.PINZU, TileType.THREE))))
        # As player 0 is in riichi, Chi should be illegal in reaction phase
        chi_try = Chi([Tile(Suit.PINZU, TileType.TWO), Tile(Suit.PINZU, TileType.FOUR)])
        self.assertFalse(g.is_legal(0, chi_try))
        # Pass reaction
        g._resolve_reactions()
        g.current_player_idx = 2
        # Provide pon candidate: ensure player 0 has two 5m left only if available; after kan, no 5m in hand.
        # We'll use 3p pon scenario instead.
        g._player_hands[2][0] = Tile(Suit.PINZU, TileType.THREE)
        self.assertTrue(g.step(2, Discard(Tile(Suit.PINZU, TileType.THREE))))
        pon_try = Pon([Tile(Suit.PINZU, TileType.THREE), Tile(Suit.PINZU, TileType.THREE)])
        self.assertFalse(g.is_legal(0, pon_try))


if __name__ == '__main__':
    unittest.main(verbosity=2)



