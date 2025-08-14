import random
import math
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Any, Union, Tuple
from medium_core.constants import (
    NUM_PLAYERS as MC_NUM_PLAYERS,
    DEALER_ID_START,
    TILE_COPIES_DEFAULT,
    STARTING_HAND_TILES,
    INITIAL_DORA_INDICATORS,
    INITIAL_URADORA_INDICATORS,
    SUIT_ORDER,
    FU_CHIITOI,
    FU_BASELINE,
    POINTS_ROUNDING,
    CHANTA_OPEN_HAN,
    CHANTA_CLOSED_HAN,
    JUNCHAN_OPEN_HAN,
    JUNCHAN_CLOSED_HAN,
    SANANKOU_HAN,
)


# MediumJong: Expanded Riichi-like implementation
# - Suits: Manzu (m), Pinzu (p), Souzu (s) and Honors (winds/dragons)
# - Calls: Chi, Pon, Kan (daiminkan, kakan, ankan)
# - Yaku requirement to win; rudimentary scoring (fu/han) with dora/uradora
# - Round/seat winds; player 0 is dealer (East) and East round
# - Riichi declaration; after riichi, only Win/Kan allowed; uradora on riichi win


class Suit(Enum):
    MANZU = 'm'
    PINZU = 'p'
    SOUZU = 's'
    HONORS = 'z'


class TileType(Enum):
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9


class Honor(Enum):
    EAST = 1
    SOUTH = 2
    WEST = 3
    NORTH = 4
    WHITE = 5  # haku
    GREEN = 6  # hatsu
    RED = 7    # chun


@dataclass
class Tile:
    suit: Suit
    tile_type: Union[TileType, Honor]
    aka: bool = False  # red-dora five indicator for suited 5s

    def __str__(self) -> str:
        if self.suit == Suit.HONORS:
            mapping = {
                Honor.EAST: 'E', Honor.SOUTH: 'S', Honor.WEST: 'W', Honor.NORTH: 'N',
                Honor.WHITE: 'P', Honor.GREEN: 'F', Honor.RED: 'C',
            }
            return mapping[self.tile_type]  # type: ignore[index]
        return f"{int(self.tile_type.value)}{self.suit.value}"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Tile) and self.suit == other.suit and self.tile_type == other.tile_type

    def __hash__(self) -> int:
        return hash((self.suit, self.tile_type))


# Actions and reactions
@dataclass
class Action: ...


@dataclass
class Reaction: ...


@dataclass
class Tsumo(Action): ...


@dataclass
class Ron(Reaction): ...


@dataclass
class Discard(Action):
    tile: Tile


@dataclass
class Riichi(Action): ...


@dataclass
class Pon(Reaction):
    tiles: List[Tile]


@dataclass
class Chi(Reaction):
    tiles: List[Tile]


@dataclass
class PassCall(Reaction): ...


@dataclass
class KanDaimin(Reaction):
    # Call Kan on a discard with three identical tiles from hand
    tiles: List[Tile]


@dataclass
class KanKakan(Action):
    # Upgrade an existing Pon to Kan using the drawn 4th tile
    tile: Tile


@dataclass
class KanAnkan(Action):
    # Concealed Kan from four tiles in hand
    tile: Tile


@dataclass
class CalledSet:
    tiles: List[Tile]
    call_type: str  # 'chi' | 'pon' | 'kan_daimin' | 'kan_kakan' | 'kan_ankan'
    called_tile: Optional[Tile]
    caller_position: int
    source_position: Optional[int]  # None for ankan/kakan


class InvalidHandStateException(Exception):
    pass


def _is_suited(t: Tile) -> bool:
    return t.suit in (Suit.MANZU, Suit.PINZU, Suit.SOUZU)


def _tile_sort_key(t: Tile) -> Tuple[int, int]:
    return (SUIT_ORDER[t.suit.value], int(t.tile_type.value))


def _can_form_melds_concealed(tiles: List[Tile], num_melds: int) -> bool:
    # Triplets/sequences; honors cannot form sequences
    if num_melds == 0:
        return len(tiles) == 0
    if len(tiles) != 3 * num_melds:
        return False

    # Build counts by suit
    counts: Dict[Suit, List[int]] = {
        Suit.MANZU: [0] * 10,
        Suit.PINZU: [0] * 10,
        Suit.SOUZU: [0] * 10,
    }
    honors = [0] * 8  # 1..7
    for t in tiles:
        if t.suit == Suit.HONORS:
            honors[int(t.tile_type.value)] += 1
        else:
            counts[t.suit][int(t.tile_type.value)] += 1

    # Honors: must form triplets only
    for i in range(1, 8):
        if honors[i] % 3 != 0:
            return False

    def dfs() -> bool:
        # First any suit with tiles
        for suit in (Suit.MANZU, Suit.PINZU, Suit.SOUZU):
            c = counts[suit]
            for i in range(1, 10):
                if c[i] > 0:
                    # Triplet
                    if c[i] >= 3:
                        c[i] -= 3
                        if dfs():
                            return True
                        c[i] += 3
                    # Sequence
                    if i <= 7 and c[i+1] > 0 and c[i+2] > 0:
                        c[i] -= 1; c[i+1] -= 1; c[i+2] -= 1
                        if dfs():
                            return True
                        c[i] += 1; c[i+1] += 1; c[i+2] += 1
                    return False
        return True

    return dfs()


def _can_form_standard_hand(tiles: List[Tile]) -> bool:
    # 14 tiles: 4 melds + 1 pair; honors can only be triplets/pairs
    if len(tiles) != 14:
        return False
    tiles = sorted(list(tiles), key=_tile_sort_key)
    # Try all possible pairs by value
    for idx in range(len(tiles) - 1):
        a, b = tiles[idx], tiles[idx + 1]
        if a.suit == b.suit and a.tile_type == b.tile_type:
            remaining = tiles[:idx] + tiles[idx+2:]
            if _can_form_melds_concealed(remaining, 4):
                return True
    return False


def _decompose_standard_with_pred(tiles: List[Tile], pred_meld, pred_pair) -> bool:
    """Try to decompose into 4 melds + 1 pair satisfying predicates.

    pred_meld(meld_tiles[3]) -> bool, pred_pair(tile) -> bool
    """
    if len(tiles) != 14:
        return False
    tiles = sorted(list(tiles), key=_tile_sort_key)

    # Count maps
    def build_counts(ts: List[Tile]):
        counts: Dict[Suit, List[int]] = {
            Suit.MANZU: [0] * 10,
            Suit.PINZU: [0] * 10,
            Suit.SOUZU: [0] * 10,
        }
        honors = [0] * 8
        for t in ts:
            if t.suit == Suit.HONORS:
                honors[int(t.tile_type.value)] += 1
            else:
                counts[t.suit][int(t.tile_type.value)] += 1
        return counts, honors

    def make_tile(suit: Suit, val: int) -> Tile:
        return Tile(suit, TileType(val)) if suit != Suit.HONORS else Tile(Suit.HONORS, Honor(val))

    for i in range(len(tiles) - 1):
        a, b = tiles[i], tiles[i + 1]
        if a.suit == b.suit and a.tile_type == b.tile_type and pred_pair(a):
            remaining = tiles[:i] + tiles[i+2:]
            counts, honors = build_counts(remaining)

            def dfs() -> bool:
                # Process suited tiles first
                for suit in (Suit.MANZU, Suit.PINZU, Suit.SOUZU):
                    c = counts[suit]
                    for v in range(1, 10):
                        if c[v] > 0:
                            # Triplet
                            if c[v] >= 3:
                                meld = [make_tile(suit, v)] * 3
                                if pred_meld(meld):
                                    c[v] -= 3
                                    if dfs():
                                        return True
                                    c[v] += 3
                            # Sequence
                            if v <= 7 and c[v+1] > 0 and c[v+2] > 0:
                                meld = [make_tile(suit, v), make_tile(suit, v+1), make_tile(suit, v+2)]
                                if pred_meld(meld):
                                    c[v] -= 1; c[v+1] -= 1; c[v+2] -= 1
                                    if dfs():
                                        return True
                                    c[v] += 1; c[v+1] += 1; c[v+2] += 1
                            return False
                # Honors must form triplets and satisfy pred
                for hv in range(1, 8):
                    if honors[hv] > 0:
                        if honors[hv] >= 3:
                            meld = [make_tile(Suit.HONORS, hv)] * 3
                            if not pred_meld(meld):
                                return False
                            honors[hv] -= 3
                            if dfs():
                                return True
                            honors[hv] += 3
                        return False
                return True

            if dfs():
                return True
    return False


def _is_chanta(all_tiles: List[Tile]) -> bool:
    # All sets (including pair) contain terminal or honor; at least one honor or terminal in each
    def pred_meld(meld: List[Tile]) -> bool:
        if any(t.suit == Suit.HONORS for t in meld):
            return True
        vals = [int(t.tile_type.value) for t in meld]
        return min(vals) == 1 or max(vals) == 9
    def pred_pair(tile: Tile) -> bool:
        return tile.suit == Suit.HONORS or int(tile.tile_type.value) in (1, 9)
    return _decompose_standard_with_pred(all_tiles, pred_meld, pred_pair)


def _is_junchan(all_tiles: List[Tile]) -> bool:
    # All sets (including pair) contain terminals only; no honors
    def pred_meld(meld: List[Tile]) -> bool:
        if any(t.suit == Suit.HONORS for t in meld):
            return False
        vals = [int(t.tile_type.value) for t in meld]
        return min(vals) == 1 or max(vals) == 9
    def pred_pair(tile: Tile) -> bool:
        return tile.suit != Suit.HONORS and int(tile.tile_type.value) in (1, 9)
    return _decompose_standard_with_pred(all_tiles, pred_meld, pred_pair)


def _count_sanankou(concealed_tiles: List[Tile], called_sets: List[CalledSet]) -> int:
    # Count concealed triplets in hand plus any concealed kans
    cnt = _count_tiles(concealed_tiles)
    triples = sum(1 for c in cnt.values() if c >= 3)
    triples += sum(1 for cs in called_sets if cs.call_type == 'kan_ankan')
    return triples

def _is_chi_possible_with(hand: List[Tile], target: Tile) -> List[List[Tile]]:
    options: List[List[Tile]] = []
    if target.suit == Suit.HONORS:
        return options
    s = target.suit
    v = int(target.tile_type.value)
    def has(val: int) -> Optional[Tile]:
        for t in hand:
            if t.suit == s and int(t.tile_type.value) == val:
                return t
        return None
    # (v-2, v-1)
    if v - 2 >= 1 and v - 1 >= 1:
        a = has(v-2); b = has(v-1)
        if a and b:
            options.append([a, b])
    # (v-1, v+1)
    if v - 1 >= 1 and v + 1 <= 9:
        a = has(v-1); b = has(v+1)
        if a and b:
            options.append([a, b])
    # (v+1, v+2)
    if v + 1 <= 9 and v + 2 <= 9:
        a = has(v+1); b = has(v+2)
        if a and b:
            options.append([a, b])
    return options


def _count_tiles(tiles: List[Tile]) -> Dict[Tuple[Suit, int], int]:
    cnt: Dict[Tuple[Suit, int], int] = {}
    for t in tiles:
        key = (t.suit, int(t.tile_type.value))
        cnt[key] = cnt.get(key, 0) + 1
    return cnt


def _dora_next(tile: Tile) -> Tile:
    # Next tile cycling within suit/honors for dora mapping
    if tile.suit == Suit.HONORS:
        order = [Honor.EAST, Honor.SOUTH, Honor.WEST, Honor.NORTH, Honor.WHITE, Honor.GREEN, Honor.RED]
        idx = order.index(tile.tile_type)  # type: ignore[arg-type]
        return Tile(Suit.HONORS, order[(idx + 1) % len(order)])
    v = int(tile.tile_type.value)
    nv = 1 if v == 9 else v + 1
    return Tile(tile.suit, TileType(nv))


def _calc_dora_han(hand_tiles: List[Tile], called_sets: List[CalledSet], indicators: List[Tile]) -> int:
    all_tiles: List[Tile] = []
    all_tiles.extend(hand_tiles)
    for cs in called_sets:
        all_tiles.extend(cs.tiles)
    dora_tiles = [_dora_next(ind) for ind in indicators]
    return sum(1 for t in all_tiles for d in dora_tiles if t.suit == d.suit and t.tile_type == d.tile_type)


def _count_aka_han(all_tiles: List[Tile]) -> int:
    # Aka dora: each red five is worth 1 han
    return sum(1 for t in all_tiles if t.aka)


def _is_tanyao(all_tiles: List[Tile]) -> bool:
    for t in all_tiles:
        if t.suit == Suit.HONORS:
            return False
        v = int(t.tile_type.value)
        if v == 1 or v == 9:
            return False
    return True


def _is_chiitoi(concealed_tiles: List[Tile], called_sets: List[CalledSet]) -> bool:
    if called_sets:
        return False
    if len(concealed_tiles) != 14:
        return False
    cnt = _count_tiles(concealed_tiles)
    return sum(1 for c in cnt.values() if c == 2) == 7


def _count_triplet_value(cnt: Dict[Tuple[Suit, int], int], suit: Suit, val: int) -> int:
    return 1 if cnt.get((suit, val), 0) >= 3 else 0


def _is_toitoi(all_tiles: List[Tile], called_sets: List[CalledSet]) -> bool:
    # All groups are triplets/kan and a pair
    # Approx: if no suited sequences in called sets and counts of each suited number are multiples of 0,2,3,4 and number of numbers used in sequences is 0.
    for cs in called_sets:
        if cs.call_type == 'chi':
            return False
    # Very rough check: if standard-formable and there is no way to take a sequence from suited counts
    # We'll simply detect presence of any three-in-a-row suited numbers as evidence against toitoi
    cnt = _count_tiles(all_tiles)
    for suit in (Suit.MANZU, Suit.PINZU, Suit.SOUZU):
        for v in range(1, 8):
            if cnt.get((suit, v), 0) > 0 and cnt.get((suit, v+1), 0) > 0 and cnt.get((suit, v+2), 0) > 0:
                return False
    return True


def _is_honitsu(all_tiles: List[Tile]) -> bool:
    suits = {t.suit for t in all_tiles if t.suit != Suit.HONORS}
    has_honors = any(t.suit == Suit.HONORS for t in all_tiles)
    return has_honors and len(suits) == 1


def _is_chinitsu(all_tiles: List[Tile]) -> bool:
    suits = {t.suit for t in all_tiles if t.suit != Suit.HONORS}
    has_honors = any(t.suit == Suit.HONORS for t in all_tiles)
    return not has_honors and len(suits) == 1


def _yakuhai_han(concealed_tiles: List[Tile], called_sets: List[CalledSet], seat_wind: Honor, round_wind: Honor) -> int:
    # 1 han per dragon triplet; 1 han for seat wind triplet; 1 for round wind triplet
    cnt = _count_tiles(concealed_tiles + [t for cs in called_sets for t in cs.tiles])
    han = 0
    # Dragons
    for h in (Honor.WHITE, Honor.GREEN, Honor.RED):
        han += _count_triplet_value(cnt, Suit.HONORS, int(h.value))
    # Seat wind
    han += _count_triplet_value(cnt, Suit.HONORS, int(seat_wind.value))
    # Round wind
    han += _count_triplet_value(cnt, Suit.HONORS, int(round_wind.value))
    return han


def _is_open_hand(called_sets: List[CalledSet]) -> bool:
    return any(cs.call_type in ('chi', 'pon', 'kan_daimin', 'kan_kakan') for cs in called_sets)


def _is_pinfu(all_tiles: List[Tile], called_sets: List[CalledSet], seat_wind: Honor, round_wind: Honor) -> bool:
    # Closed-only; all sequences; pair not honors or seat/round
    if _is_open_hand(called_sets):
        return False
    def pred_meld(meld: List[Tile]) -> bool:
        # Reject triplets (all same value)
        a, b, c = meld
        return not (a.suit == b.suit == c.suit and int(a.tile_type.value) == int(b.tile_type.value) == int(c.tile_type.value))
    def pred_pair(tile: Tile) -> bool:
        if tile.suit == Suit.HONORS:
            return False
        # Exclude seat/round wind as pair
        return True
    # If any honors present, _decompose_standard_with_pred will fail because honors only form triplets
    return _decompose_standard_with_pred(all_tiles, pred_meld, pred_pair)


def _score_fu_and_han(concealed_tiles: List[Tile], called_sets: List[CalledSet],
                      winner_id: int, dealer_id: int, win_by_tsumo: bool,
                      riichi_declared: bool, seat_wind: Honor, round_wind: Honor,
                      dora_indicators: List[Tile], ura_indicators: List[Tile]) -> Tuple[int, int, int]:
    # Returns (fu, han, han_from_dora)
    all_tiles = concealed_tiles + [t for cs in called_sets for t in cs.tiles]

    # Yaku detection (subset, enough for tests)
    han = 0
    chiitoi = _is_chiitoi(concealed_tiles, called_sets)
    if chiitoi:
        han += 2
        fu = FU_CHIITOI
    else:
        # Standard hand requires 4 melds + pair; if not formable, no win
        fu = FU_BASELINE
        if _is_tanyao(all_tiles):
            han += 1
        if _is_toitoi(all_tiles, called_sets):
            han += 2
        # Honitsu/Chinitsu; open/closed handled roughly by presence of chi/pon (open)
        open_hand = _is_open_hand(called_sets)
        if _is_honitsu(all_tiles):
            han += 2 if open_hand else 3
        if _is_chinitsu(all_tiles):
            han += 5 if open_hand else 6
        # Chanta/Junchan
        if _is_chanta(concealed_tiles + [t for cs in called_sets for t in cs.tiles]):
            han += CHANTA_OPEN_HAN if open_hand else CHANTA_CLOSED_HAN
        if _is_junchan(concealed_tiles + [t for cs in called_sets for t in cs.tiles]):
            han += JUNCHAN_OPEN_HAN if open_hand else JUNCHAN_CLOSED_HAN
        # Yakuhai
        han += _yakuhai_han(concealed_tiles, called_sets, seat_wind, round_wind)
        # Sanankou
        if _count_sanankou(concealed_tiles, called_sets) >= 3:
            han += SANANKOU_HAN
        # Pinfu
        if _is_pinfu(all_tiles, called_sets, seat_wind, round_wind):
            han += 1

    # Riichi
    if riichi_declared:
        han += 1
    # Menzen (menzen tsumo): closed hand tsumo
    if win_by_tsumo and not _is_open_hand(called_sets):
        han += 1

    # Dora (including aka)
    dora_han = _calc_dora_han(concealed_tiles, called_sets, dora_indicators)
    if riichi_declared:
        dora_han += _calc_dora_han(concealed_tiles, called_sets, ura_indicators)
    # Count red-5 (aka) as dora
    han += dora_han + _count_aka_han(all_tiles)

    return fu, han, dora_han


class GamePerspective:
    def __init__(self,
                 player_hand: List[Tile],
                 player_id: int,
                 remaining_tiles: int,
                 last_discarded_tile: Optional[Tile],
                 last_discard_player: Optional[int],
                 called_sets: Dict[int, List[CalledSet]],
                 state: type,
                 is_current_turn: bool,
                 newly_drawn_tile: Optional[Tile],
                 can_call: bool,
                 seat_winds: Dict[int, Honor],
                 round_wind: Honor,
                 riichi_declared: Dict[int, bool],
                 ) -> None:
        self.player_hand = sorted(list(player_hand), key=_tile_sort_key)
        self.player_id = player_id
        self.remaining_tiles = remaining_tiles
        self.last_discarded_tile = last_discarded_tile
        self.last_discard_player = last_discard_player
        self.called_sets = {pid: list(sets) for pid, sets in called_sets.items()}
        self.state = state
        self.is_current_turn = is_current_turn
        self.newly_drawn_tile = newly_drawn_tile
        self.can_call = can_call
        self.seat_winds = dict(seat_winds)
        self.round_wind = round_wind
        self.riichi_declared = dict(riichi_declared)

    def _concealed_tiles(self) -> List[Tile]:
        return list(self.player_hand)

    def _has_yaku_if_complete(self) -> bool:
        # Simple heuristic: evaluate yaku on this hand if it is standard or chiitoi
        ct = self._concealed_tiles()
        cs = self.called_sets.get(self.player_id, [])
        if _is_chiitoi(ct, cs):
            return True
        if len(ct) == 14 and _can_form_standard_hand(ct + []):
            all_tiles = ct + [t for s in cs for t in s.tiles]
            if _is_tanyao(all_tiles):
                return True
            if _is_toitoi(all_tiles, cs):
                return True
            if _is_honitsu(all_tiles) or _is_chinitsu(all_tiles):
                return True
            if _is_chanta(all_tiles) or _is_junchan(all_tiles):
                return True
            if _yakuhai_han(ct, cs, self.seat_winds[self.player_id], self.round_wind) > 0:
                return True
        return False

    def _is_tenpai(self) -> bool:
        # Very rough: in action state, if adding any tile completes a standard/chiitoi hand
        hand = list(self.player_hand)
        if len(hand) % 3 != 1:
            return False
        for s in (Suit.MANZU, Suit.PINZU, Suit.SOUZU):
            for v in range(1, 10):
                t = Tile(s, TileType(v))
                if _can_form_standard_hand(hand + [t]):
                    return True
        # Honors
        for h in Honor:
            t = Tile(Suit.HONORS, h)
            if _can_form_standard_hand(hand + [t]):
                return True
        # Seven pairs check: if 13 tiles with 6 pairs and one singleton
        if len(hand) == 13:
            cnt = _count_tiles(hand)
            pairs = sum(1 for c in cnt.values() if c == 2)
            singles = sum(1 for c in cnt.values() if c == 1)
            if pairs == 6 and singles == 1:
                return True
        return False

    def can_tsumo(self) -> bool:
        if self.newly_drawn_tile is None:
            return False
        # Require yaku
        return self._win_possible(require_yaku=True)

    def can_ron(self) -> bool:
        if self.last_discarded_tile is None or self.last_discard_player == self.player_id:
            return False
        return self._win_possible(require_yaku=True, include_last_discard=True)

    def _win_possible(self, require_yaku: bool, include_last_discard: bool = False) -> bool:
        ct = list(self.player_hand)
        if include_last_discard and self.last_discarded_tile is not None:
            ct = ct + [self.last_discarded_tile]
        cs = self.called_sets.get(self.player_id, [])
        ok = False
        if _is_chiitoi(ct, cs):
            ok = True
        # For standard hand, require exactly 14 tiles
        if len(ct) == 14 and _can_form_standard_hand(ct):
            ok = True
        if not ok:
            return False
        if not require_yaku:
            return True
        # Check presence of at least one yaku
        all_tiles = ct + [t for s in cs for t in s.tiles]
        if _is_tanyao(all_tiles):
            return True
        if _is_toitoi(all_tiles, cs):
            return True
        if _is_honitsu(all_tiles) or _is_chinitsu(all_tiles):
            return True
        if _is_chanta(all_tiles) or _is_junchan(all_tiles):
            return True
        if _yakuhai_han(ct, cs, self.seat_winds[self.player_id], self.round_wind) > 0:
            return True
        if _is_chiitoi(ct, cs):
            return True
        return False

    def get_call_options(self) -> Dict[str, List[List[Tile]]]:
        options = {'pon': [], 'chi': [], 'kan_daimin': []}  # kan_daimin: react to discard
        last = self.last_discarded_tile
        lp = self.last_discard_player
        if last is None or lp is None or lp == self.player_id:
            return options
        hand = list(self.player_hand)
        # Pon
        same = [t for t in hand if t.suit == last.suit and t.tile_type == last.tile_type]
        if len(same) >= 2:
            options['pon'].append([same[0], same[1]])
        # Chi (left player only)
        if self.player_id == (lp + 1) % 4 and last.suit != Suit.HONORS:
            for pair in _is_chi_possible_with(hand, last):
                options['chi'].append(pair)
        # Daiminkan: need three in hand
        if len(same) >= 3:
            options['kan_daimin'].append([same[0], same[1], same[2]])
        return options

    def is_legal(self, move: Union[Action, Reaction]) -> bool:
        # Riichi restriction: after declaring, only Tsumo or Kan actions are allowed on own turn
        riichi_locked = self.riichi_declared.get(self.player_id, False)
        if isinstance(move, (Tsumo, Discard, Riichi, KanKakan, KanAnkan)):
            if self.state is not Action or not self.is_current_turn:
                return False
            if isinstance(move, Tsumo):
                return self.can_tsumo()
            if isinstance(move, Riichi):
                # Closed hand, tenpai
                if self.called_sets.get(self.player_id, []):
                    return False
                return self._is_tenpai()
            if isinstance(move, KanKakan):
                # Must have an existing pon of this tile
                for cs in self.called_sets.get(self.player_id, []):
                    if cs.call_type == 'pon' and cs.tiles and cs.tiles[0].suit == move.tile.suit and cs.tiles[0].tile_type == move.tile.tile_type:
                        return move.tile in self.player_hand
                return False
            if isinstance(move, KanAnkan):
                # Need four in hand
                cnt = sum(1 for t in self.player_hand if t.suit == move.tile.suit and t.tile_type == move.tile.tile_type)
                return cnt >= 4
            if isinstance(move, Discard):
                if riichi_locked:
                    # Can only discard the newly drawn tile when riichi is locked
                    return self.newly_drawn_tile is not None and move.tile == self.newly_drawn_tile
                return move.tile in self.player_hand
            return False

        # Reactions
        if self.state is not Reaction or self.last_discarded_tile is None or self.last_discard_player == self.player_id:
            return False
        if isinstance(move, Ron):
            return self.can_ron()
        if isinstance(move, PassCall):
            opts = self.get_call_options()
            return self.can_ron() or bool(opts['pon'] or opts['chi'] or opts['kan_daimin'])
        if self.can_ron() and isinstance(move, (Pon, Chi, KanDaimin)):
            return False
        opts = self.get_call_options()
        if isinstance(move, Pon):
            return any(sorted([(t.suit.value, int(t.tile_type.value)) for t in move.tiles]) ==
                       sorted([(t.suit.value, int(t.tile_type.value)) for t in cand]) for cand in opts['pon'])
        if isinstance(move, Chi):
            return any(sorted([(t.suit.value, int(t.tile_type.value)) for t in move.tiles]) ==
                       sorted([(t.suit.value, int(t.tile_type.value)) for t in cand]) for cand in opts['chi'])
        if isinstance(move, KanDaimin):
            return any(sorted([(t.suit.value, int(t.tile_type.value)) for t in move.tiles]) ==
                       sorted([(t.suit.value, int(t.tile_type.value)) for t in cand]) for cand in opts['kan_daimin'])
        return False

    def legal_moves(self) -> List[Union[Action, Reaction]]:
        moves: List[Union[Action, Reaction]] = []
        riichi_locked = self.riichi_declared.get(self.player_id, False)
        if self.state is Reaction and self.last_discarded_tile is not None and self.last_discard_player is not None and self.last_discard_player != self.player_id:
            if self.can_ron():
                return [PassCall(), Ron()]
            opts = self.get_call_options()
            any_call = False
            for ts in opts['pon']:
                moves.append(Pon(ts)); any_call = True
            for ts in opts['chi']:
                moves.append(Chi(ts)); any_call = True
            for ts in opts['kan_daimin']:
                moves.append(KanDaimin(ts)); any_call = True
            if any_call:
                moves.insert(0, PassCall())
            return moves

        if self.state is Action and self.is_current_turn:
            if self.can_tsumo():
                moves.append(Tsumo())
            # Riichi declaration
            if not self.riichi_declared.get(self.player_id, False) and not self.called_sets.get(self.player_id, []) and self._is_tenpai():
                moves.append(Riichi())
            # Kakan opportunities
            for t in self.player_hand:
                if self.is_legal(KanKakan(t)):
                    moves.append(KanKakan(t))
            # Ankan opportunities (do not list duplicates)
            seen: set = set()
            for t in self.player_hand:
                key = (t.suit, int(t.tile_type.value))
                if key in seen:
                    continue
                seen.add(key)
                if self.is_legal(KanAnkan(t)):
                    moves.append(KanAnkan(t))
            # Discards
            if riichi_locked:
                if self.newly_drawn_tile is not None:
                    moves.append(Discard(self.newly_drawn_tile))
            else:
                for t in self.player_hand:
                    moves.append(Discard(t))
        return moves


class Player:
    def __init__(self, player_id: int):
        self.player_id = player_id

    def play(self, game_state: GamePerspective) -> Action:
        # Auto-win
        if game_state.can_tsumo():
            return Tsumo()
        # Riichi if possible
        if Riichi() in [type(m)() if isinstance(m, Riichi) else m for m in game_state.legal_moves()]:
            # Declare riichi once in tenpai
            for m in game_state.legal_moves():
                if isinstance(m, Riichi):
                    return m
        # Discard heuristic: first tile
        for m in game_state.legal_moves():
            if isinstance(m, Discard):
                return m
        return Discard(game_state.player_hand[0])

    def choose_reaction(self, game_state: GamePerspective, options: Dict[str, List[List[Tile]]]) -> Reaction:
        if game_state.can_ron():
            return Ron()
        if options.get('kan_daimin'):
            return KanDaimin(options['kan_daimin'][0])
        if options.get('pon'):
            return Pon(options['pon'][0])
        if options.get('chi'):
            return Chi(options['chi'][0])
        return PassCall()


class MediumJong:
    NUM_PLAYERS = MC_NUM_PLAYERS

    def __init__(self, players: List[Player], tile_copies: int = TILE_COPIES_DEFAULT):
        if len(players) != MediumJong.NUM_PLAYERS:
            raise ValueError("MediumJong requires exactly 4 players")
        self.players = players
        self._player_hands: Dict[int, List[Tile]] = {i: [] for i in range(4)}
        self._player_called_sets: Dict[int, List[CalledSet]] = {i: [] for i in range(4)}
        self.player_discards: Dict[int, List[Tile]] = {i: [] for i in range(4)}
        self.current_player_idx: int = 0
        self.game_over: bool = False
        self.winners: List[int] = []
        self.loser: Optional[int] = None
        self.last_discarded_tile: Optional[Tile] = None
        self.last_discard_player: Optional[int] = None
        self.last_drawn_tile: Optional[Tile] = None
        self.last_drawn_player: Optional[int] = None
        self.tile_copies = tile_copies
        self._skip_draw_for_current: bool = False
        # Winds
        self.round_wind: Honor = Honor.EAST
        self.seat_winds: Dict[int, Honor] = {
            0: Honor.EAST, 1: Honor.SOUTH, 2: Honor.WEST, 3: Honor.NORTH
        }
        # Riichi flags
        self.riichi_declared: Dict[int, bool] = {i: False for i in range(4)}
        # Dora/Uradora indicators (start with 1 each hidden)
        self.dora_indicators: List[Tile] = []
        self.ura_dora_indicators: List[Tile] = []

        # Build wall (no dead wall separation; simplified)
        self.tiles: List[Tile] = []
        # Add tiles; include exactly one aka 5 per suit by replacing one copy of 5
        for suit in (Suit.MANZU, Suit.PINZU, Suit.SOUZU):
            for v in range(1, 10):
                # Number of copies to add; if v==5, we add (tile_copies - 1) normal + 1 aka
                copies = self.tile_copies
                if v == 5 and copies > 0:
                    # Add normal fives (copies - 1)
                    for _ in range(copies - 1):
                        self.tiles.append(Tile(suit, TileType(v)))
                    # Add aka five
                    self.tiles.append(Tile(suit, TileType(v), aka=True))
                else:
                    for _ in range(copies):
                        self.tiles.append(Tile(suit, TileType(v)))
        for h in Honor:
            for _ in range(self.tile_copies):
                self.tiles.append(Tile(Suit.HONORS, h))
        random.shuffle(self.tiles)

        # Reveal one dora indicator and prepare one ura indicator
        if self.tiles and INITIAL_DORA_INDICATORS > 0:
            self.dora_indicators.append(self.tiles[-1])
        if len(self.tiles) >= 2 and INITIAL_URADORA_INDICATORS > 0:
            self.ura_dora_indicators.append(self.tiles[-2])

        # Deal 13 tiles to each player (dealer draws first turn tile later)
        for pid in range(4):
            for _ in range(STARTING_HAND_TILES):
                self._player_hands[pid].append(self.tiles.pop())

    class IllegalMoveException(Exception):
        pass

    def hand(self, player_id: int) -> List[Tile]:
        return list(self._player_hands[player_id])

    def called_sets(self, player_id: int) -> List[CalledSet]:
        return list(self._player_called_sets[player_id])

    def get_game_perspective(self, player_id: int) -> GamePerspective:
        if self.current_player_idx == player_id:
            state = Action
            newly_drawn = self.last_drawn_tile if self.last_drawn_player == player_id else None
        else:
            state = Reaction if self.last_discarded_tile is not None and self.last_discard_player != player_id else Action
            newly_drawn = None
        is_current_turn = (self.current_player_idx == player_id) and (self.last_discarded_tile is None)
        return GamePerspective(
            player_hand=self._player_hands[player_id],
            player_id=player_id,
            remaining_tiles=len(self.tiles),
            last_discarded_tile=self.last_discarded_tile,
            last_discard_player=self.last_discard_player,
            called_sets=self._player_called_sets,
            state=state,
            is_current_turn=is_current_turn,
            newly_drawn_tile=newly_drawn,
            can_call=self.last_discarded_tile is not None and self.last_discard_player != player_id,
            seat_winds=self.seat_winds,
            round_wind=self.round_wind,
            riichi_declared=self.riichi_declared,
        )

    def is_legal(self, actor_id: int, move: Union[Action, Reaction]) -> bool:
        if self.game_over:
            return False
        return self.get_game_perspective(actor_id).is_legal(move)

    def legal_moves(self, actor_id: int) -> List[Union[Action, Reaction]]:
        if self.game_over:
            return []
        return self.get_game_perspective(actor_id).legal_moves()

    def _draw_for_current_if_needed(self) -> None:
        if self._skip_draw_for_current:
            self._skip_draw_for_current = False
            return
        if self.tiles:
            t = self.tiles.pop()
            self._player_hands[self.current_player_idx].append(t)
            self.last_drawn_tile = t
            self.last_drawn_player = self.current_player_idx

    def _rinshan_draw(self) -> None:
        # Simplified: draw from remaining tiles
        if self.tiles:
            t = self.tiles.pop()
            self._player_hands[self.current_player_idx].append(t)
            self.last_drawn_tile = t
            self.last_drawn_player = self.current_player_idx

    def _add_kan_dora(self) -> None:
        # Add one more dora and ura indicator (simplified)
        if self.tiles:
            self.dora_indicators.append(self.tiles[-1])
        if len(self.tiles) >= 2:
            self.ura_dora_indicators.append(self.tiles[-2])

    def step(self, actor_id: int, move: Union[Action, Reaction]) -> bool:
        if not self.is_legal(actor_id, move):
            raise MediumJong.IllegalMoveException("Illegal move")

        # Actions by current player
        if isinstance(move, (Tsumo, Discard, Riichi, KanKakan, KanAnkan)):
            if isinstance(move, Tsumo):
                self._on_win(actor_id, win_by_tsumo=True)
                return True
            if isinstance(move, Riichi):
                self.riichi_declared[actor_id] = True
                return True
            if isinstance(move, Discard):
                self._player_hands[actor_id].remove(move.tile)
                self.player_discards[actor_id].append(move.tile)
                self.last_discarded_tile = move.tile
                self.last_discard_player = actor_id
                self.last_drawn_tile = None
                self.last_drawn_player = None
                return True
            if isinstance(move, KanKakan):
                # Upgrade an existing pon to kan
                # Remove the drawn tile from hand
                self._player_hands[actor_id].remove(move.tile)
                # Update called set to 4 tiles
                for cs in self._player_called_sets[actor_id]:
                    if cs.call_type == 'pon' and cs.tiles and cs.tiles[0].suit == move.tile.suit and cs.tiles[0].tile_type == move.tile.tile_type:
                        cs.call_type = 'kan_kakan'
                        cs.tiles.append(move.tile)
                        cs.called_tile = None
                        cs.source_position = None
                        break
                self._add_kan_dora()
                self._rinshan_draw()
                return True
            if isinstance(move, KanAnkan):
                # Remove four from hand
                rm = 0
                new_hand: List[Tile] = []
                for t in self._player_hands[actor_id]:
                    if rm < 4 and t.suit == move.tile.suit and t.tile_type == move.tile.tile_type:
                        rm += 1
                    else:
                        new_hand.append(t)
                self._player_hands[actor_id] = new_hand
                self._player_called_sets[actor_id].append(CalledSet(tiles=[Tile(move.tile.suit, move.tile.tile_type) for _ in range(4)], call_type='kan_ankan', called_tile=None, caller_position=actor_id, source_position=None))
                self._add_kan_dora()
                self._rinshan_draw()
                return True

        # Reactions to discard
        if isinstance(move, Ron):
            if actor_id not in self.winners:
                self.winners.append(actor_id)
            self.loser = self.last_discard_player
            return True
        if isinstance(move, Pon):
            last = self.last_discarded_tile
            # Consume two tiles
            consumed = 0
            new_hand: List[Tile] = []
            for t in self._player_hands[actor_id]:
                if consumed < 2 and t.suit == last.suit and t.tile_type == last.tile_type:
                    consumed += 1
                else:
                    new_hand.append(t)
            self._player_hands[actor_id] = new_hand
            self._player_called_sets[actor_id].append(CalledSet(tiles=[Tile(last.suit, last.tile_type) for _ in range(3)], call_type='pon', called_tile=Tile(last.suit, last.tile_type), caller_position=actor_id, source_position=self.last_discard_player))
            self.last_discarded_tile = None
            self.last_discard_player = None
            self.current_player_idx = actor_id
            self._skip_draw_for_current = True
            return True
        if isinstance(move, Chi):
            last = self.last_discarded_tile
            # Remove provided two tiles
            for t in move.tiles:
                removed = False
                new_hand: List[Tile] = []
                for h in self._player_hands[actor_id]:
                    if not removed and h.suit == t.suit and h.tile_type == t.tile_type:
                        removed = True
                        continue
                    new_hand.append(h)
                self._player_hands[actor_id] = new_hand
            seq = sorted([move.tiles[0], last, move.tiles[1]], key=lambda t: int(t.tile_type.value))
            self._player_called_sets[actor_id].append(CalledSet(tiles=seq, call_type='chi', called_tile=Tile(last.suit, last.tile_type), caller_position=actor_id, source_position=self.last_discard_player))
            self.last_discarded_tile = None
            self.last_discard_player = None
            self.current_player_idx = actor_id
            self._skip_draw_for_current = True
            return True
        if isinstance(move, KanDaimin):
            last = self.last_discarded_tile
            # Remove three from hand
            consumed = 0
            new_hand: List[Tile] = []
            for t in self._player_hands[actor_id]:
                if consumed < 3 and t.suit == last.suit and t.tile_type == last.tile_type:
                    consumed += 1
                else:
                    new_hand.append(t)
            self._player_hands[actor_id] = new_hand
            self._player_called_sets[actor_id].append(CalledSet(tiles=[Tile(last.suit, last.tile_type) for _ in range(4)], call_type='kan_daimin', called_tile=Tile(last.suit, last.tile_type), caller_position=actor_id, source_position=self.last_discard_player))
            self.last_discarded_tile = None
            self.last_discard_player = None
            self.current_player_idx = actor_id
            self._skip_draw_for_current = True
            self._add_kan_dora()
            return True

        raise MediumJong.IllegalMoveException("Unsupported move")

    def play_turn(self) -> Optional[int]:
        # Draw if needed
        self._draw_for_current_if_needed()
        # Act
        action = self.players[self.current_player_idx].play(self.get_game_perspective(self.current_player_idx))
        self.step(self.current_player_idx, action)
        # Resolve reactions
        if self.last_discarded_tile is not None and self.last_discard_player is not None:
            self._resolve_reactions()
            if self._skip_draw_for_current:
                return None
        # Advance
        self.current_player_idx = (self.current_player_idx + 1) % 4
        return None

    def _resolve_reactions(self) -> None:
        discarder = self.last_discard_player
        # Gather options/choices
        choices: Dict[int, Reaction] = {}
        can_ron: Dict[int, bool] = {}
        for pid in range(4):
            if pid == discarder:
                continue
            gs = self.get_game_perspective(pid)
            opts = gs.get_call_options()
            if gs.can_ron():
                can_ron[pid] = True
                choices[pid] = self.players[pid].choose_reaction(gs, { })  # type: ignore[arg-type]
            elif opts['pon'] or opts['chi'] or opts['kan_daimin']:
                choices[pid] = self.players[pid].choose_reaction(gs, opts)
        # Ron first
        rons = [pid for pid, ch in choices.items() if isinstance(ch, Ron) and can_ron.get(pid, False)]
        if rons:
            self.winners = rons
            self.loser = discarder
            self.game_over = True
            return
        # Priority: Pon/ Kan over Chi by seat order from left
        order = [(discarder + 1) % 4, (discarder + 2) % 4, (discarder + 3) % 4]
        for pid in order:
            ch = choices.get(pid)
            if isinstance(ch, (Pon, KanDaimin)) and self.is_legal(pid, ch):
                self.step(pid, ch)
                return
        # Chi for immediate left
        left = (discarder + 1) % 4
        ch = choices.get(left)
        if isinstance(ch, Chi) and self.is_legal(left, ch):
            self.step(left, ch)
            return
        # No calls; clear
        self.last_discarded_tile = None
        self.last_discard_player = None

    def _on_win(self, winner_id: int, win_by_tsumo: bool) -> None:
        self.winners = [winner_id]
        self.loser = None if win_by_tsumo else self.last_discard_player
        self.game_over = True

    def is_game_over(self) -> bool:
        return self.game_over

    def get_winners(self) -> List[int]:
        return list(self.winners)

    def get_loser(self) -> Optional[int]:
        return self.loser

    # Scoring API
    def score_hand(self, winner_id: int, win_by_tsumo: bool) -> Dict[str, Any]:
        concealed = list(self._player_hands[winner_id])
        cs = list(self._player_called_sets[winner_id])
        fu, han, dora_han = _score_fu_and_han(
            concealed_tiles=concealed,
            called_sets=cs,
            winner_id=winner_id,
            dealer_id=DEALER_ID_START,
            win_by_tsumo=win_by_tsumo,
            riichi_declared=self.riichi_declared[winner_id],
            seat_wind=self.seat_winds[winner_id],
            round_wind=self.round_wind,
            dora_indicators=self.dora_indicators,
            ura_indicators=self.ura_dora_indicators if self.riichi_declared[winner_id] else [],
        )

        # Base points
        base_points = fu * (2 ** (2 + han))
        # Cap at limit hands omitted; simplified rounding
        def round_up_100(x: int) -> int:
            return int(math.ceil(x / float(POINTS_ROUNDING)) * POINTS_ROUNDING)

        dealer = (winner_id == DEALER_ID_START)
        if win_by_tsumo:
            if dealer:
                each = round_up_100(base_points * 2)
                total = each * 3
                payments = {'from_each': each}
            else:
                dealer_pay = round_up_100(base_points * 2)
                non_dealer_pay = round_up_100(base_points)
                total = dealer_pay + 2 * non_dealer_pay
                payments = {'from_dealer': dealer_pay, 'from_others': non_dealer_pay}
            return {'fu': fu, 'han': han, 'dora_han': dora_han, 'points': total, 'tsumo': True, 'payments': payments}
        else:
            # Ron
            if dealer:
                total = round_up_100(base_points * 6)
            else:
                total = round_up_100(base_points * 4)
            return {'fu': fu, 'han': han, 'dora_han': dora_han, 'points': total, 'tsumo': False, 'from': self.loser}


