import random
import math
import numpy as np
from typing import List, Optional, Dict, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import copy

from src.core.constants import MAX_CALLED_SETS_PER_PLAYER

# Import PyTorch; keep legacy flag name for compatibility with tests
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import optim
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# Backward-compat alias used by tests (interpreted as "deep learning available")
TENSORFLOW_AVAILABLE = TORCH_AVAILABLE


class Suit(Enum):
    """Enum for tile suits"""
    PINZU = 'p'
    SOUZU = 's'


class TileType(Enum):
    """Enum for the 9 types of tiles (1-9)"""
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9


@dataclass
class Tile:
    """Represents a single tile"""
    suit: Suit
    tile_type: TileType
    
    def __str__(self):
        return f"{self.tile_type.value}{self.suit.value}"
    
    def __eq__(self, other):
        if isinstance(other, Tile):
            return self.suit == other.suit and self.tile_type == other.tile_type
        return False
    
    def __hash__(self):
        return hash((self.suit, self.tile_type))


@dataclass
class Action:
    """Represents a player action"""
    pass

@dataclass
class Reaction:
    """Represents a player reaction"""
    pass


@dataclass
class Tsumo(Action):
    """Action to declare a win (tsumo)"""
    pass


@dataclass
class Ron(Reaction):
    """Action to declare a win on another player's discard (ron)"""
    pass


@dataclass
class Discard(Action):
    """Action to discard a tile"""
    tile: Tile


@dataclass
class Pon(Reaction):
    """Action to call pon (take discarded tile to complete triplet)"""
    tiles: List[Tile]  # The two tiles from hand to complete the triplet
    

@dataclass
class Chi(Reaction):
    """Action to call chi (take discarded tile from left player to complete sequence)"""
    tiles: List[Tile]  # The two tiles from hand to complete the sequence


@dataclass
class PassCall(Reaction):
    """Action to pass on a call opportunity"""
    pass


@dataclass
class CalledSet:
    """Represents a called set (pon or chi)"""
    tiles: List[Tile]  # All 3 tiles in the set
    call_type: str  # "pon" or "chi"
    called_tile: Tile  # The tile that was called
    caller_position: int  # Position of the calling player
    source_position: int  # Position of the player who discarded the tile



def _can_form_melds(tiles: List[Tile], num_melds: int) -> bool:
    """Return True if tiles can be partitioned into num_melds melds (triplets or sequences).

    Simplified rules:
    - Two suits (Pinzu/Souzu), numbers 1..9
    - Melds are either triplets (xxx) or sequences (n,n+1,n+2) in the same suit
    - Exactly 3 * num_melds tiles must be present
    """
    if num_melds <= 0:
        return len(tiles) == 0
    if len(tiles) != 3 * num_melds:
        return False

    # Build counts per suit
    counts = {
        Suit.PINZU: [0] * 10,  # index 1..9
        Suit.SOUZU: [0] * 10,
    }
    for t in tiles:
        counts[t.suit][t.tile_type.value] += 1

    def dfs() -> bool:
        # Find first suit/index with tiles remaining
        for suit in (Suit.PINZU, Suit.SOUZU):
            suit_counts = counts[suit]
            for i in range(1, 10):
                if suit_counts[i] > 0:
                    # Try triplet
                    if suit_counts[i] >= 3:
                        suit_counts[i] -= 3
                        if dfs():
                            return True
                        suit_counts[i] += 3
                    # Try sequence i,i+1,i+2
                    if i <= 7 and suit_counts[i+1] > 0 and suit_counts[i+2] > 0:
                        suit_counts[i] -= 1
                        suit_counts[i+1] -= 1
                        suit_counts[i+2] -= 1
                        if dfs():
                            return True
                        suit_counts[i] += 1
                        suit_counts[i+1] += 1
                        suit_counts[i+2] += 1
                    # Neither worked
                    return False
        # No tiles remain anywhere => successfully partitioned
        return True

    return dfs()


def _can_form_four_melds(tiles: List[Tile]) -> bool:
    """Backward-compatible helper for tests expecting four melds from 12 tiles."""
    return _can_form_melds(tiles, 4)

class InvalidHandStateException(Exception):
    pass

class GamePerspective:
    """Container for game state information available to a player.

    Not a dataclass to allow custom logic. Provides helper methods to check winning conditions
    under the simplified rule of completing exactly four melds.
    """
    def __init__(
        self,
        player_hand: List[Tile],
        remaining_tiles: int,
        player_id: int,
        other_players_discarded: Dict[int, List[Tile]],
        called_sets: Dict[int, List[CalledSet]],
        last_discarded_tile: Optional[Tile] = None,
        last_discard_player: Optional[int] = None,
        can_call: bool = False,
        state: type = Action,
        newly_drawn_tile: Optional[Tile] = None,
        visible_tiles: Optional[List[str]] = None,
        player_discards: Optional[Dict[int, List[str]]] = None,
    ) -> None:
        self.player_hand = list(player_hand)
        if len(self.player_hand) == 0:
            raise InvalidHandStateException()
        self.remaining_tiles = int(remaining_tiles)
        self.player_id = int(player_id)
        self.other_players_discarded = {k: list(v) for k, v in other_players_discarded.items()}
        self.called_sets = {k: list(v) for k, v in called_sets.items()}
        self.last_discarded_tile = last_discarded_tile
        self.last_discard_player = last_discard_player
        self.can_call = bool(can_call)
        self.state = state
        self.newly_drawn_tile = newly_drawn_tile
        # Optional fields used by other components; keep for compatibility
        self.player_discards: Dict[int, List[str]] = dict(player_discards) if player_discards is not None else {}
        self.visible_tiles: List[str] = list(visible_tiles) if visible_tiles is not None else []

    def can_tsumo(self) -> bool:
        """Return True if player's hand including the newly drawn tile completes remaining melds.

        Accounts for already-called sets (melds), requiring only the remaining melds to be formed
        from the concealed tiles in hand.
        """
        if self.newly_drawn_tile is None:
            return False
        num_called = len(self.called_sets.get(self.player_id, []))
        remaining_melds = max(0, 4 - num_called)
        if remaining_melds == 0:
            return False
        return _can_form_melds(list(self.player_hand), remaining_melds)

    def can_ron(self) -> bool:
        """Return True if player's hand plus last discard completes remaining melds.

        Considers number of already-called sets for the player and only requires the
        remaining melds to be formed from the current concealed tiles plus the discard.
        """
        if self.last_discarded_tile is None:
            return False
        # Cannot ron on your own discard
        if self.last_discard_player == self.player_id:
            return False
        num_called = len(self.called_sets.get(self.player_id, []))
        remaining_melds = max(0, 4 - num_called)
        if remaining_melds == 0:
            return False
        tiles = list(self.player_hand) + [self.last_discarded_tile]
        return _can_form_melds(tiles, remaining_melds)


# Backward-compatibility alias for older API/tests
GameState = GamePerspective


class Player:
    """Base player class"""
    # should remain stateless
    def __init__(self, player_id: int):
        self.player_id = player_id
        # Back-reference to owning game; set by SimpleJong when players are attached
        self._game: Optional['SimpleJong'] = None
        # Optional recorder set by higher-level orchestrators (e.g., data generator)
        self._recorder = None
    
    def play(self, game_state: GamePerspective) -> Action:
        """
        Player's turn - returns an Action (Tsumo, Ron, or Discard)
        First checks if the player can win with the current hand
        """
        # Decision is stateless: use game_state contents
        if not game_state.player_hand:
            return Tsumo()
        
        # Check if we can win with current hand (Tsumo)
        if game_state.can_tsumo():
            return Tsumo()
        
        # Check if we can win with the most recent discarded tile (Ron)
        if game_state.last_discarded_tile is not None and game_state.last_discard_player is not None \
           and game_state.last_discard_player != self.player_id:
            if game_state.can_ron():
                return Ron()
        
        # If we can't win, discard the most isolated tile (fewest neighbors within ±2 in same suit)
        return Discard(self.choose_isolated_discard_state(game_state))

    def can_win(self) -> bool:
        """Simplified: return True if our current hand can form 4 melds either by tsumo
        (12 tiles in hand) or by ron using the last discarded tile.

        This is provided to satisfy simple tests; broader integrations use GamePerspective methods.
        """
        if self._game is None:
            return False
        # If game is over and we are recorded as a winner, honor that outcome
        try:
            if getattr(self._game, 'game_over', False) and hasattr(self._game, 'winners'):
                if self.player_id in self._game.winners:
                    return True
        except Exception:
            pass
        hand = self._game._player_hands.get(self.player_id, [])
        # Tsumo check: already holding 12 tiles
        if len(hand) == 12 and _can_form_four_melds(hand):
            return True
        # Ron check: use last discarded tile if not ours
        last_tile = self._game.last_discarded_tile
        last_player = self._game.last_discard_player
        if last_tile is not None and last_player != self.player_id and len(hand) == 11:
            return _can_form_four_melds(list(hand) + [last_tile])
        return False

    def choose_isolated_discard_state(self, game_state: GamePerspective) -> Tile:
        hand = game_state.player_hand
        if not hand:
            return Tile(Suit.PINZU, TileType.ONE)

        def neighbor_count(target: Tile) -> int:
            tv = target.tile_type.value
            return sum(
                1
                for t in hand
                if t is not target and t.suit == target.suit and 1 <= abs(t.tile_type.value - tv) <= 2
            )

        def edge_distance(target: Tile) -> int:
            max_val = 9
            return min(target.tile_type.value - 1, max_val - target.tile_type.value)

        ranked = sorted(
            hand,
            key=lambda t: (
                neighbor_count(t),
                edge_distance(t),
                t.tile_type.value,
                0 if t.suit == Suit.PINZU else 1,
            ),
        )
        return ranked[0]

    # --- Recording hooks (no-op by default) ---
    def set_recorder(self, recorder: Any) -> None:
        """Attach a recorder that will be notified at decision points."""
        self._recorder = recorder

    def notify_turn_opportunity(self, game_state: GamePerspective) -> None:
        """Called by the game at the start of this player's turn before acting."""
        if self._recorder is not None:
            self._recorder.record(game_state, self, kind='turn')

    def notify_reaction_opportunity(self, game_state: GamePerspective) -> None:
        """Called by the game when this player can react to a discard."""
        if self._recorder is not None:
            self._recorder.record(game_state, self, kind='reaction')

    def choose_reaction(self, game_state: GamePerspective, options: Dict[str, List[List[Tile]]]) -> Reaction:
        """Choose a reaction given all legal options simultaneously.

        Default policy: Ron > Pon > Chi > Pass.
        """
        # Winning reaction takes precedence
        if game_state.can_ron():
            return Ron()
        # Among calls, prefer Pon over Chi
        if options.get('pon'):
            return Pon(options['pon'][0])
        if options.get('chi'):
            return Chi(options['chi'][0])
        return PassCall()

class SimpleJong:
    """Simplified Mahjong game with Pinzu and Souzu tiles"""
    NUM_PLAYERS = 4
    def __init__(self, players: List[Player], tile_copies: int = 4):
        if len(players) != SimpleJong.NUM_PLAYERS:
            raise ValueError("SimpleJong requires exactly 4 players")
        
        self.players = players
        # Hands and called sets are managed by the game for stateless players
        self._player_hands: Dict[int, List[Tile]] = {i: [] for i in range(SimpleJong.NUM_PLAYERS)}
        self._player_called_sets: Dict[int, List[CalledSet]] = {i: [] for i in range(SimpleJong.NUM_PLAYERS)}
        self.tiles: List[Tile] = []
        # Track discards per player for visibility/rendering/learning
        self.player_discards: Dict[int, List[Tile]] = {i: [] for i in range(SimpleJong.NUM_PLAYERS)}
        self.current_player_idx = 0
        self.game_over = False
        self.winner = None
        self.winners: List[int] = []
        # The single losing player on a Ron (the discarder). None for Tsumo or draws.
        self.loser: Optional[int] = None
        self.last_discarded_tile = None
        self.last_discard_player = None
        # Track the last drawn tile (and who drew it)
        self.last_drawn_tile: Optional[Tile] = None
        self.last_drawn_player: Optional[int] = None
        self.tile_copies = max(1, int(tile_copies))
        # If True, the current player must NOT draw a tile at the start of their turn
        self._skip_draw_for_current: bool = False
        
        # Initialize tiles: configurable copies of each tile type (1-9) for both suits
        for suit in Suit:
            for tile_type in TileType:
                for _ in range(self.tile_copies):
                    self.tiles.append(Tile(suit, tile_type))
        
        # Shuffle tiles
        random.shuffle(self.tiles)
        
        # Deal 11 tiles to each player
        for i, _ in enumerate(self.players):
            for _ in range(11):
                if self.tiles:
                    tile = self.tiles.pop()
                    self._player_hands[i].append(tile)

        # Attach back-references
        for i, p in enumerate(self.players):
            try:
                p._game = self  # type: ignore[attr-defined]
            except Exception:
                pass

    # Event emission removed for simplification during refactor
    
    def hand(self, player_id: int) -> List[Tile]:
        return self._player_hands[player_id].copy()
    
    def called_sets(self, player_id: int) -> List[CalledSet]:
        return self._player_called_sets[player_id].copy()
    
    def player_discards(self, player_id: int) -> List[Tile]:
        return self.player_discards[player_id].copy()
    
    def get_game_perspective(self, player_id: int) -> GamePerspective:
        """Gets a snapshot of the game for a specific player"""
        player = self.players[player_id]
        
        # Get other players' discarded tiles
        other_players_discarded = {}
        for i, p in enumerate(self.players):
            if i != player_id:
                other_players_discarded[i] = [] # Simplified for now
        
        # Get called sets for all players
        called_sets = {}
        for i in range(4):
            called_sets[i] = self._player_called_sets[i].copy()
        
        # Compute visible tiles from per-player discards
        aggregated_discards: List[Tile] = []
        for i in range(4):
            aggregated_discards.extend(self.player_discards.get(i, []))

        # Determine perspective state and newly drawn tile
        if self.current_player_idx == player_id:
            state = Action
            newly_drawn = self.last_drawn_tile if self.last_drawn_player == player_id else None
        else:
            # We are observing a discard from someone else
            state = Reaction if (self.last_discarded_tile is not None and self.last_discard_player != player_id) else Action
            newly_drawn = None

        gs = GamePerspective(
            player_hand=self._player_hands[player_id].copy(),
            remaining_tiles=len(self.tiles),
            player_id=player_id,
            other_players_discarded=other_players_discarded,
            called_sets=called_sets,
            last_discarded_tile=self.last_discarded_tile,
            last_discard_player=self.last_discard_player,
            can_call=self.last_discarded_tile is not None and self.last_discard_player != player_id,
            state=state,
            newly_drawn_tile=newly_drawn,
        )
        gs.player_discards = {i: [str(t) for t in self.player_discards[i]] for i in range(4)}  # type: ignore[attr-defined]

        return gs
    
    def check_for_calls(self) -> Optional[Tuple[int, str, List[Tile]]]:
        """
        Check if any player can make a call on the discarded tile
        Returns (player_id, call_type, tiles) for the highest priority call, or None
        Priority: Pon > Chi (and pon can be called by any player, chi only by left player)
        """
        
        call_results = []
        
        # Check all other players for possible calls
        for i, player in enumerate(self.players):
            if i != self.last_discard_player:  # Can't call your own discard
                
                # Add pon calls (higher priority)
                for tiles in self.pon_calls():
                    call_results.append((i, 'pon', tiles))
                
                # Add chi calls (lower priority)
                for tiles in self.chi_calls():
                    call_results.append((i, 'chi', tiles))
        
        # If there are any calls, prioritize pon over chi
        # For now, just take the first pon if available, otherwise first chi
        pon_calls = [call for call in call_results if call[1] == 'pon']
        chi_calls = [call for call in call_results if call[1] == 'chi']
        
        if pon_calls:
            return pon_calls[0]
        elif chi_calls:
            return chi_calls[0]
        
        return None
    
    def play_round(self) -> Optional[int]:
        """
        Play through one full game.
        - Each turn: current player draws (unless skipped), takes an action, and discards.
        - After each discard, resolve reactions with priority: Ron > Pon > Chi.
          Multiple Rons can occur on the same discard (multiple winners).
          Pon/Chi transfer the turn to the caller and skip the next draw for that player.
        Returns the first winner's player_id (for compatibility) or None if no winner.
        """
        while not self.game_over and (self.tiles or self.last_discarded_tile is not None):
            if self._resolve_outstanding_reactions_if_any():
                return self.winner

            # Start of turn: draw tile if needed
            self._draw_for_current_if_needed()

            # Let the current player act
            current_player = self.players[self.current_player_idx]
            game_state = self.get_game_perspective(self.current_player_idx)
            action = current_player.play(game_state)

            # Apply the chosen action via unified step handler (propagate any IllegalMoveException)
            self.step(self.current_player_idx, action)

            # If the action resulted in game end, break
            if self.game_over:
                return (self.winners[0] if self.winners else None)

            # If a discard just happened, resolve immediate reactions by priority
            if self.last_discarded_tile is not None and self.last_discard_player is not None:
                if self._resolve_reactions_after_discard():
                    return (self.winners[0] if self.winners else None)
                # If a call transferred the turn, continue loop (skip advancing turn)
                if self._skip_draw_for_current:
                    continue

            # Advance to next player (clockwise in this simplified engine)
            self.current_player_idx = (self.current_player_idx + 1) % 4

            # End game if wall empty and no pending discard
            if not self.tiles and self.last_discarded_tile is None:
                self.game_over = True
                return None

        self.game_over = True
        return None

    # --- Decomposed helpers for round flow ---
    def _resolve_outstanding_reactions_if_any(self) -> bool:
        """Resolve any pending reactions that existed before a new turn could begin.
        Returns True if the game ended during resolution.
        """
        if self.last_discarded_tile is None or self.last_discard_player is None:
            return False

        return self._solicit_and_apply_reactions()

    def _draw_for_current_if_needed(self) -> None:
        if self._skip_draw_for_current:
            self._skip_draw_for_current = False
            return
        if self.tiles:
            new_tile = self.tiles.pop()
            self._player_hands[self.current_player_idx].append(new_tile)
            self.last_drawn_tile = new_tile
            self.last_drawn_player = self.current_player_idx

    def _resolve_reactions_after_discard(self) -> bool:
        """Resolve reactions immediately after a discard. Returns True if game ended."""
        return self._solicit_and_apply_reactions()

    # --- Legality API ---
    class IllegalMoveException(Exception):
        pass

    def is_legal(self, actor_id: int, move: Union[Action, Reaction]) -> bool:
        """Return True if the given move by actor_id is legal in the current state."""
        if self.game_over:
            return False
        # Action legality
        if isinstance(move, (Tsumo, Discard)):
            if actor_id != self.current_player_idx:
                return False
            gs = self.get_game_perspective(actor_id)
            if isinstance(move, Tsumo):
                return gs.can_tsumo()
            if isinstance(move, Discard):
                return move.tile in self._player_hands[actor_id]
            return False

        # Reaction legality
        if self.last_discarded_tile is None or self.last_discard_player is None:
            return False
        if actor_id == self.last_discard_player:
            return False
        rs = self.get_game_perspective(actor_id)
        # If Ron is available, it is the only legal reaction in this simplified ruleset
        if rs.can_ron():
            return isinstance(move, Ron)
        if isinstance(move, Ron):
            return rs.can_ron()
        if isinstance(move, Pon):
            opts = self.get_call_options(rs)
            legal_sets = opts.get('pon', [])
            def key(ts: List[Tile]):
                return sorted([(t.suit.value, t.tile_type.value) for t in ts])
            return any(key(move.tiles) == key(s) for s in legal_sets)
        if isinstance(move, Chi):
            opts = self.get_call_options(rs)
            legal_sets = opts.get('chi', [])
            def key(ts: List[Tile]):
                return sorted([(t.suit.value, t.tile_type.value) for t in ts])
            return any(key(move.tiles) == key(s) for s in legal_sets)
        return False

    def legal_moves(self, actor_id: int) -> List[Union[Action, Reaction]]:
        """Enumerate all legal moves for the given actor in the current state.

        Exactly one category is produced depending on the game phase:
        - Action phase (no outstanding discard): only current player may act.
          Returns some subset of {Tsumo, Discard(tile) for each tile in hand}.
        - Reaction phase (there is an outstanding discard): only non-discarders may react.
          Returns subset of {Ron, Pon(two tiles), Chi(two tiles)} based on call options.
        """
        if self.game_over:
            return []

        moves: List[Union[Action, Reaction]] = []

        # If there is an outstanding discard, we are in reaction phase
        if self.last_discarded_tile is not None and self.last_discard_player is not None:
            # Discarder cannot react to their own tile
            if actor_id == self.last_discard_player:
                return []
            rs = self.get_game_perspective(actor_id)
            # Ron (and disallow Pon/Chi if Ron is available per simplified rules)
            if rs.can_ron():
                moves.append(Ron())
                return moves
            # Pon/Chi options (only when Ron is not available)
            opts = self.get_call_options(rs)
            for tiles in opts.get('pon', []):
                moves.append(Pon(tiles))
            for tiles in opts.get('chi', []):
                moves.append(Chi(tiles))
            return moves

        # Otherwise, action phase: only current player may act
        if actor_id != self.current_player_idx:
            return []

        gs = self.get_game_perspective(actor_id)
        # Tsumo if possible
        if gs.can_tsumo():
            moves.append(Tsumo())
        # All discards from current hand
        for t in list(self._player_hands[actor_id]):
            moves.append(Discard(t))
        return moves

    def _solicit_and_apply_reactions(self) -> bool:
        """Solicit reactions and apply them using the unified step() API.

        Priority: Ron (multiple allowed) > Pon (one, tie-break by seat order from left of discarder) > Chi (left only).
        Returns True if the game ended.
        """
        if self.last_discarded_tile is None or self.last_discard_player is None:
            return False

        discarder = self.last_discard_player

        # Build options and solicit choices
        per_player_options: Dict[int, Dict[str, Any]] = {}
        chosen: Dict[int, Reaction] = {}
        for i in range(4):
            if i == discarder:
                continue
            rs = self.get_game_perspective(i)
            call_opts = self.get_call_options(rs)
            opts = {
                'ron': rs.can_ron(),
                'pon': call_opts.get('pon', []),
                'chi': call_opts.get('chi', []),
            }
            per_player_options[i] = opts
            if opts['ron'] or opts['pon'] or opts['chi']:
                choice = self.players[i].choose_reaction(rs, {'pon': opts.get('pon', []), 'chi': opts.get('chi', [])})
                chosen[i] = choice

        # 1) Ron(s): any players who chose Ron and can_ron
        ronners: List[int] = []
        for i, choice in chosen.items():
            if isinstance(choice, Ron) and per_player_options.get(i, {}).get('ron'):
                ronners.append(i)
        if ronners:
            # Multiple Rons: end immediately; record all winners and the single loser (discarder)
            self.winners = ronners[:]
            self.winner = ronners[0]
            self.loser = discarder
            self.game_over = True
            return True

        # 2) Pon: among players who chose Pon, pick by seat order starting from left of discarder
        seat_order = [ (discarder + 1 + k) % 4 for k in range(3) ]
        for i in seat_order:
            if i in chosen and isinstance(chosen[i], Pon):
                move = chosen[i]
                if self.is_legal(i, move):
                    self.step(i, move)
                    return False

        # 3) Chi: only left of discarder may chi, and only if chosen
        left = (discarder + 1) % 4
        if left in chosen and isinstance(chosen[left], Chi):
            move = chosen[left]
            if self.is_legal(left, move):
                self.step(left, move)
                return False

        # No reactions accepted; clear pending discard
        self.last_discarded_tile = None
        self.last_discard_player = None
        return False

    # --- Unified step API ---
    def step(self, actor_id: int, move: Union[Action, Reaction]) -> bool:
        """Apply a single Action or Reaction for the given actor.

        Raises IllegalMoveException if not legal. Returns True if applied.
        """
        if not self.is_legal(actor_id, move):
            raise SimpleJong.IllegalMoveException("Illegal move")

        # Action: only current player may act
        if isinstance(move, (Tsumo, Discard)):
            gs = self.get_game_perspective(actor_id)
            if isinstance(move, Tsumo):
                self.winner = actor_id
                self.winners = [actor_id]
                self.loser = None
                self.game_over = True
                return True
            if isinstance(move, Discard):
                # Execute discard
                self._player_hands[actor_id].remove(move.tile)
                self.player_discards[actor_id].append(move.tile)
                self.last_discarded_tile = move.tile
                self.last_discard_player = actor_id
                self.last_drawn_tile = None
                self.last_drawn_player = None
                return True

        # Reaction: must have a pending discard and actor cannot be discarder
        rs = self.get_game_perspective(actor_id)

        # Ron
        if isinstance(move, Ron):
            # Multiple rons supported externally by calling step() for each winner
            if not self.winners:
                self.winner = actor_id
            if actor_id not in self.winners:
                self.winners.append(actor_id)
            self.loser = self.last_discard_player
            self.game_over = True
            return True

        # Pon
        if isinstance(move, Pon):
            # Remove two tiles that match the last discard by value
            last = self.last_discarded_tile
            self._remove_matching_tiles(actor_id, last.suit, last.tile_type, 2)
            # Record called set
            called = CalledSet(
                tiles=[Tile(last.suit, last.tile_type), Tile(last.suit, last.tile_type), Tile(last.suit, last.tile_type)],
                call_type='pon',
                called_tile=Tile(last.suit, last.tile_type),
                caller_position=actor_id,
                source_position=self.last_discard_player if self.last_discard_player is not None else -1,
            )
            self._player_called_sets[actor_id].append(called)
            # we should never have 4 called sets
            assert len(self._player_called_sets[actor_id]) <= MAX_CALLED_SETS_PER_PLAYER
            # Transfer turn
            self.last_discarded_tile = None
            self.last_discard_player = None
            self.current_player_idx = actor_id
            self._skip_draw_for_current = True
            return True

        # Chi
        if isinstance(move, Chi):
            # Remove the two specified tiles by value
            for t in move.tiles:
                self._remove_matching_tiles(actor_id, t.suit, t.tile_type, 1)
            # Record called set
            last = self.last_discarded_tile
            called = CalledSet(
                tiles=[Tile(move.tiles[0].suit, move.tiles[0].tile_type), Tile(last.suit, last.tile_type), Tile(move.tiles[1].suit, move.tiles[1].tile_type)],
                call_type='chi',
                called_tile=Tile(last.suit, last.tile_type),
                caller_position=actor_id,
                source_position=self.last_discard_player if self.last_discard_player is not None else -1,
            )
            self._player_called_sets[actor_id].append(called)
            self.last_discarded_tile = None
            self.last_discard_player = None
            self.current_player_idx = actor_id
            self._skip_draw_for_current = True
            return True

        # Should not be reachable
        raise SimpleJong.IllegalMoveException("Unsupported move type")

    def get_call_options(self, reaction_state: GamePerspective) -> Dict[str, List[List[Tile]]]:
        """Compute legal call options (pon/chi) for a reacting player given the current last discard.

        Returns a dict: {'pon': List[List[Tile]], 'chi': List[List[Tile]]}
        where each inner list contains the two tiles from the player's hand required to complete the call.
        """
        options: Dict[str, List[List[Tile]]] = {'pon': [], 'chi': []}
        last = self.last_discarded_tile
        last_from = self.last_discard_player
        if last is None or last_from is None:
            return options
        # Cannot react to own discard
        if last_from == reaction_state.player_id:
            return options

        # Build count map for player's hand
        hand = list(reaction_state.player_hand)

        # Pon: need two identical tiles to last
        same = [t for t in hand if t.suit == last.suit and t.tile_type == last.tile_type]
        if len(same) >= 2:
            options['pon'].append([same[0], same[1]])

        # Chi: only the left player relative to discarder
        left_player = (last_from + 1) % 4
        if reaction_state.player_id == left_player and last.suit in (Suit.PINZU, Suit.SOUZU):
            v = last.tile_type.value
            s = last.suit
            def has_tile(val: int) -> Optional[Tile]:
                for t in hand:
                    if t.suit == s and t.tile_type.value == val:
                        return t
                return None
            # (v-2, v-1)
            if v - 2 >= 1 and v - 1 >= 1:
                a = has_tile(v - 2)
                b = has_tile(v - 1)
                if a and b:
                    options['chi'].append([a, b])
            # (v-1, v+1)
            if v - 1 >= 1 and v + 1 <= 9:
                a = has_tile(v - 1)
                b = has_tile(v + 1)
                if a and b:
                    options['chi'].append([a, b])
            # (v+1, v+2)
            if v + 1 <= 9 and v + 2 <= 9:
                a = has_tile(v + 1)
                b = has_tile(v + 2)
                if a and b:
                    options['chi'].append([a, b])

        return options

    def _remove_matching_tiles(self, player_id: int, suit: Suit, tile_type: TileType, count: int) -> None:
        """Remove up to 'count' tiles matching suit and tile_type from player's hand by value."""
        removed = 0
        new_hand: List[Tile] = []
        for t in self._player_hands[player_id]:
            if removed < count and t.suit == suit and t.tile_type == tile_type:
                removed += 1
            else:
                new_hand.append(t)
        self._player_hands[player_id] = new_hand
    
    def get_winner(self) -> Optional[int]:
        """Get the winner's player_id, or None if no winner"""
        return self.winner

    def get_winners(self) -> List[int]:
        """Get all winners (supports multiple Rons). Empty if no winner."""
        return list(self.winners)
    
    def get_loser(self) -> Optional[int]:
        """Get the losing player's id (the discarder) when a Ron occurs; None otherwise."""
        return self.loser
    
    def is_game_over(self) -> bool:
        """Check if the game is over"""
        return self.game_over
    
    def get_remaining_tiles(self) -> int:
        """Get number of remaining tiles"""
        return len(self.tiles)

    def copy(self) -> 'SimpleJong':
        """Create a deep copy of the current game state suitable for search."""
        copied_players = [Player(i) for i in range(4)]
        new_game = SimpleJong(copied_players)
        # Core state
        new_game.tiles = list(self.tiles)
        new_game._player_hands = {i: list(self._player_hands[i]) for i in range(4)}
        new_game._player_called_sets = {
            i: [
                CalledSet(list(cs.tiles), cs.call_type, cs.called_tile, cs.caller_position, cs.source_position)
                for cs in self._player_called_sets[i]
            ]
            for i in range(4)
        }
        new_game.player_discards = {i: list(self.player_discards[i]) for i in range(4)}
        new_game.current_player_idx = self.current_player_idx
        new_game.game_over = self.game_over
        new_game.winner = self.winner
        new_game.winners = list(self.winners)
        new_game.loser = self.loser
        new_game.last_discarded_tile = self.last_discarded_tile
        new_game.last_discard_player = self.last_discard_player
        new_game.last_drawn_tile = self.last_drawn_tile
        new_game.last_drawn_player = self.last_drawn_player
        new_game._skip_draw_for_current = self._skip_draw_for_current
        new_game.tile_copies = self.tile_copies
        return new_game


class MCTSNode:
    """Node in the Monte Carlo Tree Search tree"""
    
    class NoLegalMoves(Exception):
        pass

    class NoActionFromPlayer(Exception):
        pass

    def __init__(self, game_state: 'SimpleJong', player_id: int, parent=None, action=None, player: Optional['SimpleHeuristicsPlayer']=None):
        self.game_state = game_state
        self.player_id = player_id
        self.parent = parent
        self.action = action  # Action that led to this node
        self.player = player or SimpleHeuristicsPlayer(player_id)
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried_actions = self._get_untried_actions()
        
    def _get_untried_actions(self) -> List[Union[Action, Reaction]]:
        """Enumerate legal moves via engine for the actor at this node."""
        if self.game_state.is_game_over():
            return []
        return list(self.game_state.legal_moves(self.player_id))
    
    def is_terminal(self) -> bool:
        """Check if this is a terminal node (game over)"""
        return self.game_state.is_game_over()
    
    def is_fully_expanded(self) -> bool:
        """Check if all possible actions have been tried"""
        return len(self.untried_actions) == 0
    
    def select_child(self, exploration_constant: float = 1.414) -> 'MCTSNode':
        """Select a child node using UCB1 formula"""
        if not self.children:
            return None
        
        # UCB1 formula: argmax(vi + c * sqrt(ln(N) / ni))
        # Treat unvisited children with infinite exploration term to ensure they get selected
        best_child = None
        best_score = float('-inf')

        for child in self.children:
            if child.visits == 0:
                ucb_score = float('inf')
            else:
                exploitation = child.value / child.visits
                exploration = exploration_constant * math.sqrt(max(1.0, math.log(self.visits)) / child.visits)
                ucb_score = exploitation + exploration

            # Prefer Tsumo/Ron on equal scores
            tie_break = 0
            if isinstance(child.action, Tsumo):
                tie_break = 2
            elif isinstance(child.action, Ron):
                tie_break = 1

            candidate = (ucb_score, tie_break)
            current_best = (best_score, -1 if best_child is None else (2 if isinstance(best_child.action, Tsumo) else (1 if isinstance(best_child.action, Ron) else 0)))

            if candidate > current_best:
                best_score = ucb_score
                best_child = child

        return best_child
    
    def expand(self) -> 'MCTSNode':
        """Expand by choosing a heuristic action and creating one child."""
        if not self.untried_actions:
            raise MCTSNode.NoLegalMoves("No legal moves available during expand")
        # Gather legal moves for actor
        legal = list(self.untried_actions)
        # Select action via heuristic preference; if heuristic returns None or not in legal, pick first
        chosen = self.player.select_action(self.game_state, self.player_id, legal)
        if chosen is None:
            raise MCTSNode.NoActionFromPlayer("Heuristic player returned no action")
        if chosen not in legal:
            raise MCTSNode.NoActionFromPlayer("Heuristic player returned illegal action")
        # Mark as tried
        # Remove by identity if present; fallback to pop(0) if not found
        try:
            self.untried_actions.remove(chosen)
        except ValueError:
            raise MCTSNode.NoActionFromPlayer("Selected action vanished from untried set")

        # Apply to copied state
        new_game_state = self.game_state.copy()
        new_game_state.step(self.player_id, chosen)

        # Determine next actor: advance clockwise by default
        next_player_id = new_game_state.current_player_idx if not new_game_state.is_game_over() else self.player_id
        child = MCTSNode(new_game_state, next_player_id, parent=self, action=chosen, player=self.player)
        self.children.append(child)
        return child
    
    def _copy_game_state(self) -> 'SimpleJong':
        # Legacy method retained for compatibility; now delegates to engine
        return self.game_state.copy()
    
    def simulate(self) -> float:
        """Simulate a playout using heuristic decisions until terminal or wall empty."""
        current_state = self.game_state.copy()
        current_player_id = self.player_id
        while not current_state.is_game_over() and (current_state.tiles or current_state.last_discarded_tile is not None):
            legal = current_state.legal_moves(current_player_id)
            if not legal:
                raise MCTSNode.NoLegalMoves("No legal moves available during simulate")
            action = self.player.select_action(current_state, current_player_id, legal)
            if action is None:
                raise MCTSNode.NoActionFromPlayer("Player returned no action in simulate")
            current_state.step(current_player_id, action)
            if current_state.is_game_over():
                break
            # Progress turn clockwise by default
            current_player_id = current_state.current_player_idx if current_state.last_discarded_tile is None else (current_player_id + 1) % 4
        return self._get_reward(current_state, self.player_id)
    
    def _select_random_action(self, player: 'Player', possible_actions: Dict[str, List], game_state: 'SimpleJong') -> Action:
        # Legacy stub no longer used
        legal = game_state.legal_moves(self.player_id)
        return legal[0] if legal else None

    def _get_reward(self, game_state: 'SimpleJong', player_id: int) -> float:
        """Get reward for the player based on game outcome"""
        if not game_state.is_game_over():
            return 0.0  # Draw
        
        if game_state.winner == player_id:
            return 1.0  # Win
        else:
            return -1.0  # Loss
    
    def backpropagate(self, reward: float):
        """Backpropagate the reward up the tree"""
        node = self
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent
    
    def get_best_action(self) -> Action:
        """Get the best action based on visit counts"""
        if not self.children:
            return None
        
        best_child = max(self.children, key=lambda child: child.visits)
        return best_child.action


class PQNetwork:
    """Policy-value network implemented in PyTorch (keeps legacy API where feasible)."""

    class _TorchPQ(nn.Module):
        def __init__(self, input_dim: int, hidden_size: int):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
            self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 2)
            # Heads
            self.head_action = nn.Linear(hidden_size // 2, 5)
            self.head_tile1 = nn.Linear(hidden_size // 2, 18)
            self.head_tile2 = nn.Linear(hidden_size // 2, 18)
            self.head_value = nn.Linear(hidden_size // 2, 1)

        def forward(self, x: torch.Tensor):
            x = F.relu(self.fc1(x))
            x = F.dropout(x, p=0.3, training=self.training)
            x = F.relu(self.fc2(x))
            x = F.dropout(x, p=0.3, training=self.training)
            x = F.relu(self.fc3(x))
            pa = F.softmax(self.head_action(x), dim=-1)
            pt1 = F.softmax(self.head_tile1(x), dim=-1)
            pt2 = F.softmax(self.head_tile2(x), dim=-1)
            val = torch.tanh(self.head_value(x))
            return pa, pt1, pt2, val

    def __init__(self, hidden_size: int = 128, embedding_dim: int = 4, max_turns: int = 50):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for PQNetwork. Please install torch.")
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.max_turns = max_turns
        # Compute flattened feature size used by our simple MLP
        self._flat_dim = (12 * 5) + (4 * self.max_turns * self.embedding_dim) + 50
        self.model = PQNetwork._TorchPQ(self._flat_dim, hidden_size)
        self.model.eval()
    
    def evaluate(self, game_state: 'GamePerspective') -> Tuple[Dict[str, np.ndarray], float]:
        """
        Evaluate a game state and return (policy, value)
        policy: dict containing three heads:
          - 'action': np.ndarray shape (5,)
          - 'tile1': np.ndarray shape (18,)
          - 'tile2': np.ndarray shape (18,)
        value: estimated state value (-1 to 1)
        """
        # Extract features
        features = self._extract_features(game_state)
        
        # Flatten inputs into a single vector
        flat_parts: List[np.ndarray] = []
        # hand (12,5)
        flat_parts.append(features[0].reshape(-1))
        # 4 discard tensors
        for i in range(1, 5):
            flat_parts.append(features[i].reshape(-1))
        # game state (50,)
        flat_parts.append(features[5].reshape(-1))
        x = np.concatenate(flat_parts, axis=0)[None, :].astype(np.float32)
        with torch.no_grad():
            pa, pt1, pt2, val = self.model(torch.from_numpy(x))
        pred_action, pred_tile1, pred_tile2, value = (
            pa.numpy(), pt1.numpy(), pt2.numpy(), val.numpy()
        )
        policy = {
            'action': pred_action[0],
            'tile1': pred_tile1[0],
            'tile2': pred_tile2[0]
        }
        return policy, float(value[0][0])
    
    def _extract_features(self, game_state: 'GamePerspective') -> List[np.ndarray]:
        """Extract features from game state using the new convolutional architecture"""
        
        # 1. Hand features: (12, 5) - 12 tiles, 5 features (4 embedding + 1 called flag)
        hand_features = self._encode_hand_convolutional(game_state.player_hand, game_state.called_sets.get(game_state.player_id, []))
        
        # 2. Discard pile features: (max_turns, embedding_dim) for each player
        discard_features = []
        for player_id in range(4):
            player_discards = self._get_player_discards(game_state, player_id)
            discard_features.append(self._encode_discard_pile_convolutional(player_discards))
        
        # 3. Additional game state features
        game_state_features = self._extract_additional_features(game_state)
        
        return [hand_features] + discard_features + [game_state_features]
    
    def _encode_hand_convolutional(self, hand: List[Tile], called_sets: List['CalledSet']) -> np.ndarray:
        """Encode hand as (12, 5) tensor for convolutional processing"""
        # Initialize with zeros: (12, 5) - 12 tiles, 5 features (4 embedding + 1 called flag)
        hand_tensor = np.zeros((12, 5))
        
        # Create a set of called tiles for quick lookup
        called_tiles = set()
        for called_set in called_sets:
            for tile in called_set.tiles:
                called_tiles.add(tile)
        
        # Fill in the hand tensor
        for i, tile in enumerate(hand[:12]):  # Limit to 12 tiles
            # Get tile embedding (first 4 features)
            tile_embedding = self._get_tile_embedding(tile)
            hand_tensor[i, :4] = tile_embedding
            
            # Set called flag (5th feature)
            hand_tensor[i, 4] = 1.0 if tile in called_tiles else 0.0
        
        return hand_tensor
    
    def _encode_discard_pile_convolutional(self, discards: List[str]) -> np.ndarray:
        """Encode discard pile as (max_turns, embedding_dim) tensor"""
        # Initialize with zeros
        discard_tensor = np.zeros((self.max_turns, self.embedding_dim))
        
        # Fill in the discard tensor (most recent discards first)
        for i, tile_str in enumerate(discards[:self.max_turns]):
            try:
                # Parse tile string (e.g., "1p", "5s")
                tile_type = int(tile_str[:-1])
                suit = Suit(tile_str[-1])
                tile = Tile(suit, TileType(tile_type))
                
                # Get tile embedding
                tile_embedding = self._get_tile_embedding(tile)
                discard_tensor[i] = tile_embedding
            except:
                continue
        
        return discard_tensor
    
    def _get_player_discards(self, game_state: 'GamePerspective', player_id: int) -> List[str]:
        """Get discards for a specific player"""
        if hasattr(game_state, 'player_discards') and player_id in game_state.player_discards:
            return game_state.player_discards[player_id]
        return []
    
    def _get_tile_embedding(self, tile: Tile) -> np.ndarray:
        """Get embedding for a single tile"""
        # For now, use a simple hash-based embedding
        # In a full implementation, you might want to learn these embeddings
        embedding = np.zeros(self.embedding_dim)
        tile_idx = self._get_tile_index(tile)
        
        # Simple hash-based embedding
        np.random.seed(tile_idx)  # For reproducible embeddings
        embedding = np.random.randn(self.embedding_dim) * 0.1
        np.random.seed()  # Reset seed
        
        return embedding
    
    def _get_tile_index(self, tile: Tile) -> int:
        """Get index for a tile in the embedding matrix"""
        return (tile.tile_type.value - 1) * 2 + (0 if tile.suit == Suit.PINZU else 1)
    
    def _extract_additional_features(self, game_state: 'GamePerspective') -> np.ndarray:
        """Extract additional game state features"""
        features = []
        
        # Remaining tiles count (normalized)
        features.append(game_state.remaining_tiles / 72.0)
        
        # Can call feature
        features.append(1.0 if game_state.can_call else 0.0)
        
        # Number of tiles in hand
        features.append(len(game_state.player_hand) / 14.0)  # Max 14 tiles
        
        # Number of called sets
        your_called_sets = game_state.called_sets.get(game_state.player_id, [])
        features.append(len(your_called_sets) / 4.0)  # Max 4 called sets
        
        # Number of opponents' called sets
        total_opponent_sets = sum(len(sets) for player_id, sets in game_state.called_sets.items() 
                                if player_id != game_state.player_id)
        features.append(total_opponent_sets / 12.0)  # Max 12 total opponent sets
        
        # Number of visible tiles
        features.append(len(game_state.visible_tiles) / 72.0)
        
        # Last discarded tile features
        if game_state.last_discarded_tile:
            last_tile_embedding = self._get_tile_embedding(game_state.last_discarded_tile)
            features.extend(last_tile_embedding)
        else:
            features.extend([0.0] * self.embedding_dim)
        
        # Last discard player (one-hot encoded)
        if game_state.last_discard_player is not None:
            player_features = [0.0] * 4
            player_features[game_state.last_discard_player] = 1.0
            features.extend(player_features)
        else:
            features.extend([0.0] * 4)
        
        # Pad to 50 features
        while len(features) < 50:
            features.append(0.0)
        
        return np.array(features[:50])
    
    def get_action_probabilities(self, game_state: 'GamePerspective', possible_actions: Dict[str, List]) -> Dict[str, float]:
        """
        Get probability distribution over possible actions using the multi-head policy.
        Returns dict mapping action strings to probabilities.
        """
        policy, _ = self.evaluate(game_state)

        def tile_to_index_from_str(tile_str: str) -> int:
            # tile_str like '3p' or '5s'
            tile_type = int(tile_str[:-1])
            suit = Suit(tile_str[-1])
            return self._get_tile_index(Tile(suit, TileType(tile_type)))

        action_category_order = ['discard', 'ron', 'tsumo', 'pon', 'chi']
        action_idx = {name: i for i, name in enumerate(action_category_order)}

        action_probs: Dict[str, float] = {}

        # Tsumo and Ron (no tile heads needed)
        if possible_actions.get('tsumo'):
            action_probs['tsumo'] = float(policy['action'][action_idx['tsumo']])
        if possible_actions.get('ron'):
            action_probs['ron'] = float(policy['action'][action_idx['ron']])

        # Discards
        for tile in game_state.player_hand:
            tile_idx = self._get_tile_index(tile)
            prob = float(policy['action'][action_idx['discard']]) * float(policy['tile1'][tile_idx])
            action_probs[f'discard_{str(tile)}'] = prob

        # Pon calls (use last discarded tile to set tile1)
        if possible_actions.get('pon'):
            last_tile = game_state.last_discarded_tile
            if last_tile is not None:
                lt_idx = self._get_tile_index(last_tile)
                base = float(policy['action'][action_idx['pon']]) * float(policy['tile1'][lt_idx])
                for tiles in possible_actions['pon']:
                    # tiles is list of strings representing two tiles from hand
                    key = f"pon_{'_'.join(str(t) for t in tiles)}"
                    action_probs[key] = base

        # Chi calls (use two tiles from hand for tile1 and tile2)
        if possible_actions.get('chi'):
            base_action = float(policy['action'][action_idx['chi']])
            for tiles in possible_actions['chi']:
                # tiles is list like ['2s','4s']
                t1_idx = tile_to_index_from_str(tiles[0])
                t2_idx = tile_to_index_from_str(tiles[1])
                prob = base_action * float(policy['tile1'][t1_idx]) * float(policy['tile2'][t2_idx])
                key = f"chi_{'_'.join(str(t) for t in tiles)}"
                action_probs[key] = prob


        # Normalize probabilities
        total_prob = sum(action_probs.values())
        if total_prob > 0:
            for k in list(action_probs.keys()):
                action_probs[k] /= total_prob
        return action_probs
    
    def save_model(self, filepath: str):
        """Save the model weights (PyTorch)."""
        if not filepath.endswith('.pt'):
            filepath += '.pt'
        torch.save(self.model.state_dict(), filepath)
    
    def load_model(self, filepath: str):
        """Load the model weights (PyTorch)."""
        if not filepath.endswith('.pt'):
            filepath += '.pt'
        state = torch.load(filepath, map_location='cpu')
        self.model.load_state_dict(state)
    
    def train(self, training_data: List[Tuple['GamePerspective', Any, float]], epochs: int = 10, batch_size: int = 32):
        """
        Train the model on provided data
        training_data: List of (game_state, target_policy, target_value) tuples
          - target_policy can be either a dict with keys {'action','tile1','tile2'}
            or a single 1D vector of length 41 (5 + 18 + 18), which will be split.
        """
        if not training_data:
            return
        
        # Prepare training data
        hand_inputs = []
        discard_inputs = [[] for _ in range(4)]
        game_state_inputs = []
        y_action = []
        y_tile1 = []
        y_tile2 = []
        y_value = []
        
        for game_state, target_policy, target_value in training_data:
            features = self._extract_features(game_state)
            
            hand_inputs.append(features[0])
            for i in range(4):
                discard_inputs[i].append(features[i + 1])
            game_state_inputs.append(features[-1])
            
            # Accept dict or concatenated vector
            if isinstance(target_policy, dict):
                y_action.append(target_policy['action'])
                y_tile1.append(target_policy['tile1'])
                y_tile2.append(target_policy['tile2'])
            else:
                # Assume 1D vector of length 41: [5 | 18 | 18]
                tp = np.asarray(target_policy)
                y_action.append(tp[:5])
                y_tile1.append(tp[5:23])
                y_tile2.append(tp[23:41])
            y_value.append([target_value])
        
        # Convert to numpy arrays
        hand_inputs = np.array(hand_inputs)
        discard_inputs = [np.array(discards) for discards in discard_inputs]
        game_state_inputs = np.array(game_state_inputs)
        y_action = np.array(y_action)
        y_tile1 = np.array(y_tile1)
        y_tile2 = np.array(y_tile2)
        y_value = np.array(y_value)
        
        # Train the model
        self.model.fit(
            [hand_inputs] + discard_inputs + [game_state_inputs],
            {
                'policy_action': y_action,
                'policy_tile1': y_tile1,
                'policy_tile2': y_tile2,
                'value': y_value
            },
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )


class AIPlayer(Player):
    """AI player using MCTS with neural network value estimation"""
    
    def __init__(self, player_id: int, simulation_count: int = 1000, exploration_constant: float = 1.414, enable_pq: bool = False, pq_network: Optional['PQNetwork'] = None):
        super().__init__(player_id)
        self.simulation_count = simulation_count
        self.exploration_constant = exploration_constant
        
        # Initialize PQNetwork for policy and value estimation (if available)
        self.pq_network: Optional[PQNetwork] = None
        if enable_pq and TENSORFLOW_AVAILABLE:
            # Use provided network if supplied
            if pq_network is not None:
                self.pq_network = pq_network
            else:
                self.pq_network = PQNetwork(hidden_size=64, embedding_dim=4, max_turns=20)
        self.current_game = None
    
    def play(self, game_state: GamePerspective) -> Action:
        """Use MCTS to select the best action"""
        
        # Check for immediate wins
        if self.can_win():
            return Tsumo()
        
        # Check for Ron
        if (game_state.last_discarded_tile and 
            game_state.last_discard_player != self.player_id):
            if self.can_ron(game_state.last_discarded_tile):
                return Ron()
        
        # Use MCTS to find best action
        best_action = self._mcts_search(game_state)
        
        if best_action is None:
            # Fallback: discard isolated tile based on state
            return Discard(self.choose_isolated_discard_state(game_state))
        
        return best_action
    
    def _run_mcts(self, game_state: GamePerspective) -> 'MCTSNode':
        """Run MCTS simulations and return the root node."""
        game_copy = self._create_game_copy(game_state)
        root = MCTSNode(game_copy, self.player_id)
        for _ in range(self.simulation_count):
            node = root
            while node.is_fully_expanded() and not node.is_terminal():
                node = node.select_child(self.exploration_constant)
                if node is None:
                    break
            if node and not node.is_terminal():
                node = node.expand()
            if node:
                reward = node.simulate()
                if not node.is_terminal() and self.pq_network is not None:
                    _, value_estimate = self.pq_network.evaluate(
                        node.game_state.get_turn_snapshot(self.player_id)
                    )
                    reward = 0.5 * reward + 0.5 * float(value_estimate)
                node.backpropagate(reward)
        return root

    def _mcts_search(self, game_state: GamePerspective) -> Action:
        """Perform MCTS search to find the best action"""
        root = self._run_mcts(game_state)
        return root.get_best_action()

    def _get_policy_tile_index(self, tile: Tile) -> int:
        """Index mapping for policy tile heads (18 = 9 values x 2 suits)."""
        return (tile.tile_type.value - 1) * 2 + (0 if tile.suit == Suit.PINZU else 1)

    def get_policy_and_value(self, game_state: GamePerspective) -> Tuple[np.ndarray, float]:
        """Compute MCTS-based multi-head policy vector [5|18|18] and value estimate."""
        root = self._run_mcts(game_state)

        # Aggregate visits per action type and per tile indices for heads
        action_categories = ['discard', 'ron', 'tsumo', 'pon', 'chi']
        action_idx = {name: i for i, name in enumerate(action_categories)}
        action_counts = np.zeros(5, dtype=np.float32)
        tile1_counts = np.zeros(18, dtype=np.float32)
        tile2_counts = np.zeros(18, dtype=np.float32)

        total_visits = sum(child.visits for child in root.children) or 1
        # Count actions
        for child in root.children:
            if child.visits <= 0:
                continue
            weight = float(child.visits)
            a = child.action
            if isinstance(a, Tsumo):
                action_counts[action_idx['tsumo']] += weight
            elif isinstance(a, Ron):
                action_counts[action_idx['ron']] += weight
            elif isinstance(a, Discard):
                action_counts[action_idx['discard']] += weight
                idx = self._get_policy_tile_index(a.tile)
                if 0 <= idx < 18:
                    tile1_counts[idx] += weight
            elif isinstance(a, Pon):
                action_counts[action_idx['pon']] += weight
                # For Pon, tile1 refers to last discarded tile type
                last = root.game_state.last_discarded_tile
                if last is not None:
                    idx = self._get_policy_tile_index(last)
                    if 0 <= idx < 18:
                        tile1_counts[idx] += weight
            elif isinstance(a, Chi):
                action_counts[action_idx['chi']] += weight
                if a.tiles and len(a.tiles) >= 2:
                    t1 = self._get_policy_tile_index(a.tiles[0])
                    t2 = self._get_policy_tile_index(a.tiles[1])
                    if 0 <= t1 < 18:
                        tile1_counts[t1] += weight
                    if 0 <= t2 < 18:
                        tile2_counts[t2] += weight

        # Normalize heads
        if action_counts.sum() > 0:
            action_probs = action_counts / action_counts.sum()
        else:
            action_probs = np.full(5, 1.0 / 5.0, dtype=np.float32)

        if tile1_counts.sum() > 0:
            tile1_probs = tile1_counts / tile1_counts.sum()
        else:
            tile1_probs = np.full(18, 1.0 / 18.0, dtype=np.float32)

        if tile2_counts.sum() > 0:
            tile2_probs = tile2_counts / tile2_counts.sum()
        else:
            tile2_probs = np.full(18, 1.0 / 18.0, dtype=np.float32)

        # Value estimate from root statistics
        value_estimate: float = 0.0
        if root.visits > 0:
            value_estimate = float(root.value / root.visits)

        return np.concatenate([action_probs, tile1_probs, tile2_probs], axis=0), value_estimate
    
    def _create_game_copy(self, game_state: GamePerspective) -> 'SimpleJong':
        """Create a copy of the game state for MCTS"""
        # Create new game instance with placeholder players
        copied_players = [AIPlayer(i, self.simulation_count if i == self.player_id else 100,
                                   self.exploration_constant if i == self.player_id else 1.414,
                                   enable_pq=False)
                          if i == self.player_id else Player(i) for i in range(4)]
        new_game = SimpleJong(copied_players)
        
        # Copy game state
        new_game.tiles = []
        new_game.current_player_idx = self.player_id
        new_game.game_over = False
        new_game.winner = None
        new_game.last_discarded_tile = game_state.last_discarded_tile
        new_game.last_discard_player = game_state.last_discard_player

        return new_game
    
    def set_game_state(self, game_state: 'SimpleJong'):
        """Set the current game state for MCTS"""
        self.current_game = game_state


class SimpleHeuristicsPlayer:
    """Heuristic policy used by MCTS for action selection during expansion/simulation.

    Rules:
    - Always take Tsumo or Ron if available.
    - Prefer Pon/Chi sometimes but not always: deterministic preference Pon > Chi if available.
    - For Discard, choose the tile with the fewest neighbors within +/-2 in same suit;
      break ties by outermost (closest to 1 or 9), then by tile number and suit for stability.
    """

    def __init__(self, player_id: int):
        self.player_id = player_id

    def select_action(self, game: 'SimpleJong', actor_id: int, legal_moves: List[Union[Action, Reaction]]) -> Optional[Union[Action, Reaction]]:
        if not legal_moves:
            return None
        # 1) Immediate wins
        for m in legal_moves:
            if isinstance(m, (Tsumo, Ron)):
                return m
        # 2) Prefer calls occasionally; here we deterministically choose Pon first, else Chi
        pon_moves = [m for m in legal_moves if isinstance(m, Pon)]
        if pon_moves:
            return pon_moves[0]
        chi_moves = [m for m in legal_moves if isinstance(m, Chi)]
        if chi_moves:
            return chi_moves[0]
        # 3) Discard heuristic
        discards = [m for m in legal_moves if isinstance(m, Discard)]
        if discards:
            # Compute neighbor counts on actor's current hand
            hand = list(game._player_hands[actor_id])
            def neighbor_count(target: Tile) -> int:
                tv = target.tile_type.value
                s = target.suit
                return sum(1 for t in hand if t is not target and t.suit == s and abs(t.tile_type.value - tv) <= 2)
            def edge_distance(target: Tile) -> int:
                return min(target.tile_type.value - 1, 9 - target.tile_type.value)
            # Rank tiles per heuristic; pick the discard that matches the best tile
            ranked = sorted(hand, key=lambda t: (neighbor_count(t), edge_distance(t), t.tile_type.value, 0 if t.suit == Suit.PINZU else 1))
            best_tile = ranked[0]
            for d in discards:
                if d.tile == best_tile:
                    return d
            # Fallback: first discard
            return discards[0]
        # Fallback to first move
        return legal_moves[0]