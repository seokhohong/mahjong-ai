#!/usr/bin/env python3
"""
Text viewer for the current SimpleJong API.

- Wraps `Player` objects to log their decisions (play and reactions)
- Runs the actual engine loop (SimpleJong.play_round) so logic is identical
- Produces concise narration suitable for debugging illegal/edge states

Run:
  python3 -m core.text_viewer --games 1 --tile_copies 4
"""

from typing import List, Any, Dict
import argparse

from src.core.game import (
    SimpleJong,
    Player,
    Tile,
    Tsumo,
    Ron,
    Discard,
)


def _fmt_hand(tiles: List[Tile]) -> str:
    sorted_tiles = sorted(tiles, key=lambda t: (t.suit.value, t.tile_type.value))
    return '[' + ', '.join(str(t) for t in sorted_tiles) + ']'


def _fmt_called_sets(csets: List[Any]) -> str:
    if not csets:
        return '[]'
    parts: List[str] = []
    for cs in csets:
        try:
            tiles = getattr(cs, 'tiles', [])
            call_type = getattr(cs, 'call_type', '?')
            sorted_tiles = sorted(tiles, key=lambda t: (t.suit.value, t.tile_type.value))
            parts.append(f"{call_type}:[" + ', '.join(str(t) for t in sorted_tiles) + "]")
        except Exception:
            continue
    return '[' + '; '.join(parts) + ']'


class TextViewerPlayer(Player):
    """Wrapper player that logs decisions while delegating to a base Player.

    We keep the player stateless; the engine provides GamePerspective with all info.
    """
    def __init__(self, player_id: int, base: Player, lines: List[str]) -> None:
        super().__init__(player_id)
        self._base = base
        self._lines = lines

    def _log_turn(self, gs: Any, action: Any) -> None:
        try:
            newly = gs.newly_drawn_tile
            newly_s = str(newly) if newly is not None else 'None'
            action_s = type(action).__name__
            # Include parameterized info for discards
            try:
                if isinstance(action, Discard):
                    action_s = f"Discard {action.tile}"
            except Exception:
                pass
            my_called = _fmt_called_sets(gs.called_sets.get(gs.player_id, []))
            self._lines.append(
                f"Turn: P{self.player_id} | Hand {_fmt_hand(gs.player_hand)} | Called {my_called} | Draw {newly_s} | Action {action_s}"
            )
        except Exception:
            pass

    def _log_reaction(self, gs: Any, options: Dict[str, List[List[Tile]]], action: Any) -> None:
        try:
            last = gs.last_discarded_tile
            last_s = str(last) if last is not None else 'None'
            pon_ct = len(options.get('pon', [])) if options else 0
            chi_ct = len(options.get('chi', [])) if options else 0
            action_s = type(action).__name__
            my_called = _fmt_called_sets(gs.called_sets.get(gs.player_id, []))
            self._lines.append(
                f"Reaction: P{self.player_id} on {last_s} from P{gs.last_discard_player} | Called {my_called} | opts pon={pon_ct} chi={chi_ct} | Chosen {action_s}"
            )
        except Exception:
            pass

    def play(self, game_state: Any) -> Any:
        action = self._base.play(game_state)
        self._log_turn(game_state, action)
        return action

    def choose_reaction(self, game_state: Any, options: Dict[str, List[List[Tile]]]) -> Any:  # type: ignore[name-defined]
        action = self._base.choose_reaction(game_state, options)
        self._log_reaction(game_state, options, action)
        return action


def simulate_with_text(tile_copies: int = 4, seed: int = 0) -> List[str]:
    """Run a single game with wrapped players and return narration lines."""
    import random
    if seed:
        random.seed(seed)

    lines: List[str] = []

    # Build wrapped players
    base_players = [Player(i) for i in range(4)]
    players = [TextViewerPlayer(i, base_players[i], lines) for i in range(4)]

    game = SimpleJong(players, tile_copies=tile_copies)

    # Run the actual engine loop
    try:
        game.play_round()
    except Exception as e:
        lines.append(f"Exception during play_round: {e}")

    # Outcome
    if game.is_game_over():
        winners = getattr(game, 'get_winners', lambda: [])()
        loser = getattr(game, 'get_loser', lambda: None)()
        if winners:
            if len(winners) == 1:
                lines.append(f"Winner: P{winners[0]} | Loser: {loser}")
            else:
                lines.append(f"Winners: {winners} | Loser: {loser}")
        else:
            lines.append("Draw or no winner")
    else:
        lines.append("Round ended without game_over")

    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="Text viewer for SimpleJong")
    parser.add_argument('--games', type=int, default=1, help='How many games to simulate')
    parser.add_argument('--tile_copies', type=int, default=4, help='Copies per tile type in wall')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    args = parser.parse_args()

    for g in range(args.games):
        lines = simulate_with_text(tile_copies=max(1, int(args.tile_copies)), seed=(args.seed + g))
        print(f"=== Game {g} ===")
        for line in lines:
            print(line)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


