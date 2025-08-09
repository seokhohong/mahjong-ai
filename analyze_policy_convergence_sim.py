#!/usr/bin/env python3
"""
Simulate games to collect obvious tsumo/ron states and test MCTS policy recommendations.

This script plays simplified Mahjong games (SimpleJong) and, during play:
  - Captures states where the current player can tsumo (obvious win)
  - After a discard, checks opponents for immediate ron opportunities (obvious win)

For each captured state, it runs MCTS (using MCTSNode + SimpleHeuristicsPlayer) from that
player's perspective and checks whether the recommended action matches the obvious
action (tsumo/ron).

Usage:
  python3 analyze_policy_convergence_sim.py \
    --target 100 \
    --simulations 300 \
    --max_games 200
"""

import os
import sys
import argparse
from typing import List, Tuple

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from core.game import (  # type: ignore
    SimpleJong,
    Player,
    MCTSNode,
    SimpleHeuristicsPlayer,
    GamePerspective,
    Tsumo,
    Ron,
    Discard,
    Tile,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate and analyze obvious policy convergence with MCTS")
    parser.add_argument("--games", type=int, default=100, help="Number of games to simulate")
    parser.add_argument("--simulations", type=int, default=200, help="MCTS simulations per evaluation")
    parser.add_argument("--tile_copies", type=int, default=4, help="Number of copies per tile in the wall (SimpleJong)")
    parser.add_argument("--verbose", action="store_true", help="Print details for mismatches")
    return parser.parse_args()


def run_mcts_for_player(game: SimpleJong, actor_id: int, simulations: int) -> str:
    """Run a lightweight MCTS using MCTSNode and return the recommended action name."""
    root_game = game.copy()
    root = MCTSNode(root_game, actor_id, player=SimpleHeuristicsPlayer(actor_id))
    # Basic MCTS loop
    for _ in range(simulations):
        node = root
        while node is not None and node.is_fully_expanded() and not node.is_terminal():
            node = node.select_child(1.414)
            if node is None:
                break
        if node is None:
            break
        if not node.is_terminal():
            try:
                node = node.expand()
            except Exception:
                # If expansion fails, simulate from current node
                pass
        reward = node.simulate() if node is not None else 0.0
        if node is not None:
            node.backpropagate(reward)
    best = root.get_best_action()
    if best is None:
        return 'none'
    if isinstance(best, Tsumo):
        return 'tsumo'
    if isinstance(best, Ron):
        return 'ron'
    return best.__class__.__name__.lower()


def summarize(results: List[Tuple[str, str, GamePerspective]], verbose: bool):
    total = len(results)
    tsumo_cases = [r for r in results if r[0] == 'tsumo']
    ron_cases = [r for r in results if r[0] == 'ron']
    tsumo_correct = sum(1 for exp, rec, _ in tsumo_cases if rec == 'tsumo')
    ron_correct = sum(1 for exp, rec, _ in ron_cases if rec == 'ron')
    print(f"Collected obvious states: {total}")
    print(f"  Tsumo: {len(tsumo_cases)} | correct: {tsumo_correct} | acc: {(tsumo_correct/len(tsumo_cases)) if tsumo_cases else 0.0:.2f}")
    print(f"  Ron  : {len(ron_cases)} | correct: {ron_correct} | acc: {(ron_correct/len(ron_cases)) if ron_cases else 0.0:.2f}")
    if verbose:
        for exp, rec, gs in results:
            if exp != rec:
                print("- Mismatch:")
                print(f"  Expected: {exp} | Recommended: {rec}")
                hand = ','.join(str(t) for t in gs.player_hand)
                last = str(gs.last_discarded_tile) if gs.last_discarded_tile else 'None'
                print(f"  Player {gs.player_id} hand: [{hand}] | last_discard: {last} from {gs.last_discard_player}")


def main():
    args = parse_args()
    results: List[Tuple[str, str, GamePerspective]] = []
    games_played = 0
    ron_count = 0
    tsumo_count = 0

    for _ in range(args.games):
        players = [Player(0), Player(1), Player(2), Player(3)]
        game = SimpleJong(players, tile_copies=max(1, int(args.tile_copies)))
        games_played += 1

        safety_rounds = 0
        while not game.is_game_over() and (game.tiles or game.last_discarded_tile is not None) and safety_rounds < 500:
            safety_rounds += 1
            # If there is a pending discard, check ron states and resolve reactions
            if game.last_discarded_tile is not None and game.last_discard_player is not None:
                for opp in range(4):
                    if opp == game.last_discard_player:
                        continue
                    rs = game.get_game_perspective(opp)
                    if rs.can_ron():
                        ron_count += 1
                        rec = run_mcts_for_player(game, opp, args.simulations)
                        results.append(('ron', rec, rs))
                # Let the engine resolve reactions
                if game._resolve_reactions_after_discard():
                    break
                continue

            # Draw for current player
            game._draw_for_current_if_needed()
            pid = game.current_player_idx
            gs = game.get_game_perspective(pid)
            if gs.can_tsumo():
                tsumo_count += 1
                rec = run_mcts_for_player(game, pid, args.simulations)
                results.append(('tsumo', rec, gs))
                # Apply tsumo to end the game
                game.step(pid, Tsumo())
                break

            # Choose a discard using heuristic player
            legal = game.legal_moves(pid)
            hp = SimpleHeuristicsPlayer(pid)
            move = hp.select_action(game, pid, legal)
            if move is None:
                break
            game.step(pid, move)
            # Loop continues; reactions, if any, will be handled at top

    summarize(results, verbose=args.verbose)
    print(f"Games played: {games_played}")
    print(f"Obvious opportunities found — Ron: {ron_count}, Tsumo: {tsumo_count}")


if __name__ == "__main__":
    main()


