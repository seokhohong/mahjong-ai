#!/usr/bin/env python3
"""
Analyze policy convergence on obvious states (tsumo/ron) using MCTS.

This script loads serialized game states from a generation data directory,
identifies obvious win opportunities (tsumo) and ron opportunities, runs MCTS
from those states, and measures how often the recommended action matches the
expected obvious action.

Usage:
  python3 analyze_policy_convergence.py \
    --data_dir training_data/generation_0/data \
    --limit 100 \
    --simulations 300
"""

import os
import sys
import argparse
from typing import List, Tuple

# Repo paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from core.game import (  # type: ignore
    Player,
    AIPlayer,
    GameState,
    Tsumo,
    Ron,
    CalledSet,
)

# Import dataset loader/rehydration utilities
from train_pq_from_generation import (  # type: ignore
    load_training_samples,
    rehydrate_state,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze obvious policy convergence with MCTS")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to generation data directory (e.g., training_data/generation_0/data)")
    parser.add_argument("--limit", type=int, default=100, help="Number of samples to analyze")
    parser.add_argument("--simulations", type=int, default=300, help="MCTS simulations per state")
    parser.add_argument("--verbose", action="store_true", help="Print details for mismatches")
    return parser.parse_args()


def build_player_from_state(gs: GameState) -> Player:
    """Construct a temporary Player mirroring the GameState player's hand and called sets for checks."""
    p = Player(gs.player_id)
    p.hand = gs.player_hand.copy()
    # CalledSet is dataclass in same module, so direct copy is fine
    p.called_sets = [cs for cs in gs.called_sets.get(gs.player_id, [])]
    return p


def detect_obvious_action(gs: GameState) -> Tuple[str, bool]:
    """
    Return (expected_action, found) where expected_action is one of:
      - 'tsumo' if current player can immediately tsumo
      - 'ron'   if current player can immediately ron
      - ''      if no obvious action found
    """
    player = build_player_from_state(gs)

    # Tsumo: immediate win with current hand
    if player.can_win():
        return 'tsumo', True

    # Ron: last discard exists and belongs to another player, and winning with it
    if gs.last_discarded_tile is not None and gs.last_discard_player is not None and gs.last_discard_player != gs.player_id:
        if player.can_ron(gs.last_discarded_tile):
            return 'ron', True

    return '', False


def mcts_recommendation(gs: GameState, simulations: int) -> str:
    """Run MCTS from the given GameState and return the recommended action type as a string."""
    ai = AIPlayer(gs.player_id, simulation_count=simulations)
    # Call internal MCTS directly to avoid the short-circuiting of AIPlayer.play for tsumo/ron
    action = ai._mcts_search(gs)
    if action is None:
        return 'none'
    if isinstance(action, Tsumo):
        return 'tsumo'
    if isinstance(action, Ron):
        return 'ron'
    # Other actions map to their class names
    return action.__class__.__name__.lower()


def summarize_results(results: List[Tuple[str, str, GameState]], verbose: bool):
    total = len(results)
    tsumo_cases = [r for r in results if r[0] == 'tsumo']
    ron_cases = [r for r in results if r[0] == 'ron']

    tsumo_correct = sum(1 for exp, rec, _ in tsumo_cases if rec == 'tsumo')
    ron_correct = sum(1 for exp, rec, _ in ron_cases if rec == 'ron')

    print(f"Analyzed obvious states: {total}")
    print(f"  Tsumo cases: {len(tsumo_cases)} | correct: {tsumo_correct} | acc: { (tsumo_correct/len(tsumo_cases)) if tsumo_cases else 0.0:.2f}")
    print(f"  Ron   cases: {len(ron_cases)} | correct: {ron_correct} | acc: { (ron_correct/len(ron_cases)) if ron_cases else 0.0:.2f}")

    if verbose:
        for exp, rec, gs in results:
            if exp != rec:
                print("- Mismatch:")
                print(f"  Expected: {exp} | Recommended: {rec}")
                # Small human-friendly snapshot
                hand = ','.join(str(t) for t in gs.player_hand)
                last = str(gs.last_discarded_tile) if gs.last_discarded_tile else 'None'
                print(f"  Player {gs.player_id} hand: [{hand}] | last_discard: {last} from {gs.last_discard_player}")


def main():
    args = parse_args()
    data_dir = os.path.abspath(args.data_dir)
    print(f"Loading samples from: {data_dir}")

    # Load and rehydrate up to limit samples
    game_states, _, _ = load_training_samples(data_dir, limit_samples=args.limit)

    results: List[Tuple[str, str, GameState]] = []
    considered = 0

    for gs in game_states:
        expected, found = detect_obvious_action(gs)
        if not found:
            continue
        considered += 1
        recommended = mcts_recommendation(gs, simulations=args.simulations)
        results.append((expected, recommended, gs))

    summarize_results(results, verbose=args.verbose)
    print(f"Total obvious states considered: {considered}")


if __name__ == "__main__":
    main()


