# Full Riichi Mah-jong Project

This directory (`src/core_full`) houses a separate, clean-room implementation of the full Riichi mahjong ruleset.

Key notes:
- Separate from `SimpleJong` and its AI experiments. We will keep training/compute against `SimpleJong` until it is proven stable.
- Objective: Implement the complete rules and flow for a standard hanchan.

Scope (initial milestones):
1) Tiles: include all suits (manzu, pinzu, souzu) and all honors (winds, dragons).
2) Hand structure: 14-tile hands; winning hands require 4 sets + 1 pair.
3) Yaku (initial set): Riichi, Tanyao (all simples), Yakuhai (relevant honor triplet), Honitsu (half flush), Chinitsu (full flush).
4) Rounds: full hanchan (East 1-4, South 1-4), dealer repeats on win, seat winds rotate.
5) Scoring: start 25,000 points each; rank-based outcomes. Temporary placeholder: treat completed hands as 2,000 points until full scoring table is implemented.

File layout:
- `src/core_full/game.py`: clone of `src/core/game.py` with class name renamed to `FullRiichi`. This will diverge significantly as we implement full rules.
- `src/core_full/__init__.py`: makes the package importable.

Working approach:
- Keep APIs similar to `SimpleJong` where practical to ease testing.
- Build comprehensive tests alongside new rules.
- Do not integrate with the existing AI until the simple ruleset is validated end-to-end.

— Notes to future self: If context is reset, search for this filename `README_FULL_RIICHI.md` in repo root.
