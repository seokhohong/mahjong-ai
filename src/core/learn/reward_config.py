from __future__ import annotations

"""
Centralized reward configuration for policy-gradient training and metrics.

Adjust these constants to implement asymmetric rewards without creating
circular imports between training and network modules.
"""

# Reward assigned to actions that led to a win
WIN_REWARD: float = 1.0

# Reward (penalty) assigned to actions that led to a loss
# Use a small-magnitude negative value for asymmetric training, e.g., -0.05
LOSS_REWARD: float = -0.01

# Reward assigned to neutral outcomes (neither win nor loss)
NEUTRAL_REWARD: float = 0.0


