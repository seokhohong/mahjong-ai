"""Lightweight init to avoid importing heavy backends when not needed.

Exposes `PurePolicyNetwork` if its dependencies (PyTorch) are available, but
does not fail if they are not. This allows modules like `pure_policy_dataset`
to be imported in environments without torch.
"""

__all__ = []  # populated conditionally

try:  # pragma: no cover - tiny import guard
    from .pure_policy import PurePolicyNetwork  # type: ignore
    __all__.append('PurePolicyNetwork')
except Exception:
    # Leave PurePolicyNetwork unavailable when torch (or other deps) missing
    PurePolicyNetwork = None  # type: ignore



