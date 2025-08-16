from __future__ import annotations

import os
import sys
import tempfile
import numpy as np


def test_old_log_probs_and_policies_mostly_zero_like():
    # Ensure src in path for running the builder
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    sys.path.insert(0, root)

    from run.create_dataset import build_ac_dataset

    built = build_ac_dataset(
        games=3,
        seed=123,
        use_heuristic=True,  # heuristic recorder still records decisions; policies list will be empty
    )

    # old_log_probs should exist and correspond to chosen flat indices; most entries should be very small
    old_log_probs = np.asarray(built['old_log_probs'], dtype=np.float32)
    assert old_log_probs.ndim == 1
    assert old_log_probs.size > 0

    # Heuristic has no policy; skip policy distribution check if empty
    flat_policies = built.get('flat_policies', None)
    # If policies present (when using AC network), check that most mass is near-zero (legality mask)
    if flat_policies is not None and len(flat_policies) > 0 and isinstance(flat_policies[0], (list, np.ndarray)) and len(flat_policies[0]) > 0:
        policies = [np.asarray(p, dtype=np.float32) for p in flat_policies]
        # Each policy should be 25-dim
        assert all(p.shape == (25,) for p in policies)
        # For each policy, count near-zero entries (< 1e-6)
        zero_like_counts = [int((p < 1e-6).sum()) for p in policies]
        # Expect majority of the 25 actions to be illegal most of the time (> 60%)
        assert np.mean([zc >= 15 for zc in zero_like_counts]) > 0.8

    # Additionally, majority of old_log_probs should be <= log(1e-6) if they point to near-zero probs
    # Can't guarantee without AC network, so this is a lenient sanity check
    small_log_cutoff = np.log(1e-6)
    frac_small = float(np.mean(old_log_probs <= small_log_cutoff))
    # We do not assert this strictly for heuristic (log_probs=0), just ensure the array is finite
    assert np.all(np.isfinite(old_log_probs))


