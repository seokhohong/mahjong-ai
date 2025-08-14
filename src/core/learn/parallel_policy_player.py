from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import threading
import time
import numpy as np

from ..game import (
    GamePerspective,
    Tile,
    Tsumo,
    Ron,
    Discard,
    Pon,
    Chi,
    Reaction,
    PassCall,
)

from .pure_policy_player import PurePolicyPlayer


class _InferenceRequest:
    """Single-sample inference request with synchronization primitive."""

    def __init__(self, hand_idx: np.ndarray, disc_idx: np.ndarray, called_idx: np.ndarray, game_state: np.ndarray):
        self.hand_idx = hand_idx
        self.disc_idx = disc_idx
        self.called_idx = called_idx
        self.game_state = game_state
        self._event = threading.Event()
        self.result: Optional[np.ndarray] = None

    def set_result(self, probs: np.ndarray) -> None:
        self.result = probs
        self._event.set()

    def wait_result(self) -> np.ndarray:
        self._event.wait()
        assert self.result is not None
        return self.result


class ParallelPolicyPredictor:
    """
    Thread-safe batching predictor for PurePolicyNetwork.model.predict.

    Collects requests and flushes either when `max_batch_size` is reached or
    when `max_wait_ms` has elapsed since the first pending request.
    """

    def __init__(self, network: Any, max_batch_size: int = 32, max_wait_ms: int = 2):
        self._network = network
        self._max_batch_size = max(1, int(max_batch_size))
        self._max_wait_ms = max(0, int(max_wait_ms))
        self._cv = threading.Condition()
        self._queue: List[_InferenceRequest] = []
        self._stopped = False
        self._worker = threading.Thread(target=self._run, name="ParallelPolicyPredictor", daemon=True)
        self._worker.start()

    def stop(self) -> None:
        with self._cv:
            self._stopped = True
            self._cv.notify_all()
        self._worker.join(timeout=1.0)

    def predict_single(self, hand_idx: np.ndarray, disc_idx: np.ndarray, called_idx: np.ndarray, game_state: np.ndarray) -> np.ndarray:
        req = _InferenceRequest(hand_idx, disc_idx, called_idx, game_state)
        with self._cv:
            self._queue.append(req)
            # Wake worker to possibly flush
            self._cv.notify()
        return req.wait_result()

    def _run(self) -> None:
        """Background worker that flushes requests as batches."""
        while True:
            with self._cv:
                # Wait until there is at least one request or we are stopping
                while not self._stopped and not self._queue:
                    self._cv.wait()
                if self._stopped:
                    return

                # We have at least one request. Determine deadline for timeout-based flush.
                flush_deadline = time.time() + (self._max_wait_ms / 1000.0)
                # Wait for either batch to fill or timeout
                while not self._stopped and len(self._queue) < self._max_batch_size:
                    now = time.time()
                    remaining = flush_deadline - now
                    if remaining <= 0:
                        break
                    self._cv.wait(timeout=remaining)
                    if self._queue and len(self._queue) >= self._max_batch_size:
                        break
                if self._stopped:
                    return

                # Pop up to max_batch_size requests to form a batch
                batch: List[_InferenceRequest] = self._queue[: self._max_batch_size]
                del self._queue[: len(batch)]

            # Prepare batched numpy inputs
            hands = np.stack([r.hand_idx for r in batch], axis=0)
            discs = np.stack([r.disc_idx for r in batch], axis=0)
            called = np.stack([r.called_idx for r in batch], axis=0)
            gss = np.stack([r.game_state for r in batch], axis=0)

            # Run network predict once for the batch
            probs_batch = self._network.model.predict([hands, discs, called, gss], verbose=0)

            # Dispatch results back to individual requests
            for r, probs in zip(batch, probs_batch):
                r.set_result(np.asarray(probs))


_predictor_registry_lock = threading.Lock()
_predictor_registry: Dict[int, ParallelPolicyPredictor] = {}


def get_or_create_predictor(network: Any, max_batch_size: int = 32, max_wait_ms: int = 2) -> ParallelPolicyPredictor:
    key = id(network)
    with _predictor_registry_lock:
        pred = _predictor_registry.get(key)
        if pred is None:
            pred = ParallelPolicyPredictor(network, max_batch_size=max_batch_size, max_wait_ms=max_wait_ms)
            _predictor_registry[key] = pred
        return pred


class ParallelPolicyPlayer(PurePolicyPlayer):
    """
    Drop-in replacement for PurePolicyPlayer that performs batched GPU inference.

    Semantics are identical to PurePolicyPlayer; only the inference path is
    parallelized and batched across threads/games.
    """

    def __init__(self, player_id: int, network: Any, predictor: Optional[ParallelPolicyPredictor] = None, max_batch_size: int = 32, max_wait_ms: int = 2):
        super().__init__(player_id, network)
        self._predictor = predictor or get_or_create_predictor(network, max_batch_size=max_batch_size, max_wait_ms=max_wait_ms)

    # Override only the inference call; reuse all selection/mapping logic from base class
    def predict_policy_probs(self, gs: GamePerspective) -> np.ndarray:  # type: ignore[override]
        hand_idx, disc_idx, called_idx, game_state = self._encode_inputs(gs)
        probs = self._predictor.predict_single(hand_idx, disc_idx, called_idx, game_state)
        return probs


