from __future__ import annotations

import sys
import time
import threading
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed


class ProgressReporter:
    """Thread-safe progress printer for parallel execution.

    Prints a single updating line to stderr with throughput and ETA.
    """

    def __init__(self, total: int, desc: str = "Progress", interval_sec: float = 0.5, stream=None):
        self.total = max(0, int(total))
        self.desc = desc
        self.interval_sec = max(0.05, float(interval_sec))
        self._count = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_time = 0.0
        self._stream = stream if stream is not None else sys.stdout

    def start(self) -> None:
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._run, name="ProgressReporter", daemon=True)
        self._thread.start()
        # Print initial line immediately
        self._print_line(final=False)

    def update(self, n: int = 1) -> None:
        if n <= 0:
            return
        with self._lock:
            self._count += int(n)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_sec):
            self._print_line()

    def _print_line(self, final: bool = False) -> None:
        with self._lock:
            done = min(self._count, self.total)
        elapsed = max(1e-6, time.time() - self._start_time)
        rate = done / elapsed
        remaining = max(0, self.total - done)
        eta = (remaining / rate) if rate > 0 else 0.0
        pct = (100.0 * done / self.total) if self.total > 0 else 100.0
        msg = f"{self.desc}: {done}/{self.total} ({pct:5.1f}%) | {rate:6.1f}/s | ETA {eta:6.1f}s"
        endc = "\n" if final else "\r"
        # Write to configured stream; only silence known IO issues
        try:
            self._stream.write(msg + endc)
            self._stream.flush()
        except (BrokenPipeError, OSError):
            return

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._print_line(final=True)


class ParallelJong:
    """Run multiple `SimpleJong` games concurrently with optional progress reporting.

    This class is agnostic to the specific `Player` implementations used by the
    games. All it does is call `play_round()` on each provided game instance.
    """

    def __init__(self, games: List["SimpleJong"], threads: Optional[int] = None, num_concurrent: Optional[int] = None, progress_desc: str = "Running games"):
        if threads is None or threads <= 0:
            try:
                import os
                threads = max(1, min(len(games), (os.cpu_count() or 1)))
            except Exception:
                threads = max(1, len(games))
        self.games = list(games)
        self.threads = int(threads)
        # Number of games allowed to be in-flight at once (may exceed threads, but effective concurrency is bounded by threads)
        if num_concurrent is None or num_concurrent <= 0:
            num_concurrent = len(self.games)
        self.num_concurrent = int(max(1, num_concurrent))
        self.progress_desc = progress_desc

    def run(self, show_progress: bool = True) -> List["SimpleJong"]:
        if not self.games:
            return []
        reporter: Optional[ProgressReporter] = None
        if show_progress:
            reporter = ProgressReporter(total=len(self.games), desc=self.progress_desc)
            reporter.start()

        def _play(g: "SimpleJong") -> "SimpleJong":
            g.play_round()
            if reporter is not None:
                reporter.update(1)
            return g

        if self.threads == 1:
            for g in self.games:
                _play(g)
            if reporter is not None:
                reporter.close()
            return self.games

        with ThreadPoolExecutor(max_workers=self.threads) as ex:
            # Submit up to num_concurrent to start, then replenish as tasks complete
            in_flight = set()
            total = len(self.games)
            idx = 0
            initial = min(self.num_concurrent, total)
            for _ in range(initial):
                in_flight.add(ex.submit(_play, self.games[idx]))
                idx += 1
            while in_flight:
                for fut in as_completed(in_flight, timeout=None):
                    in_flight.remove(fut)
                    # Submit next if remaining and we want to maintain up to num_concurrent in flight
                    if idx < total:
                        in_flight.add(ex.submit(_play, self.games[idx]))
                        idx += 1
                    break  # Re-enter as_completed with updated set
        if reporter is not None:
            reporter.close()
        return self.games


