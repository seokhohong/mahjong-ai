#!/usr/bin/env python3
import sys
import os
import time
import threading
import unittest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.parallel_jong import ProgressReporter, ParallelJong  # type: ignore
from core.game import SimpleJong, Player  # type: ignore


class _CaptureStream:
    def __init__(self):
        self._lock = threading.Lock()
        self._parts = []

    def write(self, s: str) -> None:
        with self._lock:
            self._parts.append(s)

    def flush(self) -> None:
        return

    def getvalue(self) -> str:
        with self._lock:
            return ''.join(self._parts)


class TestProgressOutput(unittest.TestCase):
    def test_progress_reporter_prints_initial_line_immediately(self):
        cap = _CaptureStream()
        pr = ProgressReporter(total=5, desc="TestPR", interval_sec=0.05, stream=cap)
        start_t = time.time()
        pr.start()
        # No sleep needed because ProgressReporter.start() prints once immediately
        first = cap.getvalue()
        self.assertIn('TestPR:', first)
        self.assertTrue(('0/5' in first) or ('1/5' in first))  # allow possible race if update lands fast

        pr.update(3)
        # Wait a bit longer than interval to allow background refresh
        time.sleep(0.08)
        mid = cap.getvalue()
        self.assertIn('\r', mid)  # carriage-return update present
        # Close prints final newline
        pr.close()
        out = cap.getvalue()
        self.assertIn('\n', out)
        self.assertIn('5', out)  # total appears

        # Sanity: ensure the initial write happened near start (i.e., not only at close)
        self.assertLess(time.time() - start_t, 10.0)

    def test_parallel_jong_writes_progress_to_stdout(self):
        # Capture stdout because ParallelJong's ProgressReporter defaults to stdout
        cap = _CaptureStream()
        orig_stdout = sys.stdout
        sys.stdout = cap
        try:
            games = [SimpleJong([Player(i) for i in range(4)], tile_copies=2) for _ in range(3)]
            pj = ParallelJong(games, threads=2, progress_desc='UnitTestProgress')
            pj.run(show_progress=True)
        finally:
            sys.stdout = orig_stdout

        out = cap.getvalue()
        # Expect at least one progress line with the description; final line ends with \n
        self.assertIn('UnitTestProgress:', out)
        self.assertRegex(out, r"\d/3")
        self.assertTrue(out.endswith('\n'))

    def test_compete_play_n_games_emits_progress(self):
        # Ensure root is on path to import compete module
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        if root not in sys.path:
            sys.path.insert(0, root)
        import compete  # type: ignore

        cap = _CaptureStream()
        orig_stdout = sys.stdout
        sys.stdout = cap
        try:
            # Small run to be quick; models are '-' so baseline players are used
            compete.play_n_games(2, '-,-,-,-')
        finally:
            sys.stdout = orig_stdout

        out = cap.getvalue()
        self.assertIn('Competing:', out)
        self.assertIn('/2', out)
        self.assertTrue(out.endswith('\n'))


if __name__ == '__main__':
    unittest.main(verbosity=2)


