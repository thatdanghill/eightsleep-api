import asyncio
import json
import os
from bisect import bisect_left, bisect_right
from collections import defaultdict
from pathlib import Path
from statistics import median
import time
from typing import Dict, List, Optional, Tuple

from app.schema import Event


DEFAULT_STATE_FILE = Path(os.getenv("STATE_FILE_PATH", "/data/state.json"))


class AppState:
    def __init__(
        self,
        queue_maxsize: int = 10_000,
        window_seconds: int = 300,
        state_file: Optional[Path] = DEFAULT_STATE_FILE,
    ):
        # Async queue for ingestion → inference
        self.event_queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=queue_maxsize)

        self.user_windows: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
        self.window_seconds = window_seconds
        self.state_file = state_file

        # Stats
        self.ingest_requests_total = 0
        self.events_received_total = 0
        self.ingest_last_ts = 0.0
        self.inference_calls = 0
        self.queue_rejections = 0

        # Lock for thread safety
        self._lock = asyncio.Lock()

    async def record_ingest(self, batch_size: int):
        """
        Atomic update to stats to do with request count
        :param batch_size: the number of events in the batch
        """
        async with self._lock:
            self.ingest_requests_total += 1
            self.events_received_total += batch_size
            self.ingest_last_ts = time.time()

    async def increment_inference_calls(self):
        """
        Increment the number of ingest calls made
        """
        async with self._lock:
            self.inference_calls += 1

    async def increment_queue_rejections(self):
        """
        Increment the number of queue rejections
        """
        async with self._lock:
            self.queue_rejections += 1

    async def trim_user_window(
        self, user_id: str, cutoff: int
    ) -> List[Tuple[int, float]]:
        """
        Trim a user's rolling window according to a cutoff timestamp.
        Windows are assumed to be sorted, as insertion happens in sorted order, and we use the lock.
        :param user_id: the user_id
        :param cutoff: the timestamp before which to trim the window
        :return: The trimmed window
        """
        async with self._lock:
            window = self.user_windows.get(user_id)
            if not window:
                return []

            idx = bisect_left(window, (cutoff, float("-inf")))
            if idx > 0:
                del window[:idx]

            return list(window)

    async def insert_user_window(self, user_id: str, timestamp: int, score: float):
        """
        Insert a score into a user's window in order, that is, maintaining the window order by timestamp
        :param user_id: the user_id
        :param timestamp: the timestamp at which to insert
        :param score: the model output score
        """
        async with self._lock:
            window = self.user_windows[user_id]
            idx = bisect_right(window, (timestamp, float("inf")))
            window.insert(idx, (timestamp, score))

    async def median_of_medians(
        self, reference_ts: Optional[int] = None
    ) -> Optional[float]:
        """
        Calculate a median across user-level medians using a single
        cutoff point. Using one cutoff per request minimizes drift from
        the sliding window while we iterate over many users.
        """
        if reference_ts is None:
            reference_ts = int(time.time())

        cutoff = reference_ts - self.window_seconds
        medians: List[float] = []

        async with self._lock:
            windows_snapshot = {
                user_id: list(window)
                for user_id, window in self.user_windows.items()
                if window
            }

        for window in windows_snapshot.values():
            idx = bisect_left(window, (cutoff, float("-inf")))
            if idx >= len(window):
                continue

            scores = [score for _, score in window[idx:]]
            medians.append(median(scores))

        if not medians:
            return None

        return median(medians)

    async def save_to_file(self, path: Optional[Path] = None):
        """
        Save the state to a state.json file. This allows us to reload the service if it gets
        interrupted.
        :param path: a path to the state JSON file
        """
        path = path or self.state_file
        if path is None:
            return

        async with self._lock:
            data = {
                "window_seconds": self.window_seconds,
                "user_windows": {
                    user: list(window) for user, window in self.user_windows.items()
                },
                "ingest_requests_total": self.ingest_requests_total,
                "events_received_total": self.events_received_total,
                "ingest_last_ts": self.ingest_last_ts,
                "inference_calls": self.inference_calls,
                "queue_rejections": self.queue_rejections,
            }

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        # Write atomically via temp file then replace
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(data, f)
        tmp_path.replace(path)

    async def load_from_file(self, path: Optional[Path] = None):
        """
        Load the state from a file in order to pick up where we left off
        :param path: a path to the state JSON file to load from
        """
        path = path or self.state_file
        if path is None or not path.exists():
            return

        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return

        async with self._lock:
            self.window_seconds = data.get("window_seconds", self.window_seconds)
            self.user_windows = defaultdict(
                list,
                {
                    user: [(int(ts), float(score)) for ts, score in window]
                    for user, window in data.get("user_windows", {}).items()
                },
            )
            self.ingest_requests_total = data.get(
                "ingest_requests_total", self.ingest_requests_total
            )
            self.events_received_total = data.get(
                "events_received_total", self.events_received_total
            )
            self.ingest_last_ts = data.get("ingest_last_ts", self.ingest_last_ts)
            self.inference_calls = data.get("inference_calls", self.inference_calls)
            self.queue_rejections = data.get("queue_rejections", self.queue_rejections)


app_state = AppState()
