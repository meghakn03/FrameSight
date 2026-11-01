"""Dependency-free, bounded session statistics."""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict, dataclass
from threading import RLock


@dataclass(frozen=True)
class Detection:
    category: str
    confidence: float
    track_id: int | None
    box: tuple


class Statistics:
    def __init__(self, history_limit=1800, detection_limit=50000):
        self.lock = RLock()
        self.history = deque(maxlen=history_limit)
        self.detections = deque(maxlen=detection_limit)
        self.tracks = {}
        self.observations = Counter()
        self.frames = 0
        self.elapsed = 0.0
        self.current = Counter()

    def update(self, detections, timestamp, latency):
        with self.lock:
            self.frames += 1
            self.elapsed += latency
            self.current = Counter(d.category for d in detections)
            for d in detections:
                self.observations[d.category] += 1
                # Assign each ID to its first observed category to avoid double counting.
                if d.track_id is not None:
                    self.tracks.setdefault(d.track_id, d.category)
                row = asdict(d)
                row.update(frame=self.frames, time_seconds=round(timestamp, 3))
                self.detections.append(row)
            self.history.append(dict(frame=self.frames, time_seconds=round(timestamp, 3),
                                     objects=len(detections)))

    def snapshot(self):
        with self.lock:
            return dict(frames=self.frames, fps=self.frames / self.elapsed if self.elapsed else 0,
                        current=dict(self.current), unique=dict(Counter(self.tracks.values())),
                        observations=dict(self.observations), history=list(self.history),
                        detections=list(self.detections))
