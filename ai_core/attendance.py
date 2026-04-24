from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from threading import Lock
from typing import Dict, List, Optional, Protocol

try:
    from redis import Redis
except Exception:  # pragma: no cover - optional dependency
    Redis = None


@dataclass
class AttendanceRecord:
    name: str
    timestamp: str


class AttendanceStateStore(Protocol):
    def should_mark(self, name: str, ttl_seconds: int, now: datetime) -> bool:
        ...


class InMemoryAttendanceState:
    def __init__(self) -> None:
        self._last_marked: Dict[str, datetime] = {}
        self._lock = Lock()

    def should_mark(self, name: str, ttl_seconds: int, now: datetime) -> bool:
        with self._lock:
            previous = self._last_marked.get(name)
            if previous and (now - previous) < timedelta(seconds=ttl_seconds):
                return False
            self._last_marked[name] = now
            return True


class RedisAttendanceState:
    def __init__(self, redis_client: Redis, key_prefix: str = "attendance:last") -> None:
        if Redis is None:
            raise RuntimeError("redis package is required for RedisAttendanceState.")
        self.redis = redis_client
        self.key_prefix = key_prefix

    def should_mark(self, name: str, ttl_seconds: int, now: datetime) -> bool:
        key = f"{self.key_prefix}:{name}"
        # SET NX EX atomically enforces distributed de-duplication across instances.
        was_set = self.redis.set(name=key, value=now.isoformat(timespec="seconds"), ex=ttl_seconds, nx=True)
        return bool(was_set)


class AttendanceLogger:
    """AI-side attendance tracker with duplicate suppression window."""

    def __init__(
        self,
        duplicate_window_seconds: int = 60,
        state_store: AttendanceStateStore | None = None,
    ) -> None:
        self.duplicate_window_seconds = duplicate_window_seconds
        self.state_store = state_store or InMemoryAttendanceState()
        self.records: List[AttendanceRecord] = []
        self._records_lock = Lock()

    def mark_present(self, name: str, timestamp: Optional[datetime] = None) -> bool:
        if name == "Unknown":
            return False

        now = timestamp or datetime.now()
        if not self.state_store.should_mark(name, self.duplicate_window_seconds, now):
            return False

        with self._records_lock:
            self.records.append(
                AttendanceRecord(name=name, timestamp=now.isoformat(timespec="seconds"))
            )
        return True

    def get_records(self) -> List[Dict[str, str]]:
        with self._records_lock:
            return [asdict(record) for record in self.records]

    def summary(self) -> str:
        with self._records_lock:
            if not self.records:
                return "Attendance log is empty."
            lines = ["Attendance Log:"]
            for record in self.records:
                lines.append(f"- {record.name} @ {record.timestamp}")
        return "\n".join(lines)
