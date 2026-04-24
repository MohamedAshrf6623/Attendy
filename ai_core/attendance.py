from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional


@dataclass
class AttendanceRecord:
    name: str
    timestamp: str


class AttendanceLogger:
    """AI-side attendance tracker with duplicate suppression window."""

    def __init__(self, duplicate_window_seconds: int = 60) -> None:
        self.duplicate_window = timedelta(seconds=duplicate_window_seconds)
        self.last_marked: Dict[str, datetime] = {}
        self.records: List[AttendanceRecord] = []

    def mark_present(self, name: str, timestamp: Optional[datetime] = None) -> bool:
        if name == "Unknown":
            return False

        now = timestamp or datetime.now()
        previous = self.last_marked.get(name)
        if previous and (now - previous) < self.duplicate_window:
            return False

        self.last_marked[name] = now
        self.records.append(AttendanceRecord(name=name, timestamp=now.isoformat(timespec="seconds")))
        return True

    def get_records(self) -> List[Dict[str, str]]:
        return [asdict(record) for record in self.records]

    def summary(self) -> str:
        if not self.records:
            return "Attendance log is empty."
        lines = ["Attendance Log:"]
        for record in self.records:
            lines.append(f"- {record.name} @ {record.timestamp}")
        return "\n".join(lines)
