from __future__ import annotations
from typing import Dict, Any, List
import os, json
from .types import EventLog, now_ts


class EventLogger:
    def __init__(self, base_dir: str = "./logs"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self.path = os.path.join(self.base_dir, "events.jsonl")

    def append(self, event: EventLog):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event.__dict__, ensure_ascii=False) + "\n")

    def read_all(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.path):
            return []
        with open(self.path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]
