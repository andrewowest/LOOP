from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch


@dataclass
class LongTermMemoryConfig:
    storage_path: Path = Path("memory/persistent/knowledge.jsonl")
    max_entries: int = 10000
    device: Optional[torch.device] = None

    def __post_init__(self) -> None:
        self.storage_path = Path(self.storage_path)


class LongTermMemory:
    """Append-only JSONL store for persistent, high-importance memories."""

    def __init__(self, config: LongTermMemoryConfig) -> None:
        self.config = config
        self.storage_path = Path(config.storage_path)
        self.device = config.device or torch.device("cpu")
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._line_count = self._count_existing_lines()

    def consolidate(self, entries: Iterable[Dict[str, str]]) -> None:
        entries = list(entries)
        if not entries:
            return

        with self.storage_path.open("a", encoding="utf-8") as handle:
            for entry in entries:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

        self._line_count += len(entries)
        if self.config.max_entries > 0 and self._line_count > self.config.max_entries:
            self._truncate()

    def load_recent(self, limit: int = 32) -> List[Dict[str, str]]:
        if not self.storage_path.exists():
            return []
        result: List[Dict[str, str]] = []
        for record in self._read_records()[-limit:]:
            try:
                result.append(json.loads(record))
            except json.JSONDecodeError:
                continue
        return result

    def _count_existing_lines(self) -> int:
        if not self.storage_path.exists():
            return 0
        return len(self._read_records())

    def _truncate(self) -> None:
        records = self._read_records()
        trimmed = records[-self.config.max_entries :]
        self.storage_path.write_text("\n".join(trimmed) + "\n", encoding="utf-8")
        self._line_count = len(trimmed)

    def _read_records(self) -> List[str]:
        # Split only on "\n". str.splitlines() also splits on \r, \v, \f, and
        # the Unicode line/paragraph separators, which can appear unescaped
        # inside JSON string values and would corrupt records on read-back.
        text = self.storage_path.read_text(encoding="utf-8")
        if not text:
            return []
        records = text.split("\n")
        if records and records[-1] == "":
            records.pop()
        return records
