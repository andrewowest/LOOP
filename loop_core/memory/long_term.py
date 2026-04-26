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

    def _count_existing_lines(self) -> int:
        if not self.storage_path.exists():
            return 0
        with self.storage_path.open("r", encoding="utf-8") as handle:
            return sum(1 for _ in handle)

    def _truncate(self) -> None:
        lines = self.storage_path.read_text(encoding="utf-8").splitlines()
        trimmed = lines[-self.config.max_entries :]
        self.storage_path.write_text("\n".join(trimmed) + "\n", encoding="utf-8")
        self._line_count = len(trimmed)

    def load_recent(self, limit: int = 32) -> List[Dict[str, str]]:
        if not self.storage_path.exists():
            return []
        lines = self.storage_path.read_text(encoding="utf-8").splitlines()
        result: List[Dict[str, str]] = []
        for line in lines[-limit:]:
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return result

    def summarize(self, encoder, limit: int = 128) -> Optional[torch.Tensor]:
        entries = self.load_recent(limit=limit)
        if not entries:
            return None
        embeddings = [encoder(entry["text"]) for entry in entries if "text" in entry]
        if not embeddings:
            return None
        stacked = torch.stack(embeddings, dim=0).to(self.device)
        return stacked.mean(dim=0)
