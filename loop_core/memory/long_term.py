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


class LongTermMemory:
    def __init__(self, config: LongTermMemoryConfig) -> None:
        self.config = config
        self.storage_path = config.storage_path
        self.device = config.device or torch.device("cpu")
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

    def consolidate(self, entries: Iterable[Dict[str, str]]) -> None:
        entries = list(entries)
        if not entries:
            return

        with self.storage_path.open("a", encoding="utf-8") as handle:
            for entry in entries:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

        self._truncate_if_needed()

    def _truncate_if_needed(self) -> None:
        if self.config.max_entries <= 0:
            return
        if not self.storage_path.exists():
            return
        lines = self.storage_path.read_text(encoding="utf-8").splitlines()
        if len(lines) <= self.config.max_entries:
            return
        trimmed = lines[-self.config.max_entries :]
        self.storage_path.write_text("\n".join(trimmed) + "\n", encoding="utf-8")

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
