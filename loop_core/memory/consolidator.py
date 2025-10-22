from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from typing import Dict, Optional

import torch

from .associative_memory import AssociativeMemory
from .long_term import LongTermMemory
from .working_memory import WorkingMemory


class MemoryConsolidator:
    def __init__(
        self,
        working_memory: WorkingMemory,
        associative_memory: AssociativeMemory,
        long_term_memory: LongTermMemory,
        encoder,
        config: Optional[Dict[str, float]] = None,
    ) -> None:
        self.working_memory = working_memory
        self.associative_memory = associative_memory
        self.long_term_memory = long_term_memory
        self.encoder = encoder
        self.device = working_memory.device

        config = config or {}
        self.wm_to_am_threshold = float(config.get("wm_to_am_threshold", 0.55))
        self.am_to_ltm_threshold = float(config.get("am_to_ltm_threshold", 0.75))
        self.min_mentions = int(config.get("min_mentions", 2))
        self.hypnotize_bonus = float(config.get("hypnotize_bonus", 0.95))
        self.emphasis_bonus = float(config.get("emphasis_bonus", 0.25))

        self.mention_counts: Dict[str, int] = defaultdict(int)
        self.persisted_keys = set()

    def process_turn(self, user_text: str, ai_text: str) -> None:
        self._ingest_event(user_text, "user", 0.0, force_long_term=False)
        self._ingest_event(ai_text, "assistant", 0.0, force_long_term=False)

    def record_hypnotize(self, payload: str) -> None:
        self._ingest_event(payload, "hypnotize", self.hypnotize_bonus, force_long_term=True)

    def _ingest_event(self, text: str, kind: str, base_hint: float, force_long_term: bool) -> None:
        normalized = (text or "").strip()
        if not normalized:
            return

        key = normalized.lower()
        self.mention_counts[key] += 1
        mentions = self.mention_counts[key]

        importance_hint = self._importance_hint(normalized, base_hint)
        metadata = {
            "type": kind,
            "text": normalized,
            "key": key,
            "mentions": mentions,
            "importance_hint": importance_hint,
        }
        if force_long_term:
            metadata["force_long_term"] = True

        embedding = self._encode(normalized)
        self.working_memory.update(embedding, metadata)
        self._evaluate_last_slot(force_long_term)

    def _encode(self, text: str) -> torch.Tensor:
        vector = self.encoder.encode([text], convert_to_numpy=True)
        tensor = torch.tensor(vector[0], dtype=torch.float32, device=self.device)
        return tensor

    def _importance_hint(self, text: str, base_hint: float) -> float:
        hint = base_hint
        if not text:
            return hint
        if "!" in text:
            hint = max(hint, self.emphasis_bonus)
        uppercase = sum(1 for ch in text if ch.isupper())
        if uppercase and uppercase >= 4 and uppercase > len(text) * 0.3:
            hint = max(hint, self.emphasis_bonus)
        return hint

    def _evaluate_last_slot(self, forced: bool) -> None:
        index = self.working_memory.last_slot_index
        if index is None:
            return
        slot = self.working_memory.storage[index]
        if slot is None:
            return

        importance = float(self.working_memory.importance[index].item())
        metadata = slot["metadata"] or {}
        embedding = slot["embedding"]

        to_associative = importance >= self.wm_to_am_threshold or forced
        if not to_associative:
            return

        self.associative_memory.write(embedding, dict(metadata))

        mentions = metadata.get("mentions", 0)
        force_long_term = forced or metadata.get("force_long_term", False)
        meets_long_term = importance >= self.am_to_ltm_threshold and mentions >= self.min_mentions
        if not (force_long_term or meets_long_term):
            return

        key = metadata.get("key")
        if key and key in self.persisted_keys and not force_long_term:
            return

        entry = {
            "text": metadata.get("text", ""),
            "type": metadata.get("type", ""),
            "mentions": mentions,
            "importance": importance,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        self.long_term_memory.consolidate([entry])
        if key:
            self.persisted_keys.add(key)
