from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, Optional

import torch

from .associative_memory import AssociativeMemory
from .long_term import LongTermMemory
from .working_memory import WorkingMemory


class MemoryConsolidator:
    """Promotes content across memory tiers based on importance and mention frequency."""

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
        # User turns carry more weight than assistant turns: the user expresses
        # intent, the assistant restates. user_base_hint sits just above the
        # default wm_to_am_threshold so a single declarative user turn is
        # promotable; repeated mentions then compound via mention_bonus.
        self.user_base_hint = float(config.get("user_base_hint", 0.6))
        self.assistant_base_hint = float(config.get("assistant_base_hint", 0.3))
        self.mention_bonus = float(config.get("mention_bonus", 0.15))

        self.mention_counts: Dict[str, int] = defaultdict(int)
        self.persisted_keys = set()

    def process_turn(self, user_text: str, ai_text: str) -> None:
        self._ingest_event(user_text, "user", self.user_base_hint, force_long_term=False)
        self._ingest_event(ai_text, "assistant", self.assistant_base_hint, force_long_term=False)

    def record_hypnotize(self, payload: str) -> None:
        self._ingest_event(payload, "hypnotize", self.hypnotize_bonus, force_long_term=True)

    def absorb_persona(self, persona) -> None:
        """Ingest a PersonaProfile's hypnotize directives as forced LTM entries.

        Use this after load_persona_profile() so directives written into a
        profile file behave identically to runtime !hypnotize="..." commands.
        """
        for directive in getattr(persona, "hypnotize_directives", None) or []:
            self.record_hypnotize(directive)

    def _ingest_event(self, text: str, kind: str, base_hint: float, force_long_term: bool) -> None:
        normalized = (text or "").strip()
        if not normalized:
            return

        key = normalized.lower()
        self.mention_counts[key] += 1
        mentions = self.mention_counts[key]

        importance_hint = self._importance_hint(normalized, base_hint)
        if mentions > 1:
            importance_hint = min(0.9, importance_hint + self.mention_bonus * (mentions - 1))
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
        if "!" in text or _is_shouting(text):
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
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.long_term_memory.consolidate([entry])
        if key:
            self.persisted_keys.add(key)


def _is_shouting(text: str) -> bool:
    """Heuristic: at least 4 uppercase letters and >30% of the text is caps."""
    upper = sum(1 for ch in text if ch.isupper())
    return upper >= 4 and upper > len(text) * 0.3
