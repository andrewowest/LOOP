from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from loop_core.memory import AssociativeMemory, LongTermMemory, WorkingMemory
from loop_core.controller import Controller


@dataclass
class PersonaProfile:
    persona_facts: List[str] = field(default_factory=list)
    tone_guidelines: List[str] = field(default_factory=list)
    response_guidelines: List[str] = field(default_factory=list)
    hypnotize_directive: Optional[str] = None
    working_memory_slots: Optional[int] = None
    working_memory_decay: Optional[float] = None
    prior_conservatism: Optional[float] = None
    logo: Optional[str] = None

    def create_coordinator(
        self,
        working_memory: WorkingMemory,
        associative_memory: AssociativeMemory,
        long_term_memory: LongTermMemory,
        controller: Controller,
    ):
        from loop_core.rag import PersonaCoordinator

        return PersonaCoordinator(
            persona=self,
            working_memory=working_memory,
            associative_memory=associative_memory,
            long_term_memory=long_term_memory,
            controller=controller,
        )


SECTION_MAP: Dict[str, str] = {
    "meta overview": "meta",
    "voice, tone & rhetoric": "tone",
    "interaction protocols": "interaction",
    "bayesian weights & numerics": "numerics",
    "quick-reference": "cheatsheet",
}


def _normalize_section_name(line: str) -> Optional[str]:
    match = re.match(r"^\s*(?:\d+\)\s+)?([A-Z].+?)\s*$", line)
    if not match:
        return None
    return match.group(1).strip().lower()


def load_persona_profile(path: Path) -> PersonaProfile:
    profile = PersonaProfile()
    if not path.exists():
        return profile

    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    current_section = None
    hypnotize_pattern = re.compile(r"!hypnotize=\"(.+?)\"", re.IGNORECASE)

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        normalized = _normalize_section_name(line)
        if normalized and any(normalized.startswith(prefix) for prefix in SECTION_MAP):
            current_section = normalized
            continue

        hypo_match = hypnotize_pattern.search(line)
        if hypo_match:
            profile.hypnotize_directive = hypo_match.group(1).strip()
            continue

        if not line.startswith("- "):
            continue

        entry = line[2:].strip()
        if not entry:
            continue

        if current_section is None:
            continue

        if "meta" in current_section:
            profile.persona_facts.append(entry)
        elif "voice" in current_section:
            profile.tone_guidelines.append(entry)
        elif "interaction" in current_section or "quick-reference" in current_section:
            profile.response_guidelines.append(entry)
        elif "bayesian" in current_section:
            if entry.startswith("WM_slots_default"):
                try:
                    profile.working_memory_slots = int(entry.split("=")[1].strip())
                except (IndexError, ValueError):
                    pass
            elif entry.startswith("WM_decay_rate_base"):
                try:
                    profile.working_memory_decay = float(entry.split("=")[1].strip().split()[0])
                except (IndexError, ValueError):
                    pass
            elif entry.startswith("prior_conservatism"):
                try:
                    profile.prior_conservatism = float(entry.split("=")[1].strip())
                except (IndexError, ValueError):
                    pass

    def _dedupe(items: List[str]) -> List[str]:
        seen = set()
        result: List[str] = []
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            result.append(item)
        return result

    profile.persona_facts = _dedupe(profile.persona_facts)
    profile.tone_guidelines = _dedupe(profile.tone_guidelines)
    profile.response_guidelines = _dedupe(profile.response_guidelines)

    return profile
