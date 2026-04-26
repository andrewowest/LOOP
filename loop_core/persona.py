from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Union

from loop_core.controller import Controller
from loop_core.memory import AssociativeMemory, LongTermMemory, WorkingMemory

if TYPE_CHECKING:
    from loop_core.rag import PersonaCoordinator


@dataclass
class PersonaProfile:
    """Configurable agent personality.

    Attributes:
        name: Speaker label used when the coordinator builds prompts.
        persona_facts: Identity / role facts about the agent.
        tone_guidelines: How the agent should sound.
        response_guidelines: How the agent should structure replies.
        hypnotize_directives: Imperative directives implanted via the
            !hypnotize="..." syntax. Treated as non-negotiable rules at
            prompt time.
        working_memory_slots / decay / prior_conservatism: Optional tuning
            overrides parsed from a profile file's "Bayesian weights"
            section.
        logo: Optional ASCII art for CLI front-ends.
    """

    name: str = "Assistant"
    persona_facts: List[str] = field(default_factory=list)
    tone_guidelines: List[str] = field(default_factory=list)
    response_guidelines: List[str] = field(default_factory=list)
    hypnotize_directives: List[str] = field(default_factory=list)
    working_memory_slots: Optional[int] = None
    working_memory_decay: Optional[float] = None
    prior_conservatism: Optional[float] = None
    logo: Optional[str] = None

    @property
    def hypnotize_directive(self) -> Optional[str]:
        # Back-compat shim for the original singular attribute.
        return self.hypnotize_directives[-1] if self.hypnotize_directives else None

    def create_coordinator(
        self,
        working_memory: WorkingMemory,
        associative_memory: AssociativeMemory,
        long_term_memory: LongTermMemory,
        controller: Controller,
    ) -> "PersonaCoordinator":
        from loop_core.rag import PersonaCoordinator

        return PersonaCoordinator(
            persona=self,
            working_memory=working_memory,
            associative_memory=associative_memory,
            long_term_memory=long_term_memory,
            controller=controller,
        )


# Persona files use a simple section-based plain-text format. A line that
# matches a section heading switches the active section; bullet lines
# ("- ...") are collected into the corresponding list. Optional numeric
# tunables live under "Bayesian weights & numerics".
_SECTION_PREFIXES: Dict[str, str] = {
    "meta overview": "facts",
    "voice, tone": "tone",
    "interaction protocols": "interaction",
    "bayesian weights": "numerics",
    "quick-reference": "interaction",
}

_HYPNOTIZE_RE = re.compile(r'!hypnotize="(.+?)"', re.IGNORECASE)
_HEADING_RE = re.compile(r"^\s*(?:\d+\)\s+)?([A-Z].+?)\s*$")


def _match_section(line: str) -> Optional[str]:
    match = _HEADING_RE.match(line)
    if not match:
        return None
    heading = match.group(1).strip().lower()
    for prefix, section in _SECTION_PREFIXES.items():
        if heading.startswith(prefix):
            return section
    return None


def _parse_numeric(entry: str) -> Optional[tuple[str, str]]:
    if "=" not in entry:
        return None
    key, _, rest = entry.partition("=")
    rest = rest.strip()
    if not rest:
        return None
    return key.strip(), rest.split()[0]


def load_persona_profile(path: Union[str, Path]) -> PersonaProfile:
    """Load a PersonaProfile from a plain-text profile file.

    Returns an empty default profile if the path does not exist, so callers
    can treat the file as optional configuration.
    """
    path = Path(path)
    profile = PersonaProfile()
    if not path.exists():
        return profile

    text = path.read_text(encoding="utf-8")
    section: Optional[str] = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        for directive in _HYPNOTIZE_RE.findall(line):
            cleaned = directive.strip()
            if cleaned and cleaned not in profile.hypnotize_directives:
                profile.hypnotize_directives.append(cleaned)

        new_section = _match_section(line)
        if new_section is not None:
            section = new_section
            continue

        if not line.startswith("- ") or section is None:
            continue
        entry = line[2:].strip()
        if not entry:
            continue

        if section == "facts":
            if entry.lower().startswith("name:"):
                profile.name = entry.split(":", 1)[1].strip() or profile.name
            else:
                profile.persona_facts.append(entry)
        elif section == "tone":
            profile.tone_guidelines.append(entry)
        elif section == "interaction":
            profile.response_guidelines.append(entry)
        elif section == "numerics":
            parsed = _parse_numeric(entry)
            if parsed is None:
                continue
            key, value = parsed
            try:
                if key == "WM_slots_default":
                    profile.working_memory_slots = int(value)
                elif key == "WM_decay_rate_base":
                    profile.working_memory_decay = float(value)
                elif key == "prior_conservatism":
                    profile.prior_conservatism = float(value)
            except ValueError:
                continue

    profile.persona_facts = _dedupe(profile.persona_facts)
    profile.tone_guidelines = _dedupe(profile.tone_guidelines)
    profile.response_guidelines = _dedupe(profile.response_guidelines)
    return profile


def _dedupe(items: List[str]) -> List[str]:
    seen: set = set()
    result: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


