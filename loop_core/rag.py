from __future__ import annotations

from typing import List, Optional, Tuple

from loop_core.controller import Controller
from loop_core.memory import AssociativeMemory, LongTermMemory, WorkingMemory
from loop_core.persona import PersonaProfile


_HISTORY_MAX = 10
_HISTORY_RENDERED = 4


class PersonaCoordinator:
    """Builds prompts that combine retrieved context with persona constraints."""

    def __init__(
        self,
        persona: PersonaProfile,
        working_memory: Optional[WorkingMemory] = None,
        associative_memory: Optional[AssociativeMemory] = None,
        long_term_memory: Optional[LongTermMemory] = None,
        controller: Optional[Controller] = None,
    ) -> None:
        self.persona = persona
        self.working_memory = working_memory
        self.associative_memory = associative_memory
        self.long_term_memory = long_term_memory
        self.controller = controller
        self.conversation_history: List[Tuple[str, str]] = []

    def add_persona_fact(self, fact: str) -> None:
        fact = fact.strip()
        if fact and fact not in self.persona.persona_facts:
            self.persona.persona_facts.append(fact)

    def build_prompt(self, user_input: str, system_instruction: Optional[str] = None) -> str:
        name = self.persona.name
        if system_instruction is None:
            system_instruction = (
                f"You are {name}. Respond using the persona facts and guidelines below."
            )

        sections = [system_instruction]

        if self.persona.hypnotize_directives:
            sections.append(
                "Hard directives (non-negotiable):\n"
                + "\n".join(f"- {d}" for d in self.persona.hypnotize_directives)
            )

        sections.append("Tone guidelines:\n" + _bullets(
            self.persona.tone_guidelines, default="- Use a natural, consistent tone."
        ))
        sections.append("Response style:\n" + _bullets(
            self.persona.response_guidelines, default="- Keep responses structured and concise."
        ))
        sections.append("Persona facts:\n" + _bullets(
            self.persona.persona_facts, default="- (no extra persona overrides)"
        ))

        history_block = "\n".join(
            f"User: {user}\n{name}: {ai}"
            for user, ai in self.conversation_history[-_HISTORY_RENDERED:]
        )
        if history_block:
            sections.append("Conversation:\n" + history_block)

        sections.append(f"User: {user_input}\n{name}:")
        return "\n\n".join(sections)

    def update_conversation(self, user_input: str, ai_response: str) -> None:
        self.conversation_history.append((user_input, ai_response))
        if len(self.conversation_history) > _HISTORY_MAX:
            self.conversation_history = self.conversation_history[-_HISTORY_MAX:]


def _bullets(items: List[str], default: str) -> str:
    if not items:
        return default
    return "\n".join(f"- {item}" for item in items)
