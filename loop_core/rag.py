from __future__ import annotations

from typing import List, Optional

from loop_core.controller import Controller
from loop_core.memory import AssociativeMemory, LongTermMemory, WorkingMemory
from loop_core.persona import PersonaProfile


class PersonaCoordinator:
    """Coordinates retrieval and persona enforcement using LOOP components."""

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

        self.conversation_history: List[tuple[str, str]] = []

    def add_persona_fact(self, fact: str) -> None:
        fact = fact.strip()
        if not fact:
            return
        if fact not in self.persona.persona_facts:
            self.persona.persona_facts.append(fact)

    def build_prompt(self, user_input: str, system_instruction: Optional[str] = None) -> str:
        if system_instruction is None:
            system_instruction = "You are Andrew. Respond using the persona facts and guidelines."  # type: ignore[str-bytes-safe]

        tone_block = "\n".join(f"- {item}" for item in self.persona.tone_guidelines) or "- Use Andrew's natural tone."
        response_block = "\n".join(f"- {item}" for item in self.persona.response_guidelines) or "- Keep responses structured and concise."
        facts_block = "\n".join(f"- {fact}" for fact in self.persona.persona_facts) or "- (no extra persona overrides)"

        history_block = "\n".join(
            f"User: {user}\nYou: {ai}"
            for user, ai in self.conversation_history[-4:]
        )

        prompt = f"""{system_instruction}

Tone guidelines:
{tone_block}

Response style:
{response_block}

Persona facts:
{facts_block}

Conversation:
{history_block}

User: {user_input}
Andrew:"""
        return prompt

    def update_conversation(self, user_input: str, ai_response: str) -> None:
        self.conversation_history.append((user_input, ai_response))
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
