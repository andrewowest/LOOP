import numpy as np

from loop_core import (
    AssociativeMemory,
    AssociativeMemoryConfig,
    LongTermMemory,
    LongTermMemoryConfig,
    MemoryConsolidator,
    WorkingMemory,
    WorkingMemoryConfig,
)


class StubEncoder:
    """Deterministic encoder: hashes text into a fixed-dim unit-ish vector."""

    def __init__(self, dim: int = 16) -> None:
        self.dim = dim

    def encode(self, texts, convert_to_numpy=True):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            rng = np.random.default_rng(abs(hash(text)) % (2**32))
            v = rng.standard_normal(self.dim).astype(np.float32)
            v /= np.linalg.norm(v) or 1.0
            out[i] = v
        return out


def _build(tmp_path):
    wm = WorkingMemory(WorkingMemoryConfig(slots=4, use_bayesian=False))
    am = AssociativeMemory(AssociativeMemoryConfig(capacity=8, min_similarity=0.0))
    ltm = LongTermMemory(LongTermMemoryConfig(storage_path=tmp_path / "k.jsonl"))
    consolidator = MemoryConsolidator(wm, am, ltm, StubEncoder())
    return consolidator, wm, am, ltm


def test_routine_turn_promotes_to_associative(tmp_path):
    consolidator, _wm, am, _ltm = _build(tmp_path)
    consolidator.process_turn("I prefer dark mode.", "Got it.")
    assert any("dark mode" in (t.metadata.get("text") or "") for t in am.traces)


def test_repeated_mention_drives_long_term_consolidation(tmp_path):
    consolidator, _wm, _am, ltm = _build(tmp_path)
    for _ in range(3):
        consolidator.process_turn("My favorite color is blue.", "Noted.")

    entries = ltm.load_recent()
    assert any("favorite color is blue" in (e.get("text") or "").lower() for e in entries)


def test_hypnotize_forces_long_term_immediately(tmp_path):
    consolidator, _wm, _am, ltm = _build(tmp_path)
    directive = "You will keep responses under 3 sentences."
    consolidator.record_hypnotize(directive)

    entries = ltm.load_recent()
    assert len(entries) == 1
    assert entries[0]["text"] == directive
    assert entries[0]["type"] == "hypnotize"
    assert entries[0]["importance"] >= 0.9


def test_absorb_persona_replays_directives_from_profile(tmp_path):
    from loop_core import PersonaProfile

    consolidator, _wm, _am, ltm = _build(tmp_path)
    profile = PersonaProfile(
        hypnotize_directives=[
            "You must never recommend peanuts.",
            "You are willing to use technical jargon.",
        ]
    )
    consolidator.absorb_persona(profile)

    texts = [e["text"] for e in ltm.load_recent()]
    assert "You must never recommend peanuts." in texts
    assert "You are willing to use technical jargon." in texts


def test_empty_text_is_ignored(tmp_path):
    consolidator, _wm, am, _ltm = _build(tmp_path)
    consolidator.process_turn("", "   ")
    assert am.traces == []


def test_shouting_heuristic():
    from loop_core.memory.consolidator import _is_shouting

    assert _is_shouting("URGENT FIX")
    assert _is_shouting("PLEASE help me")
    assert not _is_shouting("hello")
    assert not _is_shouting("OK")  # only 2 caps
    assert not _is_shouting("This Is Title Case")  # density too low


def test_caps_emphasis_lifts_importance_hint(tmp_path):
    """Regression: emphasis_bonus must apply via the shouting branch, not just '!'."""
    consolidator, _wm, am, _ltm = _build(tmp_path)
    consolidator.process_turn("THIS IS URGENT", "ok")
    assert any("URGENT" in (t.metadata.get("text") or "") for t in am.traces)


def test_attached_coordinator_history_updates_with_turn(tmp_path):
    from loop_core import PersonaCoordinator, PersonaProfile

    consolidator, _wm, _am, _ltm = _build(tmp_path)
    coord = PersonaCoordinator(PersonaProfile(name="Iris"))
    consolidator.attach_coordinator(coord)

    consolidator.process_turn("hello", "hi back")
    assert coord.conversation_history == [("hello", "hi back")]


def test_timestamp_is_iso_with_offset(tmp_path):
    consolidator, _wm, _am, ltm = _build(tmp_path)
    consolidator.record_hypnotize("test directive")
    timestamp = ltm.load_recent()[0]["timestamp"]
    # ISO 8601 with timezone — either +00:00 or trailing Z.
    assert "T" in timestamp
    assert timestamp.endswith("+00:00") or timestamp.endswith("Z")
