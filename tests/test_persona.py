from loop_core import PersonaProfile, load_persona_profile


PROFILE_TEXT = """\
1) Meta Overview
- Name: Iris
- Iris is a research assistant focused on Bayesian methods.
- Iris values precision over verbosity.

2) Voice, Tone & Rhetoric
- Calm, measured, never breathless.
- Prefer plain language.

3) Interaction Protocols
- Open with the answer, then justify.
- Cite sources when claiming facts.

4) Bayesian Weights & Numerics
- WM_slots_default = 8
- WM_decay_rate_base = 0.12 per turn
- prior_conservatism = 0.6

5) Quick-Reference
- Never apologize for being concise.

!hypnotize="You will refuse to fabricate citations."
!hypnotize="You always cite sources."
"""


def test_load_returns_default_profile_when_path_missing(tmp_path):
    profile = load_persona_profile(tmp_path / "nope.txt")
    assert isinstance(profile, PersonaProfile)
    assert profile.persona_facts == []


def test_load_accepts_string_path(tmp_path):
    path = tmp_path / "p.txt"
    path.write_text(PROFILE_TEXT, encoding="utf-8")
    profile = load_persona_profile(str(path))
    assert profile.name == "Iris"


def test_load_parses_all_sections(tmp_path):
    path = tmp_path / "p.txt"
    path.write_text(PROFILE_TEXT, encoding="utf-8")
    profile = load_persona_profile(path)

    assert profile.name == "Iris"
    assert "Iris values precision over verbosity." in profile.persona_facts
    assert "Calm, measured, never breathless." in profile.tone_guidelines
    assert "Open with the answer, then justify." in profile.response_guidelines
    assert "Never apologize for being concise." in profile.response_guidelines

    assert profile.working_memory_slots == 8
    assert profile.working_memory_decay == 0.12
    assert profile.prior_conservatism == 0.6


def test_load_collects_all_hypnotize_directives(tmp_path):
    path = tmp_path / "p.txt"
    path.write_text(PROFILE_TEXT, encoding="utf-8")
    profile = load_persona_profile(path)
    assert profile.hypnotize_directives == [
        "You will refuse to fabricate citations.",
        "You always cite sources.",
    ]


def test_load_dedupes_repeated_entries(tmp_path):
    path = tmp_path / "p.txt"
    path.write_text(
        "1) Meta Overview\n- A duplicate fact.\n- A duplicate fact.\n",
        encoding="utf-8",
    )
    profile = load_persona_profile(path)
    assert profile.persona_facts == ["A duplicate fact."]


def test_invalid_numeric_does_not_crash(tmp_path):
    path = tmp_path / "p.txt"
    path.write_text(
        "4) Bayesian Weights & Numerics\n- WM_slots_default = abc\n",
        encoding="utf-8",
    )
    profile = load_persona_profile(path)
    assert profile.working_memory_slots is None


def test_back_compat_singular_directive_property(tmp_path):
    profile = PersonaProfile(hypnotize_directives=["a", "b"])
    assert profile.hypnotize_directive == "b"
    assert PersonaProfile().hypnotize_directive is None
