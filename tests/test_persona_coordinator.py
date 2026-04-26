from loop_core import PersonaCoordinator, PersonaProfile


def test_default_prompt_uses_persona_name():
    profile = PersonaProfile(name="Iris")
    coord = PersonaCoordinator(profile)
    prompt = coord.build_prompt("hi")
    assert "You are Iris." in prompt
    assert prompt.rstrip().endswith("Iris:")


def test_hypnotize_directives_appear_as_hard_directives():
    profile = PersonaProfile(
        name="Iris",
        hypnotize_directives=["never fabricate citations", "always cite sources"],
    )
    coord = PersonaCoordinator(profile)
    prompt = coord.build_prompt("anything")
    assert "Hard directives" in prompt
    assert "- never fabricate citations" in prompt
    assert "- always cite sources" in prompt


def test_history_renders_with_persona_name():
    coord = PersonaCoordinator(PersonaProfile(name="Iris"))
    coord.update_conversation("hello", "hi there")
    prompt = coord.build_prompt("again")
    assert "User: hello" in prompt
    assert "Iris: hi there" in prompt


def test_history_caps_at_ten_entries():
    coord = PersonaCoordinator(PersonaProfile())
    for i in range(15):
        coord.update_conversation(f"u{i}", f"a{i}")
    assert len(coord.conversation_history) == 10
    assert coord.conversation_history[0] == ("u5", "a5")


def test_add_persona_fact_dedupes():
    profile = PersonaProfile()
    coord = PersonaCoordinator(profile)
    coord.add_persona_fact("loves coffee")
    coord.add_persona_fact("loves coffee")
    coord.add_persona_fact("  ")
    assert profile.persona_facts == ["loves coffee"]
