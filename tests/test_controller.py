import torch

from loop_core import (
    AssociativeMemory,
    AssociativeMemoryConfig,
    Controller,
    ControllerConfig,
    WorkingMemory,
    WorkingMemoryConfig,
)


def _build():
    wm = WorkingMemory(WorkingMemoryConfig(slots=4))
    am = AssociativeMemory(AssociativeMemoryConfig(capacity=4, min_similarity=0.0))
    return Controller(wm, am, ControllerConfig()), wm, am


def test_decide_action_does_not_advance_tick():
    controller, _wm, _am = _build()
    controller.decide_action(torch.randn(8))
    assert controller.state.tick == 0


def test_step_advances_tick():
    controller, _wm, _am = _build()
    controller.step()
    controller.step()
    assert controller.state.tick == 2


def test_step_decays_energy_with_floor():
    controller, _wm, _am = _build()
    controller.config = ControllerConfig(energy_decay=0.5, min_energy=0.25)
    for _ in range(20):
        controller.step()
    assert controller.state.energy_budget == 0.25


def test_decide_action_returns_retrieval_when_traces_match():
    controller, _wm, am = _build()
    target = torch.tensor([1.0, 0.0, 0.0, 0.0])
    am.write(target, {"text": "hit"})
    actions = controller.decide_action(target)
    assert actions["retrieve"] is not None


def test_step_increases_certainty_when_retrieved():
    controller, _wm, _am = _build()
    before = controller.state.certainty
    controller.step(retrieved=True)
    assert controller.state.certainty > before


def test_step_increases_curiosity_when_not_retrieved():
    controller, _wm, _am = _build()
    before = controller.state.curiosity
    controller.step(retrieved=False)
    assert controller.state.curiosity > before


def test_reset_restores_initial_state():
    controller, _wm, _am = _build()
    controller.step()
    controller.step(retrieved=True)
    controller.reset()
    assert controller.state.tick == 0
    assert controller.state.certainty == 0.5
