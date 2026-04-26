import math

import pytest
import torch

from loop_core.bayesian import BayesianEngine, BayesianEngineConfig


def test_invalid_slot_count_rejected():
    with pytest.raises(ValueError):
        BayesianEngine(BayesianEngineConfig(slots=0))


def test_prior_log_odds_set_correctly():
    engine = BayesianEngine(BayesianEngineConfig(slots=3, prior_log_odds=math.log(3)))
    probs = engine.get_probabilities()
    assert all(abs(p - 0.75) < 1e-6 for p in probs)


def test_update_shifts_only_target_slot():
    engine = BayesianEngine(BayesianEngineConfig(slots=3))
    engine.update(1, math.log(9))  # logit -> 0.9
    probs = engine.get_probabilities()
    assert abs(probs[0] - 0.5) < 1e-6
    assert abs(probs[1] - 0.9) < 1e-3
    assert abs(probs[2] - 0.5) < 1e-6


def test_normalize_returns_sigmoid_tensor():
    engine = BayesianEngine(BayesianEngineConfig(slots=2, prior_log_odds=0.0))
    result = engine.normalize()
    assert isinstance(result, torch.Tensor)
    assert torch.allclose(result, torch.tensor([0.5, 0.5]))


def test_reset_returns_to_prior():
    engine = BayesianEngine(BayesianEngineConfig(slots=2, prior_log_odds=0.5))
    engine.update(0, 5.0)
    engine.reset()
    assert torch.allclose(engine.log_odds, torch.full((2,), 0.5))
