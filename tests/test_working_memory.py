import pytest
import torch

from loop_core import WorkingMemory, WorkingMemoryConfig


def test_initialization_creates_empty_slots():
    wm = WorkingMemory(WorkingMemoryConfig(slots=4))
    assert len(wm.storage) == 4
    assert all(slot is None for slot in wm.storage)


def test_invalid_slot_count_rejected():
    with pytest.raises(ValueError):
        WorkingMemory(WorkingMemoryConfig(slots=0))


def test_update_writes_to_first_empty_slot():
    wm = WorkingMemory(WorkingMemoryConfig(slots=4, use_bayesian=False))
    wm.update(torch.randn(128), {"text": "test", "importance_hint": 0.8})

    assert wm.storage[0] is not None
    assert wm.storage[0]["metadata"]["text"] == "test"
    assert wm.last_slot_index == 0


def test_decay_lowers_importance_of_existing_slots():
    wm = WorkingMemory(WorkingMemoryConfig(slots=4, decay_rate=0.5, use_bayesian=False))
    wm.update(torch.randn(128), {"text": "first"})
    initial = wm.importance[0].item()
    wm.update(torch.randn(128), {"text": "second"})
    assert wm.importance[0].item() < initial


def test_full_buffer_evicts_least_important_slot():
    wm = WorkingMemory(WorkingMemoryConfig(slots=2, decay_rate=0.0, use_bayesian=False))
    wm.update(torch.randn(128), {"text": "a", "importance_hint": 0.9})
    wm.update(torch.randn(128), {"text": "b", "importance_hint": 0.1})
    wm.update(torch.randn(128), {"text": "c", "importance_hint": 0.5})

    texts = {slot["metadata"]["text"] for slot in wm.storage if slot}
    assert "a" in texts and "c" in texts and "b" not in texts


def test_reset_clears_storage_and_importance():
    wm = WorkingMemory(WorkingMemoryConfig(slots=4))
    wm.update(torch.randn(128), {"text": "test"})
    wm.reset()
    assert all(slot is None for slot in wm.storage)
    assert torch.all(wm.importance == 0)
    assert wm.last_slot_index is None


def test_pooled_state_returns_none_when_empty():
    wm = WorkingMemory(WorkingMemoryConfig(slots=4))
    assert wm.pooled_state() is None


def test_pooled_state_aggregates_active_slots():
    wm = WorkingMemory(WorkingMemoryConfig(slots=4))
    wm.update(torch.randn(128), {"text": "a"})
    wm.update(torch.randn(128), {"text": "b"})
    pooled = wm.pooled_state()
    assert pooled is not None
    assert pooled.shape == (128,)


def test_bayesian_path_increases_importance_on_hint():
    wm = WorkingMemory(WorkingMemoryConfig(slots=4, use_bayesian=True))
    wm.update(torch.randn(128), {"text": "low", "importance_hint": 0.1})
    low = wm.importance[wm.last_slot_index].item()
    wm.update(torch.randn(128), {"text": "high", "importance_hint": 0.9})
    high = wm.importance[wm.last_slot_index].item()
    assert high > low


def test_bayesian_log_odds_stay_clamped():
    """Many extreme updates must not blow up the posterior to NaN."""
    wm = WorkingMemory(WorkingMemoryConfig(slots=2, use_bayesian=True))
    for _ in range(100):
        wm.update(torch.randn(8), {"text": "x", "importance_hint": 0.99})
    assert torch.isfinite(wm.importance).all()
    assert (wm.importance <= 1.0).all() and (wm.importance >= 0.0).all()


def test_eviction_breaks_ties_by_oldest_write():
    wm = WorkingMemory(WorkingMemoryConfig(slots=2, decay_rate=0.0, use_bayesian=False))
    wm.update(torch.randn(8), {"text": "old", "importance_hint": 0.5})
    wm.update(torch.randn(8), {"text": "new", "importance_hint": 0.5})
    wm.update(torch.randn(8), {"text": "newest", "importance_hint": 0.5})

    texts = {slot["metadata"]["text"] for slot in wm.storage if slot}
    assert "old" not in texts
    assert "newest" in texts


def test_temperature_modulates_with_importance():
    wm = WorkingMemory(
        WorkingMemoryConfig(
            slots=4, use_bayesian=False, temperature_floor=0.2, temperature_ceiling=1.0
        )
    )
    wm.update(torch.randn(128), {"text": "a", "importance_hint": 0.05})
    cool = wm.temperature
    wm.reset()
    wm.update(torch.randn(128), {"text": "b", "importance_hint": 0.95})
    assert wm.temperature > cool
