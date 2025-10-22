import torch
import pytest
from loop_core import WorkingMemory, WorkingMemoryConfig


def test_working_memory_initialization():
    config = WorkingMemoryConfig(slots=4)
    wm = WorkingMemory(config)
    assert len(wm.storage) == 4
    assert all(slot is None for slot in wm.storage)


def test_working_memory_update():
    config = WorkingMemoryConfig(slots=4, use_bayesian=False)
    wm = WorkingMemory(config)
    
    embedding = torch.randn(128)
    metadata = {"text": "test", "importance_hint": 0.8}
    
    result = wm.update(embedding, metadata)
    assert result is not None
    assert wm.storage[0] is not None
    assert wm.storage[0]["metadata"]["text"] == "test"


def test_working_memory_decay():
    config = WorkingMemoryConfig(slots=4, decay_rate=0.5, use_bayesian=False)
    wm = WorkingMemory(config)
    
    embedding = torch.randn(128)
    wm.update(embedding, {"text": "first"})
    initial_importance = wm.importance[0].item()
    
    wm.update(torch.randn(128), {"text": "second"})
    decayed_importance = wm.importance[0].item()
    
    assert decayed_importance < initial_importance


def test_working_memory_reset():
    config = WorkingMemoryConfig(slots=4)
    wm = WorkingMemory(config)
    
    wm.update(torch.randn(128), {"text": "test"})
    assert wm.storage[0] is not None
    
    wm.reset()
    assert all(slot is None for slot in wm.storage)
    assert torch.all(wm.importance == 0)


def test_working_memory_pooled_state():
    config = WorkingMemoryConfig(slots=4)
    wm = WorkingMemory(config)
    
    assert wm.pooled_state() is None
    
    wm.update(torch.randn(128), {"text": "test1"})
    wm.update(torch.randn(128), {"text": "test2"})
    
    pooled = wm.pooled_state()
    assert pooled is not None
    assert pooled.shape == (128,)
