import torch

from loop_core import AssociativeMemory, AssociativeMemoryConfig


def test_recall_returns_empty_when_store_empty():
    am = AssociativeMemory(AssociativeMemoryConfig(capacity=4))
    assert am.recall(torch.randn(8)) == []


def test_recall_finds_most_similar_trace():
    am = AssociativeMemory(AssociativeMemoryConfig(capacity=4, min_similarity=0.0))
    target = torch.tensor([1.0, 0.0, 0.0, 0.0])
    am.write(target, {"text": "needle"})
    am.write(torch.tensor([0.0, 1.0, 0.0, 0.0]), {"text": "noise"})

    results = am.recall(target, k=1)
    assert len(results) == 1
    assert results[0][1].metadata["text"] == "needle"


def test_recall_filters_below_min_similarity():
    am = AssociativeMemory(AssociativeMemoryConfig(capacity=4, min_similarity=0.95))
    am.write(torch.tensor([1.0, 0.0, 0.0, 0.0]), {"text": "orthogonal"})
    results = am.recall(torch.tensor([0.0, 1.0, 0.0, 0.0]), k=1)
    assert results == []


def test_recall_increments_recall_count():
    am = AssociativeMemory(AssociativeMemoryConfig(capacity=4, min_similarity=0.0))
    target = torch.tensor([1.0, 0.0, 0.0, 0.0])
    am.write(target, {"text": "x"})
    am.recall(target, k=1)
    am.recall(target, k=1)
    assert am.traces[0].recall_count == 2


def test_eviction_drops_stale_unrecalled_trace_not_useful_one():
    """Regression: prior implementation evicted the most-recalled trace."""
    am = AssociativeMemory(AssociativeMemoryConfig(capacity=2, min_similarity=0.0))
    keep = torch.tensor([1.0, 0.0, 0.0, 0.0])
    stale = torch.tensor([0.0, 1.0, 0.0, 0.0])
    am.write(keep, {"text": "keep"})
    am.write(stale, {"text": "stale"})

    # Recall "keep" repeatedly so it becomes the useful trace.
    for _ in range(3):
        am.recall(keep, k=1)

    # Writing a third trace must evict "stale", not "keep".
    am.write(torch.tensor([0.0, 0.0, 1.0, 0.0]), {"text": "new"})
    texts = {t.metadata["text"] for t in am.traces}
    assert "keep" in texts
    assert "stale" not in texts


def test_decay_lowers_recall_count_floor_zero():
    am = AssociativeMemory(AssociativeMemoryConfig(capacity=2))
    am.write(torch.randn(4), {"text": "x"})
    am.traces[0].recall_count = 2
    am.decay()
    am.decay()
    am.decay()
    assert am.traces[0].recall_count == 0
