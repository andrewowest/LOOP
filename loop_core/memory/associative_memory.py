from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class AssociativeMemoryConfig:
    capacity: int = 512
    device: Optional[torch.device] = None
    similarity_temperature: float = 0.2
    retention_factor: float = 0.95
    min_similarity: float = 0.15


@dataclass
class MemoryTrace:
    embedding: torch.Tensor
    metadata: Dict[str, Any]
    recall_count: int = 0
    age: int = 0


class AssociativeMemory:
    def __init__(self, config: AssociativeMemoryConfig) -> None:
        if config.capacity <= 0:
            raise ValueError("AssociativeMemory requires positive capacity.")

        self.config = config
        self.device = config.device or torch.device("cpu")
        self.traces: List[MemoryTrace] = []
        self.step_counter = 0

    def write(self, embedding: torch.Tensor, metadata: Dict[str, Any]) -> None:
        embedding = embedding.detach().to(self.device)
        self.step_counter += 1

        if len(self.traces) >= self.config.capacity:
            victim_idx = self._select_eviction_index()
            self.traces.pop(victim_idx)

        self.traces.append(
            MemoryTrace(
                embedding=embedding,
                metadata=metadata,
                recall_count=0,
                age=self.step_counter,
            )
        )

    def _select_eviction_index(self) -> int:
        scores = []
        for idx, trace in enumerate(self.traces):
            retention = (self.step_counter - trace.age + 1)
            score = trace.recall_count + retention * (1.0 - self.config.retention_factor)
            scores.append(score)
        max_idx = scores.index(max(scores))
        return max_idx

    def recall(self, query: torch.Tensor, k: int = 5) -> List[Tuple[float, MemoryTrace]]:
        if not self.traces:
            return []

        query = query.detach().to(self.device)
        similarities = []
        for trace in self.traces:
            sim = F.cosine_similarity(query, trace.embedding, dim=-1)
            similarities.append(float(sim))

        topk = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)[:k]
        threshold = self.config.min_similarity

        results: List[Tuple[float, MemoryTrace]] = []
        for idx, sim in topk:
            if sim < threshold:
                continue
            trace = self.traces[idx]
            trace.recall_count += 1
            weighted_sim = sim / max(self.config.similarity_temperature, 1e-6)
            results.append((weighted_sim, trace))

        return results

    def decay(self) -> None:
        for trace in self.traces:
            trace.recall_count = max(0, trace.recall_count - 1)
