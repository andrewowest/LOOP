from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from loop_core.bayesian import BayesianEngine, BayesianEngineConfig


@dataclass
class WorkingMemoryConfig:
    slots: int = 6
    decay_rate: float = 0.1
    temperature_floor: float = 0.35
    temperature_ceiling: float = 1.0
    device: Optional[torch.device] = None
    use_bayesian: bool = True
    bayesian_prior_log_odds: float = 0.0
    bayesian_decay: float = 0.1


class WorkingMemory:
    def __init__(self, config: WorkingMemoryConfig) -> None:
        if config.slots <= 0:
            raise ValueError("WorkingMemory requires at least one slot.")

        self.config = config
        self.device = config.device or torch.device("cpu")
        self.storage: List[Optional[Dict[str, Any]]] = [None] * config.slots
        self.importance = torch.zeros(config.slots, device=self.device)
        self._use_bayes = config.use_bayesian
        self._bayes_engine: Optional[BayesianEngine] = None
        self.last_slot_index: Optional[int] = None
        if self._use_bayes:
            bayes_cfg = BayesianEngineConfig(
                slots=config.slots,
                prior_log_odds=config.bayesian_prior_log_odds,
                device=self.device,
            )
            self._bayes_engine = BayesianEngine(bayes_cfg)

        self.temperature = config.temperature_ceiling

    def reset(self) -> None:
        for idx in range(len(self.storage)):
            self.storage[idx] = None
        self.importance.zero_()
        self.temperature = self.config.temperature_ceiling
        self.last_slot_index = None
        if self._bayes_engine is not None:
            self._bayes_engine.reset()

    def _select_slot(self) -> int:
        for idx, item in enumerate(self.storage):
            if item is None:
                return idx
        min_idx = torch.argmin(self.importance).item()
        return int(min_idx)

    def update(self, embedding: torch.Tensor, metadata: Dict[str, Any]) -> torch.Tensor:
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        embedding = embedding.to(self.device)

        if self._bayes_engine is None:
            self.importance = self.importance * (1.0 - self.config.decay_rate)
        else:
            self._bayes_engine.log_odds -= self.config.bayesian_decay
            self._bayes_engine.log_odds.clamp_(-10.0, 10.0)

        slot_idx = self._select_slot()
        energy = torch.linalg.norm(embedding).clamp_min(1e-6)
        normalized_importance = torch.tanh(energy).item()

        importance_hint = metadata.get("importance_hint") if metadata else None
        if importance_hint is not None:
            normalized_importance = max(normalized_importance, float(importance_hint))
        normalized_importance = max(0.0, min(0.99, normalized_importance))

        self.storage[slot_idx] = {
            "embedding": embedding.detach(),
            "metadata": metadata,
            "energy": float(energy.item()),
        }

        if self._bayes_engine is None:
            self.importance[slot_idx] = normalized_importance
        else:
            prob_hint = max(0.05, min(0.95, normalized_importance))
            if importance_hint is not None:
                prob_hint = max(prob_hint, max(0.05, min(0.95, float(importance_hint))))
            log_likelihood = float(torch.logit(torch.tensor(prob_hint)))
            self._bayes_engine.update(slot_idx, log_likelihood)
            self._bayes_engine.log_odds.clamp_(-10.0, 10.0)
            self.importance = self._bayes_engine.normalize()

        self.last_slot_index = slot_idx
        self._update_temperature()
        return embedding.mean(dim=0)

    def _update_temperature(self) -> None:
        avg_importance = float(self.importance.mean().item()) if self.importance.numel() else 0.0
        span = self.config.temperature_ceiling - self.config.temperature_floor
        self.temperature = (
            self.config.temperature_floor
            + span * min(1.0, max(0.0, avg_importance))
        )

    def summary(self) -> List[Dict[str, Any]]:
        summary: List[Dict[str, Any]] = []
        for idx, slot in enumerate(self.storage):
            if slot is None:
                continue
            summary.append(
                {
                    "index": idx,
                    "energy": slot["energy"],
                    "metadata": slot["metadata"],
                    "importance": float(self.importance[idx].item()),
                    "embedding": slot["embedding"],
                }
            )
        return summary

    def pooled_state(self) -> Optional[torch.Tensor]:
        vectors = [slot["embedding"] for slot in self.storage if slot is not None]
        if not vectors:
            return None
        stacked = torch.stack([vec.mean(dim=0) for vec in vectors], dim=0)
        return stacked.mean(dim=0)
