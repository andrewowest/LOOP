from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch


@dataclass
class BayesianEngineConfig:
    slots: int
    prior_log_odds: float = 0.0  # logit(0.5)
    device: Optional[torch.device] = None


class BayesianEngine:
    """Tracks posterior belief for working-memory slots."""

    def __init__(self, config: BayesianEngineConfig) -> None:
        if config.slots <= 0:
            raise ValueError("BayesianEngine requires at least one slot.")

        self.config = config
        self.device = config.device or torch.device("cpu")
        self.log_odds = torch.full((config.slots,), config.prior_log_odds, device=self.device)

    def reset(self) -> None:
        self.log_odds.fill_(self.config.prior_log_odds)

    def update(self, slot_index: int, log_likelihood_ratio: float) -> None:
        self.log_odds[slot_index] += log_likelihood_ratio

    def normalize(self) -> torch.Tensor:
        return torch.sigmoid(self.log_odds)

    def get_probabilities(self) -> List[float]:
        return torch.sigmoid(self.log_odds).tolist()
