from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from loop_core.memory import AssociativeMemory, WorkingMemory


@dataclass
class ControllerState:
    step: int = 0
    energy_budget: float = 1.0
    curiosity: float = 0.5
    certainty: float = 0.5


@dataclass
class ControllerConfig:
    energy_decay: float = 0.02
    curiosity_gain: float = 0.1
    certainty_decay: float = 0.05
    min_energy: float = 0.1
    max_energy: float = 2.0


class Controller:
    def __init__(
        self,
        working_memory: WorkingMemory,
        associative_memory: AssociativeMemory,
        config: ControllerConfig,
    ) -> None:
        self.working_memory = working_memory
        self.associative_memory = associative_memory
        self.config = config
        self.state = ControllerState()

    def reset(self) -> None:
        self.working_memory.reset()
        self.state = ControllerState()

    def decide_action(self, query_embedding: torch.Tensor) -> Dict[str, Optional[torch.Tensor]]:
        self.state.step += 1
        self._decay_energy()

        actions: Dict[str, Optional[torch.Tensor]] = {
            "retrieve": None,
            "temperature": torch.tensor(self.working_memory.temperature),
            "state": self.state,
        }

        recalled = self.associative_memory.recall(query_embedding, k=3)
        if recalled:
            weighted = [trace.embedding for _, trace in recalled]
            actions["retrieve"] = torch.stack(weighted).mean(dim=0)
            self.state.certainty = min(1.0, self.state.certainty + 0.1)
        else:
            self.state.curiosity = min(1.0, self.state.curiosity + self.config.curiosity_gain * 0.5)
            self.state.certainty = max(0.0, self.state.certainty - self.config.certainty_decay)

        return actions

    def _decay_energy(self) -> None:
        self.state.energy_budget = max(
            self.config.min_energy,
            self.state.energy_budget * (1.0 - self.config.energy_decay),
        )

    def step(self) -> None:
        self.associative_memory.decay()
