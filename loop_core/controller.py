from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict

import torch

from loop_core.memory import AssociativeMemory, WorkingMemory


@dataclass
class ControllerState:
    tick: int = 0
    energy_budget: float = 1.0
    curiosity: float = 0.5
    certainty: float = 0.5


@dataclass
class ControllerConfig:
    energy_decay: float = 0.02
    curiosity_gain: float = 0.1
    certainty_decay: float = 0.05
    min_energy: float = 0.1


class Controller:
    """Manages attention and energy across the memory tiers."""

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

    def decide_action(self, query_embedding: torch.Tensor) -> Dict[str, Any]:
        """Probe associative memory for related context. Pure: no state mutation.

        Returned ``state`` is a snapshot — callers must not rely on it tracking
        future controller updates.
        """
        actions: Dict[str, Any] = {
            "retrieve": None,
            "temperature": torch.tensor(self.working_memory.temperature),
            "state": replace(self.state),
        }

        recalled = self.associative_memory.recall(query_embedding, k=3)
        if recalled:
            stacked = torch.stack([trace.embedding for _, trace in recalled])
            actions["retrieve"] = stacked.mean(dim=0)
        return actions

    def step(self, retrieved: bool = False) -> None:
        """Advance one turn. Decays energy, updates curiosity/certainty,
        and lets associative memory cool down."""
        self.state.tick += 1
        self._decay_energy()
        if retrieved:
            self.state.certainty = min(1.0, self.state.certainty + 0.1)
        else:
            self.state.curiosity = min(
                1.0, self.state.curiosity + self.config.curiosity_gain * 0.5
            )
            self.state.certainty = max(
                0.0, self.state.certainty - self.config.certainty_decay
            )
        self.associative_memory.decay()

    def _decay_energy(self) -> None:
        self.state.energy_budget = max(
            self.config.min_energy,
            self.state.energy_budget * (1.0 - self.config.energy_decay),
        )
