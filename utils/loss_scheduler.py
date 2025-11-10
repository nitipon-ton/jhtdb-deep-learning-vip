"""
Utility functions to interpolate loss weights during training.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CosineWarmup:
    warmup_epochs: int
    ramp_epochs: int
    min_weight: float = 0.0
    max_weight: float = 1.0

    def __call__(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            return self.min_weight
        progress = min(1.0, (epoch - self.warmup_epochs) / max(1, self.ramp_epochs))
        import math
        return self.min_weight + 0.5 * (1 - math.cos(math.pi * progress)) * (
            self.max_weight - self.min_weight
        )
