import random

import torch
from torch import Tensor

FIELD_MAX = 0.75
TWO_PI = 2 * torch.pi


class Augment:
    def __init__(self, height: int = 512, width: int = 512) -> None:
        self.h = height
        self.w = width
        self.grid_x, self.grid_y = self._generate_grid(height, width)

    def _generate_grid(
        self,
        h: int,
        w: int,
    ) -> tuple[Tensor, Tensor]:
        x = torch.linspace(0, TWO_PI, w, dtype=torch.float32)
        y = torch.linspace(0, TWO_PI, h, dtype=torch.float32)
        return torch.meshgrid(x, y, indexing="ij")

    def _resize_grid_if_needed(
        self,
        h: int,
        w: int,
    ) -> tuple[Tensor, Tensor]:
        if h != self.h or w != self.w:
            return self._generate_grid(h, w)
        return self.grid_x, self.grid_y

    def __call__(
        self,
        target: Tensor,
    ) -> Tensor:
        if target.dim() not in (3, 4, 5):
            raise ValueError(f"Invalid target shape: {target.shape}")

        h, w = target.shape[-2:]
        grid_x, grid_y = self._resize_grid_if_needed(h, w)

        phase_field = grid_x * random.uniform(0.0, 1.0) + grid_y * random.uniform(0.0, 1.0)
        phase_field = phase_field / phase_field.max() * TWO_PI * random.uniform(0.0, FIELD_MAX)

        bias = random.uniform(0, TWO_PI)
        total_phase = bias + phase_field
        phase_shift = torch.exp(1j * total_phase)

        while phase_shift.ndim < target.ndim:
            phase_shift = phase_shift.unsqueeze(0)

        return target * phase_shift
