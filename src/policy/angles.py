"""Angle wrapping helpers shared by dataset statistics and policy inference."""

from __future__ import annotations

import numpy as np
import torch


def wrap_numpy(values: np.ndarray, dims: tuple[int, ...] | list[int]) -> np.ndarray:
    if not dims:
        return values
    result = values.copy()
    result[..., dims] = (result[..., dims] + 180.0) % 360.0 - 180.0
    return result


def wrap_torch(values: torch.Tensor, dims: tuple[int, ...] | list[int]) -> torch.Tensor:
    if not dims:
        return values
    result = values.clone()
    result[..., list(dims)] = torch.remainder(result[..., list(dims)] + 180.0, 360.0) - 180.0
    return result
