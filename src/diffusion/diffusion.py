"""Gaussian diffusion training objective and a deterministic DDIM sampler."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from torch import nn


def cosine_beta_schedule(timesteps: int, offset: float = 0.008) -> torch.Tensor:
    steps = torch.linspace(0, timesteps, timesteps + 1, dtype=torch.float64)
    alpha_bar = torch.cos(((steps / timesteps) + offset) / (1 + offset) * math.pi / 2) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(1e-8, 0.999).float()


def linear_beta_schedule(timesteps: int) -> torch.Tensor:
    scale = 1000 / timesteps
    return torch.linspace(scale * 1e-4, scale * 2e-2, timesteps).clamp(max=0.999)


def _extract(values: torch.Tensor, timesteps: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    selected = values.gather(0, timesteps)
    return selected.reshape(timesteps.shape[0], *((1,) * (len(shape) - 1)))


class GaussianDiffusion(nn.Module):
    """Epsilon-prediction DDPM training with DDIM sampling."""

    def __init__(self, timesteps: int = 1000, beta_schedule: str = "cosine"):
        super().__init__()
        if timesteps < 2:
            raise ValueError(f"timesteps must be at least 2, got {timesteps}")
        if beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        elif beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule!r}")

        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.timesteps = int(timesteps)
        self.beta_schedule = beta_schedule
        self.register_buffer("betas", betas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", alpha_bars.sqrt())
        self.register_buffer("sqrt_one_minus_alpha_bars", (1.0 - alpha_bars).sqrt())

    def q_sample(
        self, clean: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        return (
            _extract(self.sqrt_alpha_bars, timesteps, clean.shape) * clean
            + _extract(self.sqrt_one_minus_alpha_bars, timesteps, clean.shape) * noise
        )

    def training_loss(
        self,
        model: nn.Module,
        clean: torch.Tensor,
        condition: torch.Tensor | None = None,
        *,
        snr_gamma: float | None = 5.0,
        timesteps: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch = clean.shape[0]
        if timesteps is None:
            timesteps = torch.randint(0, self.timesteps, (batch,), device=clean.device)
        if noise is None:
            noise = torch.randn_like(clean)
        noisy = self.q_sample(clean, timesteps, noise)
        prediction = model(noisy, timesteps, condition)
        per_sample = (prediction.float() - noise.float()).square().flatten(1).mean(1)
        if snr_gamma is not None and snr_gamma > 0:
            alpha_bar = _extract(self.alpha_bars, timesteps, clean.shape).flatten()
            snr = alpha_bar / (1.0 - alpha_bar).clamp_min(1e-8)
            weights = snr.clamp(max=float(snr_gamma)) / snr.clamp_min(1e-8)
            per_sample = per_sample * weights
        return per_sample.mean()

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: tuple[int, int, int, int, int],
        *,
        condition: torch.Tensor | None = None,
        inference_steps: int = 50,
        eta: float = 0.0,
        generator: torch.Generator | None = None,
        initial_noise: torch.Tensor | None = None,
        callback: Callable[[int, int, torch.Tensor], None] | None = None,
    ) -> torch.Tensor:
        """Sample a video with DDIM; ``eta=0`` is deterministic given the noise."""
        if not 1 <= inference_steps <= self.timesteps:
            raise ValueError(
                f"inference_steps must be in [1, {self.timesteps}], got {inference_steps}"
            )
        if eta < 0:
            raise ValueError(f"eta must be non-negative, got {eta}")
        parameter = next(model.parameters())
        device = parameter.device
        dtype = parameter.dtype
        if initial_noise is None:
            video = torch.randn(shape, device=device, dtype=dtype, generator=generator)
        else:
            if tuple(initial_noise.shape) != shape:
                raise ValueError(
                    f"Initial noise shape {tuple(initial_noise.shape)} does not match {shape}"
                )
            video = initial_noise.to(device=device, dtype=dtype)
        if condition is not None:
            condition = condition.to(device=device, dtype=dtype)

        was_training = model.training
        model.eval()
        times = torch.linspace(
            self.timesteps - 1, 0, inference_steps, device=device
        ).round().long()
        times = torch.unique_consecutive(times)
        for index, timestep_tensor in enumerate(times):
            timestep = int(timestep_tensor.item())
            next_timestep = int(times[index + 1].item()) if index + 1 < len(times) else -1
            batch_times = torch.full((shape[0],), timestep, device=device, dtype=torch.long)
            predicted_noise = model(video, batch_times, condition)

            alpha = self.alpha_bars[timestep].to(device=device, dtype=video.dtype)
            predicted_clean = (
                video - (1 - alpha).sqrt() * predicted_noise
            ) / alpha.sqrt().clamp_min(1e-8)
            predicted_clean = predicted_clean.clamp(-1.0, 1.0)
            if next_timestep < 0:
                video = predicted_clean
            else:
                alpha_next = self.alpha_bars[next_timestep].to(
                    device=device, dtype=video.dtype
                )
                sigma = eta * (
                    (1 - alpha_next)
                    / (1 - alpha).clamp_min(1e-8)
                    * (1 - alpha / alpha_next).clamp_min(0)
                ).sqrt()
                direction = (1 - alpha_next - sigma.square()).clamp_min(0).sqrt()
                if eta:
                    random_noise = torch.randn(
                        video.shape, device=device, dtype=dtype, generator=generator
                    )
                else:
                    random_noise = 0.0
                video = (
                    alpha_next.sqrt() * predicted_clean
                    + direction * predicted_noise
                    + sigma * random_noise
                )
            if callback is not None:
                callback(index + 1, len(times), video)
        model.train(was_training)
        return video.clamp(-1.0, 1.0)

    def config_dict(self) -> dict[str, int | str]:
        return {"timesteps": self.timesteps, "beta_schedule": self.beta_schedule}
