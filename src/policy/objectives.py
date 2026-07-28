"""The two generative objectives the action decoder can be trained under.

Both expose the same pair of methods, so the policy and the training loop never
branch on which one is in use:

    loss(decoder_fn, actions, weights)    -> scalar loss, extra metrics
    sample(decoder_fn, shape, device)     -> predicted actions

``decoder_fn(noisy_actions, timesteps)`` runs the U-Net. Keeping the generative
process behind that one callable is what lets the policy slot classifier-free
guidance in without either objective knowing about it: it hands over a function
that has already combined the conditional and unconditional branches.

    DiffusionObjective       DDPM training, epsilon prediction, DDIM sampling
    FlowMatchingObjective    rectified flow, velocity prediction, Euler sampling

They differ in what the decoder regresses and how a sample is produced, not in
the network, the conditioning, or the action representation. That is the point:
switching objectives changes one constructor argument.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

if TYPE_CHECKING:
    from .config import PolicyConfig

OBJECTIVES = ("diffusion", "flow")
BETA_SCHEDULES = ("cosine", "linear")
FLOW_TIME_SAMPLINGS = ("uniform", "logitnormal")

# Flow matching's time runs over [0, 1], but the sinusoidal step embedding was
# designed for the discrete diffusion range: its highest frequency is 1.0, so
# over a unit interval every channel is nearly constant and the decoder can
# barely tell one noise level from another. Scaling the time up to the diffusion
# range restores the same resolution the embedding already gets under DDPM.
FLOW_TIME_SCALE = 100.0

DecoderFn = Callable[[Tensor, Tensor], Tensor]


def _weighted_mean(error: Tensor, weights: Tensor) -> Tensor:
    return (error * weights).sum() / weights.sum().clamp_min(1e-8)


def cosine_beta_schedule(num_timesteps: int, max_beta: float = 0.999, offset: float = 0.008) -> Tensor:
    steps = torch.arange(num_timesteps + 1, dtype=torch.float64) / num_timesteps
    alphas_cumprod = torch.cos((steps + offset) / (1 + offset) * torch.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return betas.clamp(max=max_beta).float()


class DiffusionObjective(nn.Module):
    """DDPM training with epsilon prediction and deterministic DDIM sampling."""

    def __init__(
        self,
        num_train_timesteps: int = 100,
        num_inference_steps: int = 10,
        schedule: str = "cosine",
        clip_sample: bool = True,
    ):
        super().__init__()
        if schedule == "cosine":
            betas = cosine_beta_schedule(num_train_timesteps)
        elif schedule == "linear":
            betas = torch.linspace(1e-4, 0.02, num_train_timesteps)
        else:
            raise ValueError(f"Unknown beta schedule {schedule!r}")
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = min(num_inference_steps, num_train_timesteps)
        self.clip_sample = clip_sample
        self.register_buffer("alphas_cumprod", torch.cumprod(1.0 - betas, dim=0))

    def loss(self, decoder_fn: DecoderFn, actions: Tensor, weights: Tensor) -> tuple[Tensor, dict[str, float]]:
        batch = actions.shape[0]
        noise = torch.randn_like(actions)
        timesteps = torch.randint(0, self.num_train_timesteps, (batch,), device=actions.device)
        alpha = self.alphas_cumprod[timesteps].view(batch, 1, 1)
        noisy = alpha.sqrt() * actions + (1 - alpha).sqrt() * noise
        prediction = decoder_fn(noisy, timesteps.float())
        value = _weighted_mean((prediction - noise).pow(2), weights)
        return value, {}

    @torch.no_grad()
    def sample(self, decoder_fn: DecoderFn, shape: tuple[int, int, int], device: torch.device,
               dtype: torch.dtype = torch.float32) -> Tensor:
        sample = torch.randn(shape, device=device, dtype=dtype)
        # Note: the stride is integer division, so a step count that does not
        # divide num_train_timesteps evenly runs ceil(num_train / stride) steps.
        # 10 into 100 is exact; 3 actually runs 4.
        stride = max(1, self.num_train_timesteps // self.num_inference_steps)
        timeline = list(range(0, self.num_train_timesteps, stride))[::-1]
        for index, timestep in enumerate(timeline):
            batched = torch.full((shape[0],), float(timestep), device=device)
            noise_prediction = decoder_fn(sample, batched).to(dtype)
            alpha = self.alphas_cumprod[timestep]
            original = (sample - (1 - alpha).sqrt() * noise_prediction) / alpha.sqrt()
            if self.clip_sample:
                original = original.clamp(-1.0, 1.0)
            previous_timestep = timeline[index + 1] if index + 1 < len(timeline) else None
            if previous_timestep is None:
                sample = original
            else:
                alpha_previous = self.alphas_cumprod[previous_timestep]
                direction = (1 - alpha_previous).sqrt() * noise_prediction
                sample = alpha_previous.sqrt() * original + direction
        return sample


class FlowMatchingObjective(nn.Module):
    """Rectified flow: a straight path from noise to data, integrated with Euler.

    Time runs from ``t = 0`` (pure noise) to ``t = 1`` (data) along the linear
    interpolation ``x_t = (1 - t) * noise + t * action``, whose velocity is the
    constant ``action - noise``. Training regresses that velocity; sampling
    integrates it forward. There is no noise schedule to choose and no
    discretisation of the forward process - the two knobs a DDPM needs
    (``num_train_timesteps``, ``beta_schedule``) simply do not exist here.

    Compared with the diffusion objective:

    * the decoder predicts a velocity rather than the noise;
    * the training time is continuous, so every step trains a different noise
      level rather than one of 100 buckets;
    * ``num_inference_steps`` is exact - N steps means N decoder calls - where
      the DDIM sampler rounds its stride;
    * the path is straight by construction, which is why flow matching tends to
      stay usable at very few sampling steps.

    Classifier-free guidance carries over unchanged: guiding a velocity field is
    the same linear extrapolation as guiding an epsilon prediction, and the
    policy hands both objectives an already-guided ``decoder_fn``.
    """

    def __init__(
        self,
        num_inference_steps: int = 10,
        clip_sample: bool = True,
        time_sampling: str = "uniform",
        time_scale: float = FLOW_TIME_SCALE,
    ):
        super().__init__()
        if time_sampling not in FLOW_TIME_SAMPLINGS:
            raise ValueError(f"time_sampling must be one of {FLOW_TIME_SAMPLINGS}, got {time_sampling!r}")
        if num_inference_steps < 1:
            raise ValueError(f"num_inference_steps must be >= 1, got {num_inference_steps}")
        self.num_inference_steps = int(num_inference_steps)
        self.clip_sample = bool(clip_sample)
        self.time_sampling = time_sampling
        self.time_scale = float(time_scale)

    def sample_times(self, batch: int, device: torch.device) -> Tensor:
        """Training times in [0, 1], one per sample.

        ``logitnormal`` is Stable Diffusion 3's proposal: sigmoid of a standard
        normal, which concentrates on the middle of the path where the velocity
        is hardest to predict and spends less effort near the two ends, where it
        is nearly determined by the endpoint alone. ``uniform`` is the plain
        rectified-flow recipe and the default.
        """
        if self.time_sampling == "logitnormal":
            return torch.sigmoid(torch.randn(batch, device=device))
        return torch.rand(batch, device=device)

    def loss(self, decoder_fn: DecoderFn, actions: Tensor, weights: Tensor) -> tuple[Tensor, dict[str, float]]:
        batch = actions.shape[0]
        noise = torch.randn_like(actions)
        times = self.sample_times(batch, actions.device).to(actions.dtype)
        interpolated = (1 - times.view(batch, 1, 1)) * noise + times.view(batch, 1, 1) * actions
        target = actions - noise
        prediction = decoder_fn(interpolated, times * self.time_scale)
        value = _weighted_mean((prediction - target).pow(2), weights)
        return value, {}

    @torch.no_grad()
    def sample(self, decoder_fn: DecoderFn, shape: tuple[int, int, int], device: torch.device,
               dtype: torch.dtype = torch.float32) -> Tensor:
        sample = torch.randn(shape, device=device, dtype=dtype)
        step = 1.0 / self.num_inference_steps
        for index in range(self.num_inference_steps):
            now = index * step
            batched = torch.full((shape[0],), now * self.time_scale, device=device)
            velocity = decoder_fn(sample, batched).to(dtype)
            if self.clip_sample:
                # The analogue of DDIM's clipping: extrapolate to where this
                # velocity lands at t=1, clamp that endpoint into the range
                # min-max normalisation guarantees, and re-derive the velocity
                # that reaches it. Without this a guided velocity can walk the
                # sample outside the action range one Euler step at a time.
                remaining = 1.0 - now
                endpoint = (sample + remaining * velocity).clamp(-1.0, 1.0)
                velocity = (endpoint - sample) / remaining
            sample = sample + step * velocity
        return sample.clamp(-1.0, 1.0) if self.clip_sample else sample


def build_objective(config: "PolicyConfig") -> nn.Module:
    """The objective named by ``config.objective``, wired from the config."""
    if config.objective == "flow":
        return FlowMatchingObjective(
            num_inference_steps=config.num_inference_steps,
            clip_sample=config.clip_sample,
            time_sampling=config.flow_time_sampling,
        )
    if config.objective == "diffusion":
        return DiffusionObjective(
            num_train_timesteps=config.num_train_timesteps,
            num_inference_steps=config.num_inference_steps,
            schedule=config.beta_schedule,
            clip_sample=config.clip_sample,
        )
    raise ValueError(f"objective must be one of {OBJECTIVES}, got {config.objective!r}")
