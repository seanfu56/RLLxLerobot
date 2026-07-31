"""Training-free DiverseFlow inference for Cosmos3-Nano.

The sampler couples ``K`` rectified-flow trajectories through the
determinantal point process (DPP) objective from:

    Morshed and Boddeti, "DiverseFlow: Sample-Efficient Diverse Mode
    Coverage in Flows", CVPR 2025.

Cosmos uses the reverse time convention of the paper::

    x_sigma = (1 - sigma) * clean + sigma * noise
    v         = noise - clean

and samples from ``sigma=1`` to ``sigma=0``.  If ``g`` is the gradient of the
DPP log-likelihood, its guided velocity is therefore ``v - gamma * g``.  Since
the Euler step has a negative ``delta_sigma``, this moves the latent in the
positive (likelihood-increasing) gradient direction.

The flow transformer is deliberately evaluated without autograd.  The
first-order endpoint estimate treats its velocity as fixed, then differentiates
the DPP objective with respect to the current particles.  This matches the
paper's practical memory accounting (autoencoder + feature ViT backpropagation)
and avoids retaining a 16B-parameter transformer graph for every particle.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from statistics import NormalDist
from typing import Sequence

import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor

from .flow import (
    build_condition_mask,
    clean_from_velocity,
    noise_from_velocity,
    predict_velocity,
    shifted_sigmas,
)

LOGGER = logging.getLogger("cosmos3-diverse-flow")

FEATURES = ("latent", "dino")
DINO_VARIANTS = ("small", "base", "large", "giant")


@dataclass(frozen=True)
class DiverseFlowConfig:
    """Configuration of the coupled Euler sampler.

    ``diversity_scale`` is the paper's ``W`` in
    ``gamma(sigma) = W * sigma / ||grad log L||``.  The text-to-image
    experiments use 20; video is substantially higher-dimensional, so the CLI
    starts at the more conservative value 1.
    """

    steps: int = 35
    cfg_scale: float = 6.0
    flow_shift: float = 10.0
    diversity_scale: float = 1.0
    feature: str = "latent"
    feature_frames: int = 4
    dino_variant: str = "large"
    dpp_bandwidth: float = 1.0
    dpp_jitter: float = 1e-4
    diversity_every: int = 1
    quality_percentile: float = 0.995
    quality_floor: float = 1e-4
    quality_strength: float = 1.0
    gradient_epsilon: float = 1e-8
    log_every: int = 5

    def __post_init__(self) -> None:
        if self.steps < 1:
            raise ValueError(f"steps must be positive, got {self.steps}")
        if self.cfg_scale < 0:
            raise ValueError(f"cfg_scale must be non-negative, got {self.cfg_scale}")
        if self.flow_shift <= 0:
            raise ValueError(f"flow_shift must be positive, got {self.flow_shift}")
        if self.diversity_scale < 0:
            raise ValueError(
                f"diversity_scale must be non-negative, got {self.diversity_scale}"
            )
        if self.feature not in FEATURES:
            raise ValueError(f"feature must be one of {FEATURES}, got {self.feature!r}")
        if self.feature_frames < 1:
            raise ValueError(
                f"feature_frames must be positive, got {self.feature_frames}"
            )
        if self.dino_variant not in DINO_VARIANTS:
            raise ValueError(
                f"dino_variant must be one of {DINO_VARIANTS}, got {self.dino_variant!r}"
            )
        if self.dpp_bandwidth <= 0:
            raise ValueError(
                f"dpp_bandwidth must be positive, got {self.dpp_bandwidth}"
            )
        if self.dpp_jitter <= 0:
            raise ValueError(f"dpp_jitter must be positive, got {self.dpp_jitter}")
        if self.diversity_every < 1:
            raise ValueError(
                f"diversity_every must be positive, got {self.diversity_every}"
            )
        if self.quality_percentile != 0 and not 0.5 < self.quality_percentile < 1:
            raise ValueError(
                "quality_percentile must be 0 (disabled) or between 0.5 and 1, "
                f"got {self.quality_percentile}"
            )
        if not 0 < self.quality_floor <= 1:
            raise ValueError(
                f"quality_floor must be in (0, 1], got {self.quality_floor}"
            )
        if self.quality_strength < 0:
            raise ValueError(
                f"quality_strength must be non-negative, got {self.quality_strength}"
            )
        if self.gradient_epsilon <= 0:
            raise ValueError(
                f"gradient_epsilon must be positive, got {self.gradient_epsilon}"
            )
        if self.log_every < 1:
            raise ValueError(f"log_every must be positive, got {self.log_every}")


@dataclass(frozen=True)
class DiversityStep:
    """Diagnostics from one DPP-gradient evaluation."""

    objective: float
    distance_scale: float
    raw_gradient_norm: float
    minimum_quality: float


def _as_feature_matrix(features: Tensor) -> Tensor:
    if features.ndim < 2:
        raise ValueError(
            f"features must have a particle and feature dimension, got {tuple(features.shape)}"
        )
    if features.shape[0] < 2:
        raise ValueError("DiverseFlow needs at least two particles")
    return features.float().flatten(start_dim=1)


def dpp_kernel(
    features: Tensor,
    *,
    quality: Tensor | None = None,
    bandwidth: float = 1.0,
    distance_scale: Tensor | float | None = None,
    epsilon: float = 1e-8,
) -> tuple[Tensor, Tensor]:
    """Construct the RBF DPP kernel and return its distance normalization.

    The features are L2-normalized before distances are measured.  When no
    explicit ``distance_scale`` is supplied, the median upper-triangle distance
    is used as in DiverseFlow.  It is detached on purpose: differentiating the
    median makes the normalized distance of a two-particle set identically one,
    which gives ``K=2`` no repulsive gradient at all.
    """
    if bandwidth <= 0:
        raise ValueError(f"bandwidth must be positive, got {bandwidth}")
    if epsilon <= 0:
        raise ValueError(f"epsilon must be positive, got {epsilon}")

    matrix = F.normalize(_as_feature_matrix(features), dim=1, eps=epsilon)
    squared_distances = torch.cdist(matrix, matrix, p=2).square()

    if distance_scale is None:
        indexes = torch.triu_indices(
            matrix.shape[0], matrix.shape[0], offset=1, device=matrix.device
        )
        scale = squared_distances[indexes[0], indexes[1]].median().detach()
    else:
        scale = torch.as_tensor(
            distance_scale, device=matrix.device, dtype=squared_distances.dtype
        ).detach()
    scale = scale.clamp_min(epsilon)

    kernel = torch.exp(-float(bandwidth) * squared_distances / scale)
    if quality is not None:
        quality = quality.to(device=kernel.device, dtype=kernel.dtype).flatten()
        if quality.shape != (matrix.shape[0],):
            raise ValueError(
                f"quality must have shape ({matrix.shape[0]},), got {tuple(quality.shape)}"
            )
        kernel = kernel * quality[:, None] * quality[None, :]
    return kernel, scale


def dpp_log_likelihood(kernel: Tensor, *, jitter: float = 1e-4) -> Tensor:
    """Stable ``log det(L) - log det(L + I)`` for a DPP L-ensemble."""
    if kernel.ndim != 2 or kernel.shape[0] != kernel.shape[1]:
        raise ValueError(f"kernel must be square, got {tuple(kernel.shape)}")
    if jitter <= 0:
        raise ValueError(f"jitter must be positive, got {jitter}")

    identity = torch.eye(kernel.shape[0], device=kernel.device, dtype=kernel.dtype)
    numerator_sign, numerator = torch.linalg.slogdet(kernel + jitter * identity)
    denominator_sign, denominator = torch.linalg.slogdet(kernel + identity)
    if bool((numerator_sign <= 0).detach()) or bool((denominator_sign <= 0).detach()):
        raise FloatingPointError("DPP kernel is not positive definite after adding jitter")
    return numerator - denominator


def _chi_square_radius_squared(dimensions: int, percentile: float) -> float:
    """Wilson-Hilferty approximation to a chi-square quantile."""
    if dimensions < 1:
        raise ValueError(f"dimensions must be positive, got {dimensions}")
    z = NormalDist().inv_cdf(percentile)
    correction = 1.0 - 2.0 / (9.0 * dimensions)
    spread = math.sqrt(2.0 / (9.0 * dimensions))
    return dimensions * max(correction + z * spread, 0.0) ** 3


def gaussian_source_quality(
    predicted_source: Tensor,
    condition_mask: Tensor,
    *,
    percentile: float,
    floor: float,
    strength: float,
) -> Tensor | None:
    """DiverseFlow's Gaussian-prior quality term for generated positions only.

    The image-condition latent is deliberately excluded: it is fixed clean data,
    not a draw from the Gaussian source distribution.
    """
    if percentile == 0:
        return None
    if predicted_source.ndim != 5:
        raise ValueError(
            "predicted_source must be KxCxTxHxW, got "
            f"{tuple(predicted_source.shape)}"
        )
    generated_mask = 1.0 - condition_mask.to(
        device=predicted_source.device, dtype=predicted_source.dtype
    )
    generated_temporal = int(generated_mask.sum().detach().item())
    dimensions = (
        predicted_source.shape[1]
        * generated_temporal
        * predicted_source.shape[3]
        * predicted_source.shape[4]
    )
    if dimensions < 1:
        raise ValueError("The condition mask leaves no generated latent positions")

    squared_norm = (predicted_source * generated_mask).float().square().flatten(1).sum(1)
    radius_squared = _chi_square_radius_squared(dimensions, percentile)
    excess = (squared_norm - radius_squared).clamp_min(0.0)
    return torch.exp(-float(strength) * excess).clamp_min(float(floor))


def latent_trajectory_features(
    predicted_clean: Tensor,
    condition_mask: Tensor,
    *,
    temporal_features: int,
    spatial_features: int = 8,
) -> Tensor:
    """Compact order-preserving features from generated Cosmos latents.

    This is the inexpensive fallback when differentiable pixel decoding is too
    costly.  Adaptive pooling retains coarse temporal order and spatial layout
    while avoiding a DPP distance over every latent scalar.
    """
    if predicted_clean.ndim != 5:
        raise ValueError(
            f"predicted_clean must be KxCxTxHxW, got {tuple(predicted_clean.shape)}"
        )
    generated = predicted_clean * (
        1.0
        - condition_mask.to(
            device=predicted_clean.device, dtype=predicted_clean.dtype
        )
    )
    # The first temporal latent is the clean image condition and must not make
    # every particle appear artificially similar.
    generated = generated[:, :, 1:]
    if generated.shape[2] < 1:
        raise ValueError("DiverseFlow needs at least one generated latent frame")
    output_size = (
        min(temporal_features, generated.shape[2]),
        min(spatial_features, generated.shape[3]),
        min(spatial_features, generated.shape[4]),
    )
    return F.adaptive_avg_pool3d(generated.float(), output_size).flatten(1)


def _denormalize_latent(pipe, latent: Tensor) -> Tensor:
    dtype = pipe.vae.dtype
    mean = getattr(pipe, "_vae_latents_mean", None)
    inv_std = getattr(pipe, "_vae_latents_inv_std", None)
    if mean is not None and inv_std is not None:
        mean = mean.to(device=latent.device, dtype=dtype)
        inv_std = inv_std.to(device=latent.device, dtype=dtype)
        return (
            latent.to(dtype) / inv_std.view(1, -1, 1, 1, 1)
            + mean.view(1, -1, 1, 1, 1)
        )

    config = getattr(pipe.vae, "config", None)
    stored_mean = getattr(config, "latents_mean", None)
    stored_std = getattr(config, "latents_std", None)
    if stored_mean is None or stored_std is None:
        raise RuntimeError("The Cosmos VAE does not expose its latent normalization")
    mean = torch.tensor(stored_mean, device=latent.device, dtype=dtype)
    std = torch.tensor(stored_std, device=latent.device, dtype=dtype)
    return (
        latent.to(dtype) * std.view(1, -1, 1, 1, 1)
        + mean.view(1, -1, 1, 1, 1)
    )


def decode_video_tensor(pipe, latent: Tensor) -> Tensor:
    """Differentiably decode normalized Cosmos latents to ``B×3×T×H×W`` RGB."""
    decoded = pipe.vae.decode(_denormalize_latent(pipe, latent))
    video = decoded.sample if hasattr(decoded, "sample") else decoded[0]
    return video.float()


class DinoTrajectoryFeatures:
    """Differentiable DINOv2 descriptors from ordered generated video frames."""

    def __init__(
        self,
        variant: str,
        *,
        frames: int,
        device: torch.device,
        image_size: int = 224,
    ):
        from dino.data import IMAGENET_MEAN, IMAGENET_STD
        from dino.encoder import PATCH_SIZE, VARIANTS, _load_backbone

        if variant not in VARIANTS:
            raise ValueError(f"Unknown DINOv2 variant {variant!r}")
        if frames < 1:
            raise ValueError(f"frames must be positive, got {frames}")
        if image_size % PATCH_SIZE:
            raise ValueError(
                f"image_size must be divisible by {PATCH_SIZE}, got {image_size}"
            )

        entrypoint, self.embed_dim = VARIANTS[variant]
        self.variant = variant
        self.frames = int(frames)
        self.device = device
        self.image_size = int(image_size)
        self.model = _load_backbone(entrypoint).to(device).eval()
        self.model.requires_grad_(False)
        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    def __call__(self, video: Tensor) -> Tensor:
        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError(f"Expected Bx3xTxHxW decoded video, got {tuple(video.shape)}")
        if video.shape[2] < 2:
            raise ValueError("DINO trajectory features need a generated frame")

        count = min(self.frames, video.shape[2] - 1)
        indexes = torch.linspace(
            1, video.shape[2] - 1, count, device=video.device
        ).round().long()
        selected = video.index_select(2, indexes).permute(0, 2, 1, 3, 4)
        batch, frames, channels, height, width = selected.shape
        selected = selected.reshape(batch * frames, channels, height, width)
        selected = selected.add(1.0).mul(0.5).clamp(0.0, 1.0)
        selected = F.interpolate(
            selected,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        selected = (selected - self.mean) / self.std

        autocast = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if self.device.type == "cuda"
            else torch.autocast("cpu", enabled=False)
        )
        with autocast:
            output = self.model.forward_features(selected)
        per_frame = F.normalize(output["x_norm_clstoken"].float(), dim=-1)
        return F.normalize(per_frame.view(batch, frames * self.embed_dim), dim=-1)


def _objective(
    features: Tensor,
    predicted_source: Tensor,
    condition_mask: Tensor,
    config: DiverseFlowConfig,
) -> tuple[Tensor, Tensor, Tensor | None]:
    quality = gaussian_source_quality(
        predicted_source,
        condition_mask,
        percentile=config.quality_percentile,
        floor=config.quality_floor,
        strength=config.quality_strength,
    )
    kernel, distance_scale = dpp_kernel(
        features,
        quality=quality,
        bandwidth=config.dpp_bandwidth,
        epsilon=config.gradient_epsilon,
    )
    return (
        dpp_log_likelihood(kernel, jitter=config.dpp_jitter),
        distance_scale,
        quality,
    )


def diversity_gradients(
    particles: Tensor,
    velocities: Tensor,
    sigma: float,
    condition_mask: Tensor,
    config: DiverseFlowConfig,
) -> tuple[Tensor, DiversityStep]:
    """Return joint-normalized ``grad log DPP`` for every current particle."""
    if particles.shape != velocities.shape or particles.ndim != 5:
        raise ValueError(
            "particles and velocities must share KxCxTxHxW shape, got "
            f"{tuple(particles.shape)} and {tuple(velocities.shape)}"
        )
    if particles.shape[0] < 2:
        raise ValueError("DiverseFlow needs at least two particles")
    if not 0 <= sigma <= 1:
        raise ValueError(f"sigma must be in [0, 1], got {sigma}")

    fixed_velocity = velocities.detach()
    if config.feature == "latent":
        with torch.enable_grad():
            current = particles.detach().requires_grad_(True)
            predicted_clean = clean_from_velocity(current, fixed_velocity, sigma)
            predicted_source = noise_from_velocity(current, fixed_velocity, sigma)
            features = latent_trajectory_features(
                predicted_clean,
                condition_mask,
                temporal_features=config.feature_frames,
            )
            objective, distance_scale, quality = _objective(
                features, predicted_source, condition_mask, config
            )
            (gradient,) = torch.autograd.grad(objective, current)
    else:
        raise ValueError(
            "diversity_gradients handles latent features; use "
            "dino_diversity_gradients for feature='dino'"
        )

    generated_mask = 1.0 - condition_mask.to(
        device=gradient.device, dtype=gradient.dtype
    )
    gradient = gradient * generated_mask
    raw_norm = gradient.float().square().sum().sqrt()
    normalized = gradient / raw_norm.clamp_min(config.gradient_epsilon)
    minimum_quality = (
        1.0 if quality is None else float(quality.detach().min().cpu())
    )
    return normalized.detach(), DiversityStep(
        objective=float(objective.detach().cpu()),
        distance_scale=float(distance_scale.detach().cpu()),
        raw_gradient_norm=float(raw_norm.detach().cpu()),
        minimum_quality=minimum_quality,
    )


def dino_diversity_gradients(
    pipe,
    particles: Tensor,
    velocities: Tensor,
    sigma: float,
    condition_mask: Tensor,
    config: DiverseFlowConfig,
    dino: DinoTrajectoryFeatures,
) -> tuple[Tensor, DiversityStep]:
    """Memory-bounded DINO DPP gradient using per-particle decoder VJPs."""
    fixed_velocity = velocities.detach()
    predicted_clean = clean_from_velocity(particles.detach(), fixed_velocity, sigma)
    predicted_source = noise_from_velocity(particles.detach(), fixed_velocity, sigma)

    with torch.no_grad():
        feature_values = torch.cat(
            [
                dino(decode_video_tensor(pipe, predicted_clean[index : index + 1]))
                for index in range(predicted_clean.shape[0])
            ],
            dim=0,
        )

    with torch.enable_grad():
        feature_variables = feature_values.detach().requires_grad_(True)
        source_variables = predicted_source.detach().requires_grad_(True)
        objective, distance_scale, quality = _objective(
            feature_variables, source_variables, condition_mask, config
        )
        feature_gradient, source_gradient = torch.autograd.grad(
            objective,
            (feature_variables, source_variables),
            allow_unused=True,
        )

    gradients: list[Tensor] = []
    for index in range(predicted_clean.shape[0]):
        with torch.enable_grad():
            endpoint = predicted_clean[index : index + 1].detach().requires_grad_(True)
            features = dino(decode_video_tensor(pipe, endpoint))
            (endpoint_gradient,) = torch.autograd.grad(
                features,
                endpoint,
                grad_outputs=feature_gradient[index : index + 1],
            )
        if source_gradient is not None:
            endpoint_gradient = endpoint_gradient + source_gradient[index : index + 1]
        gradients.append(endpoint_gradient.detach())

    gradient = torch.cat(gradients, dim=0)
    generated_mask = 1.0 - condition_mask.to(
        device=gradient.device, dtype=gradient.dtype
    )
    gradient = gradient * generated_mask
    raw_norm = gradient.float().square().sum().sqrt()
    normalized = gradient / raw_norm.clamp_min(config.gradient_epsilon)
    minimum_quality = (
        1.0 if quality is None else float(quality.detach().min().cpu())
    )
    return normalized.detach(), DiversityStep(
        objective=float(objective.detach().cpu()),
        distance_scale=float(distance_scale.detach().cpu()),
        raw_gradient_norm=float(raw_norm.detach().cpu()),
        minimum_quality=minimum_quality,
    )


@torch.no_grad()
def encode_condition(
    pipe,
    image: Image.Image,
    *,
    num_frames: int,
    resolution: int,
    device: torch.device,
) -> Tensor:
    """Encode a repeated first frame exactly like Cosmos image conditioning."""
    video = pipe.video_processor.preprocess_video(
        [image] * num_frames, height=resolution, width=resolution
    ).to(device=device, dtype=pipe.vae.dtype)
    return pipe._encode_video(video).float()


def sample_diverse_latents(
    pipe,
    condition_latent: Tensor,
    caption: str,
    negative_prompt: str,
    seeds: Sequence[int],
    config: DiverseFlowConfig,
    *,
    device: torch.device,
    fps: float,
    num_frames: int,
    resolution: int,
    dino: DinoTrajectoryFeatures | None = None,
) -> list[Tensor]:
    """Synchronously Euler-integrate a set of coupled Cosmos trajectories."""
    if not seeds:
        raise ValueError("At least one seed is required")
    if len(set(int(seed) for seed in seeds)) != len(seeds):
        raise ValueError("DiverseFlow particles need distinct initial-noise seeds")
    if config.diversity_scale > 0 and len(seeds) < 2:
        raise ValueError("Positive DiverseFlow guidance needs at least two particles")

    pipe.transformer.eval().requires_grad_(False)
    pipe.vae.eval().requires_grad_(False)
    condition_latent = condition_latent.to(device=device, dtype=torch.float32)
    condition_mask = build_condition_mask(
        condition_latent.shape[2], device=device, dtype=condition_latent.dtype
    )

    initial: list[Tensor] = []
    for seed in seeds:
        generator = torch.Generator(device=device).manual_seed(int(seed))
        noise = torch.randn(
            condition_latent.shape,
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        initial.append(
            condition_latent * condition_mask + noise * (1.0 - condition_mask)
        )
    particles = torch.cat(initial, dim=0)

    if (
        dino is None
        and config.feature == "dino"
        and config.diversity_scale > 0
    ):
        dino = DinoTrajectoryFeatures(
            config.dino_variant,
            frames=config.feature_frames,
            device=device,
        )

    sigmas = shifted_sigmas(config.steps, config.flow_shift, device)
    dit_dtype = pipe.transformer.dtype
    for index in range(config.steps):
        sigma = float(sigmas[index])
        next_sigma = float(sigmas[index + 1])
        sigma_tensor = torch.full(
            (1, 1, 1, 1, 1), sigma, device=device, dtype=torch.float32
        )

        predicted: list[Tensor] = []
        with torch.no_grad():
            for particle in particles.split(1, dim=0):
                conditional = predict_velocity(
                    pipe,
                    pipe.transformer,
                    particle,
                    caption,
                    sigma_tensor,
                    device=device,
                    dit_dtype=dit_dtype,
                    fps=fps,
                    num_frames=num_frames,
                    resolution=resolution,
                )
                if config.cfg_scale == 1.0:
                    velocity = conditional
                else:
                    unconditional = predict_velocity(
                        pipe,
                        pipe.transformer,
                        particle,
                        negative_prompt,
                        sigma_tensor,
                        device=device,
                        dit_dtype=dit_dtype,
                        fps=fps,
                        num_frames=num_frames,
                        resolution=resolution,
                    )
                    velocity = unconditional + config.cfg_scale * (
                        conditional - unconditional
                    )
                predicted.append(velocity)
        velocities = torch.cat(predicted, dim=0)

        step_stats: DiversityStep | None = None
        if (
            config.diversity_scale > 0
            and index % config.diversity_every == 0
            and sigma > 0
        ):
            if config.feature == "dino":
                assert dino is not None
                gradient, step_stats = dino_diversity_gradients(
                    pipe,
                    particles,
                    velocities,
                    sigma,
                    condition_mask,
                    config,
                    dino,
                )
            else:
                gradient, step_stats = diversity_gradients(
                    particles,
                    velocities,
                    sigma,
                    condition_mask,
                    config,
                )
            velocities = velocities - config.diversity_scale * sigma * gradient

        with torch.no_grad():
            particles = particles + (next_sigma - sigma) * velocities
            particles = (
                condition_latent * condition_mask
                + particles * (1.0 - condition_mask)
            )

        if step_stats is not None and (
            index == 0
            or (index + 1) % config.log_every == 0
            or index + 1 == config.steps
        ):
            LOGGER.info(
                "DiverseFlow step %d/%d sigma=%.4f objective=%.4f "
                "grad=%.4g quality>=%.4g",
                index + 1,
                config.steps,
                sigma,
                step_stats.objective,
                step_stats.raw_gradient_norm,
                step_stats.minimum_quality,
            )

    return [particle.contiguous() for particle in particles.split(1, dim=0)]


@torch.no_grad()
def decode_latents(pipe, latents: Sequence[Tensor]) -> list[list[Image.Image]]:
    """Decode each final particle separately to bound peak VAE memory."""
    videos: list[list[Image.Image]] = []
    for latent in latents:
        decoded = decode_video_tensor(pipe, latent)
        frames = pipe.video_processor.postprocess_video(
            decoded, output_type="pil"
        )[0]
        videos.append(frames)
    return videos


def generate_diverse(
    pipe,
    image: Image.Image,
    caption: str,
    negative_prompt: str,
    seeds: Sequence[int],
    config: DiverseFlowConfig,
    *,
    device: torch.device,
    fps: float,
    num_frames: int,
    resolution: int,
    dino: DinoTrajectoryFeatures | None = None,
) -> list[list[Image.Image]]:
    """Encode one first frame, run the coupled sampler, and decode all particles."""
    condition_latent = encode_condition(
        pipe,
        image,
        num_frames=num_frames,
        resolution=resolution,
        device=device,
    )
    latents = sample_diverse_latents(
        pipe,
        condition_latent,
        caption,
        negative_prompt,
        seeds,
        config,
        device=device,
        fps=fps,
        num_frames=num_frames,
        resolution=resolution,
        dino=dino,
    )
    return decode_latents(pipe, latents)
