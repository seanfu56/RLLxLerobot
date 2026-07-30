"""The rectified-flow primitives Cosmos3-Nano is trained and sampled under.

``Cosmos3OmniTransformer`` does not take a latent and a timestep the way a plain
U-Net does: it consumes one packed joint text+vision token sequence per forward
call, with mRoPE ids, per-modality sequence indexes and a per-token timestep
vector. Getting that packing right is fiddly and getting it *subtly* wrong is
silent, so it is written once here and shared by everything that needs it:

    src/cosmos/train.py         LoRA supervised fine-tuning
    src/finetune/dpo_cosmos.py  preference fine-tuning (Linear-DPO)
    src/guidance/cosmos.py      classifier-guided sampling

Nothing in this module imports diffusers. The pipeline is passed in, and only
its own helpers (``tokenize_prompt``, ``_prepare_text_segment``,
``_prepare_vision_segment``) are called, so the packing can never drift from
what the installed pipeline does at inference.

The flow convention throughout, matching the Cosmos3 training recipe:

    x_sigma = (1 - sigma) * clean + sigma * noise        sigma = 0 is data
    v       = noise - clean                              the regression target

from which ``clean = x_sigma - sigma * v`` and ``noise = x_sigma + (1 - sigma) * v``.
``src/guidance`` needs both of those inversions to turn a classifier gradient
into a velocity correction.
"""

from __future__ import annotations

import torch

# Mode scale of the Waver video-time sampler used by the Cosmos3 training recipe.
WAVER_MODE_SCALE = 1.29
# The transformer takes timesteps in [0, 1000] rather than sigma in [0, 1].
TIMESTEP_SCALE = 1000.0

__all__ = [
    "TIMESTEP_SCALE",
    "WAVER_MODE_SCALE",
    "build_condition_mask",
    "clean_from_velocity",
    "noise_from_velocity",
    "predict_velocity",
    "sample_train_sigma",
    "shifted_sigmas",
]


def build_condition_mask(num_latent_frames: int, device, dtype) -> torch.Tensor:
    """Latent frame 0 is the clean image condition; the rest are noised.

    Matches the ``vision_condition_mask`` that ``prepare_latents`` builds at
    inference, so training and image-to-video sampling agree on which frames
    the model is asked to predict.
    """
    mask = torch.zeros((num_latent_frames, 1, 1), device=device, dtype=dtype)
    mask[0, 0, 0] = 1.0
    return mask


def sample_train_sigma(device: torch.device, shift: float) -> torch.Tensor:
    """Draw one noise level from the Waver video-time schedule, with flow shift."""
    u = torch.rand((1,), device=device, dtype=torch.float32)
    t = 1 - u - WAVER_MODE_SCALE * (torch.cos(torch.pi * u / 2).square() - 1 + u)
    sigma = shift * t / (1 + (shift - 1) * t)
    return sigma.view(1, 1, 1, 1, 1)


def shifted_sigmas(steps: int, shift: float, device: torch.device) -> torch.Tensor:
    """Descending sampling schedule from pure noise to data, with flow shift.

    ``steps + 1`` values from 1 down to 0: the extra entry is the endpoint, so
    an Euler integrator takes exactly ``steps`` decoder calls. The shift is the
    same reparameterisation the trainer applies to its training times, which is
    what keeps the noise levels a guided sampler visits inside the range the
    model was actually trained at.
    """
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    if shift <= 0:
        raise ValueError(f"shift must be positive, got {shift}")
    linear = torch.linspace(1.0, 0.0, steps + 1, device=device, dtype=torch.float32)
    return shift * linear / (1 + (shift - 1) * linear)


def clean_from_velocity(x_sigma: torch.Tensor, velocity: torch.Tensor, sigma: float) -> torch.Tensor:
    """The data endpoint this velocity points at, ``x0 = x_sigma - sigma * v``."""
    return x_sigma - sigma * velocity


def noise_from_velocity(x_sigma: torch.Tensor, velocity: torch.Tensor, sigma: float) -> torch.Tensor:
    """The noise endpoint, ``eps = x_sigma + (1 - sigma) * v``."""
    return x_sigma + (1.0 - sigma) * velocity


def predict_velocity(
    pipe,
    transformer,
    x_sigma: torch.Tensor,
    caption: str,
    sigma: torch.Tensor,
    *,
    device: torch.device,
    dit_dtype: torch.dtype,
    fps: float,
    num_frames: int,
    resolution: int,
) -> torch.Tensor:
    """The transformer's velocity prediction for one noised latent.

    ``x_sigma`` is ``(1, C, T, H, W)`` - the transformer packs a single clip per
    forward call and has no batch dimension - and the result has the same shape.
    Only the noised frames are meaningful; the caller masks the conditioning
    frame out with ``build_condition_mask``.
    """
    cond_input_ids, _ = pipe.tokenize_prompt(
        caption,
        negative_prompt=None,
        num_frames=num_frames,
        height=resolution,
        width=resolution,
        fps=fps,
        use_system_prompt=True,
        add_resolution_template=True,
        add_duration_template=True,
    )
    text_segment = pipe._prepare_text_segment(cond_input_ids, device=device)
    vision_segment = pipe._prepare_vision_segment(
        input_vision_tokens=x_sigma,
        has_image_condition=True,
        mrope_offset=text_segment["vision_start_temporal_offset"],
        vision_fps=fps,
        curr=text_segment["und_len"],
        device=device,
    )
    position_ids = torch.cat(
        [text_segment["text_mrope_ids"], vision_segment["vision_mrope_ids"]], dim=1
    )
    sequence_length = text_segment["und_len"] + vision_segment["num_vision_tokens"]

    vision_timestep = sigma.flatten()[0].to(device=device, dtype=torch.float32) * TIMESTEP_SCALE
    vision_timesteps = vision_timestep.expand(vision_segment["num_noisy_vision_tokens"])

    output = transformer(
        input_ids=text_segment["input_ids"],
        text_indexes=text_segment["text_indexes"],
        position_ids=position_ids,
        und_len=text_segment["und_len"],
        sequence_length=sequence_length,
        vision_tokens=[x_sigma.to(dit_dtype)],
        vision_token_shapes=vision_segment["vision_token_shapes"],
        vision_sequence_indexes=vision_segment["vision_sequence_indexes"],
        vision_mse_loss_indexes=vision_segment["vision_mse_loss_indexes"],
        vision_timesteps=vision_timesteps,
        vision_noisy_frame_indexes=vision_segment["vision_noisy_frame_indexes"],
    )
    # `.sample` holds the per-item vision velocity predictions; there is exactly
    # one item because the transformer packs a single clip per forward call.
    return output.sample[0].float()
