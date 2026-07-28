"""ResNet-18 + spatial softmax + conditional 1D U-Net over the action chunk.

Three parts, in the order the data flows:

    RgbEncoder          image -> 32 keypoints -> 64-d feature (one per obs step)
    global conditioning [state, feature] over the last n_obs_steps, flattened
    ConditionalUnet1d   noisy action chunk + timestep + conditioning -> target

What the U-Net's output *means* is the objective's business, not this file's:
under ``--objective diffusion`` it is the noise added to the chunk, under
``--objective flow`` it is the velocity of a straight path from noise to data.
Everything about the generative process - the schedule, the training target,
the sampler - lives in ``policy.objectives`` behind two methods, so the network
below is shared verbatim between the two.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .backbones import ImageNormalizer, SpatialSoftmax, replace_batchnorm_with_groupnorm
from .config import PolicyConfig, target_stats
from .modules import Normalizer, sinusoidal_embedding
from .objectives import build_objective


class RgbEncoder(nn.Module):
    """One camera frame -> a short feature vector.

    Spatial softmax is the whole point of the bottleneck: it reports *where* a
    feature fired rather than how strongly, which is the signal a "move down
    until the gripper reaches the object" policy actually needs, and it comes
    out at 64 numbers instead of 25k.
    """

    def __init__(self, config: PolicyConfig):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.IMAGENET1K_V1 if config.pretrained_backbone else None
        model = resnet18(weights=weights)
        if config.use_group_norm:
            model = replace_batchnorm_with_groupnorm(model)

        self.normalizer = ImageNormalizer()
        self.trunk = nn.Sequential(*list(model.children())[:-2])  # (B, 512, h, w)
        self.pool = SpatialSoftmax(512, num_keypoints=config.num_keypoints)
        self.project = nn.Linear(self.pool.output_dim, config.vision_feature_dim)
        self.activation = nn.ReLU()
        self.feature_dim = config.vision_feature_dim

    def forward(self, images: Tensor) -> Tensor:
        features = self.trunk(self.normalizer(images))
        return self.activation(self.project(self.pool(features)))


class Conv1dBlock(nn.Module):
    """Conv1d -> GroupNorm -> Mish, the U-Net's unit of work."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, n_groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class ConditionalResidualBlock1d(nn.Module):
    """Residual block whose activations are FiLM-modulated by the conditioning.

    The conditioning vector (timestep embedding + observation features) produces
    a per-channel scale and bias rather than being concatenated to the input, so
    the observation influences every position of the chunk equally.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
        use_film_scale_modulation: bool = True,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.use_film_scale_modulation = use_film_scale_modulation
        cond_channels = out_channels * 2 if use_film_scale_modulation else out_channels

        self.conv1 = Conv1dBlock(in_channels, out_channels, kernel_size, n_groups)
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, cond_channels))
        self.conv2 = Conv1dBlock(out_channels, out_channels, kernel_size, n_groups)
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        out = self.conv1(x)
        embedding = self.cond_encoder(cond)
        if self.use_film_scale_modulation:
            embedding = embedding.reshape(-1, 2, self.out_channels, 1)
            out = embedding[:, 0] * out + embedding[:, 1]
        else:
            out = out + embedding.unsqueeze(-1)
        return self.conv2(out) + self.residual_conv(x)


class TimestepEncoder(nn.Module):
    """Noise level -> conditioning vector.

    Shared by both objectives: diffusion feeds it a bucket index in
    [0, num_train_timesteps), flow matching a continuous time scaled into the
    same range so the sinusoidal embedding resolves it just as finely.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.Mish(), nn.Linear(dim * 4, dim))

    def forward(self, timesteps: Tensor) -> Tensor:
        return self.mlp(sinusoidal_embedding(timesteps, self.dim))


class ConditionalUnet1d(nn.Module):
    """Temporal U-Net over the action chunk.

    Convolving along time (not attending over it) is what makes this cheap and
    stable to train on little data: the receptive field grows with depth, the
    inductive bias is smoothness, and there are no attention hyper-parameters
    to get wrong.
    """

    def __init__(self, config: PolicyConfig, global_cond_dim: int):
        super().__init__()
        # The config field keeps its historical name; the encoder is objective-agnostic.
        self.step_encoder = TimestepEncoder(config.diffusion_step_embed_dim)
        cond_dim = config.diffusion_step_embed_dim + global_cond_dim
        common = {
            "cond_dim": cond_dim,
            "kernel_size": config.kernel_size,
            "n_groups": config.n_groups,
            "use_film_scale_modulation": config.use_film_scale_modulation,
        }

        in_out = [(config.action_dim, config.down_dims[0])] + list(
            zip(config.down_dims[:-1], config.down_dims[1:], strict=True)
        )

        self.down_modules = nn.ModuleList()
        for index, (dim_in, dim_out) in enumerate(in_out):
            is_last = index >= len(in_out) - 1
            self.down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1d(dim_in, dim_out, **common),
                        ConditionalResidualBlock1d(dim_out, dim_out, **common),
                        nn.Conv1d(dim_out, dim_out, 3, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1d(config.down_dims[-1], config.down_dims[-1], **common),
                ConditionalResidualBlock1d(config.down_dims[-1], config.down_dims[-1], **common),
            ]
        )

        # One up stage per down stage that halved the chunk, so every one of them
        # upsamples and the output comes back out at the full horizon.
        self.up_modules = nn.ModuleList()
        for dim_out, dim_in in reversed(in_out[1:]):
            self.up_modules.append(
                nn.ModuleList(
                    [
                        # dim_in * 2 because the skip connection is concatenated.
                        ConditionalResidualBlock1d(dim_in * 2, dim_out, **common),
                        ConditionalResidualBlock1d(dim_out, dim_out, **common),
                        nn.ConvTranspose1d(dim_out, dim_out, 4, 2, 1),
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            Conv1dBlock(config.down_dims[0], config.down_dims[0], config.kernel_size, config.n_groups),
            nn.Conv1d(config.down_dims[0], config.action_dim, 1),
        )

    def forward(self, sample: Tensor, timesteps: Tensor, global_cond: Tensor) -> Tensor:
        """``sample`` is [B, horizon, action_dim]; the result has the same shape."""
        x = sample.moveaxis(-1, -2)  # (B, action_dim, horizon) for Conv1d
        conditioning = torch.cat([self.step_encoder(timesteps), global_cond], dim=-1)

        skips: list[Tensor] = []
        for residual, residual2, downsample in self.down_modules:
            x = residual(x, conditioning)
            x = residual2(x, conditioning)
            skips.append(x)
            x = downsample(x)

        for block in self.mid_modules:
            x = block(x, conditioning)

        for residual, residual2, upsample in self.up_modules:
            x = torch.cat((x, skips.pop()), dim=1)
            x = residual(x, conditioning)
            x = residual2(x, conditioning)
            x = upsample(x)

        return self.final_conv(x).moveaxis(-1, -2)


class ChunkPolicy(nn.Module):
    """Observation -> action chunk, under whichever objective the config names.

    Two entry points, ``compute_loss`` and ``predict``. Neither branches on the
    objective: both delegate to ``self.objective``, so the training loop, the
    inference runner and the Streamlit hardware process are identical for a
    diffusion run and a flow-matching one.
    """

    def __init__(self, config: PolicyConfig):
        super().__init__()
        self.config = config

        stats = config.stats or {}
        self.state_normalizer = Normalizer(config.state_norm, stats.get("state"), config.state_dim)
        self.action_normalizer = Normalizer(
            config.action_norm, target_stats(config, stats), config.action_dim
        )

        # 1 where a target is measured relative to the current state, 0 where it
        # is an absolute command. All zero when action_repr is "absolute".
        delta_mask = torch.zeros(config.action_dim)
        if config.action_repr == "delta":
            delta_mask[:] = 1.0
            delta_mask[list(config.absolute_dims)] = 0.0
        self.register_buffer("delta_mask", delta_mask)

        self.rgb_encoder = RgbEncoder(config)
        self.global_cond_dim = self._global_cond_dim()
        self.unet = ConditionalUnet1d(config, global_cond_dim=self.global_cond_dim)

        # Stands in for "no observation at all". Always allocated so a checkpoint
        # loads whatever cond_dropout it was trained with, but only meaningful -
        # and only usable for guidance - when that dropout was above zero.
        self.null_cond = nn.Parameter(torch.randn(1, self.global_cond_dim) * 0.02)
        self.objective = build_objective(config)

    # ------------------------------------------------------------------
    # conditioning
    # ------------------------------------------------------------------

    def _global_cond_dim(self) -> int:
        """Width of the conditioning vector, before it reaches the U-Net.

        Called once from ``__init__``, after the vision encoder exists and
        before the U-Net is built. A subclass that conditions on more than the
        observation history - ``policy.goal.GoalChunkPolicy`` - widens the vector
        by overriding this.
        """
        per_step = self.rgb_encoder.feature_dim + (
            self.config.state_dim if self.config.use_proprio else 0
        )
        return per_step * self.config.n_obs_steps

    def observation_features(self, batch: dict[str, Tensor]) -> Tensor:
        """The observation half of the conditioning, [B, per_step * n_obs_steps].

        No dropout here: it is applied once to the finished vector, so that a
        subclass appending its own conditioning is dropped along with this.
        """
        images = batch["image"]  # (B, T, 3, H, W)
        batch_size, steps = images.shape[:2]
        features = self.rgb_encoder(images.flatten(0, 1)).reshape(batch_size, steps, -1)
        parts = [features]
        if self.config.use_proprio:
            state = self.state_normalizer.normalize(batch["state"])  # (B, T, D)
            parts.append(state.to(features.dtype))
        return torch.cat(parts, dim=-1).flatten(1)

    def apply_cond_dropout(self, context: Tensor) -> Tensor:
        """Replace a fraction of the batch's conditioning with the null embedding.

        Training only. Dropping every modality together is what makes the null
        branch a genuine "no observation" model rather than a half-conditioned
        one, and it is what classifier-free guidance extrapolates away from.
        """
        if self.training and self.config.cond_dropout > 0.0:
            keep = (torch.rand(context.shape[0], 1, device=context.device) >= self.config.cond_dropout)
            context = torch.where(keep, context, self.null_cond.to(context.dtype))
        return context

    def encode_context(self, batch: dict[str, Tensor]) -> Tensor:
        """Flat conditioning vector [B, global_cond_dim]."""
        return self.apply_cond_dropout(self.observation_features(batch))

    def unconditional_context(self, batch_size: int, dtype: torch.dtype) -> Tensor:
        return self.null_cond.to(dtype).expand(batch_size, -1)

    def _decoder_fn(self, context: Tensor, guidance_weight: float):
        """The decoder call the sampler drives, guided when the weight is not 1.

        Guidance costs one extra U-Net pass per sampler step, not one extra
        forward pass of the whole policy: the conditional context is encoded once
        and the unconditional one is a learned constant, so the ResNet - the
        expensive half - still runs exactly once per prediction. The two branches
        go through the U-Net as a single batch of 2B.

        The extrapolation below is linear in the U-Net's output, so it is the
        correct guidance rule for both an epsilon prediction and a velocity.
        """
        if guidance_weight == 1.0:
            return lambda action_input, timesteps: self.unet(action_input, timesteps, context)

        uncond = self.unconditional_context(context.shape[0], context.dtype)
        both = torch.cat([context, uncond], dim=0)

        def guided(action_input: Tensor, timesteps: Tensor) -> Tensor:
            doubled = self.unet(
                torch.cat([action_input, action_input], dim=0),
                torch.cat([timesteps, timesteps], dim=0),
                both,
            )
            conditional, unconditional = doubled.chunk(2, dim=0)
            return unconditional + guidance_weight * (conditional - unconditional)

        return guided

    def _current_state(self, batch: dict[str, Tensor]) -> Tensor:
        """The measured state the chunk starts from: the newest observation step."""
        return batch["state"][:, -1]

    # ------------------------------------------------------------------
    # targets
    # ------------------------------------------------------------------

    def _reference(self, batch: dict[str, Tensor]) -> Tensor:
        state = self._current_state(batch).unsqueeze(1)
        if self.config.delta_mode == "anchored":
            return state * self.delta_mask
        previous = torch.cat([state, batch["action"][:, :-1]], dim=1)
        return previous * self.delta_mask

    def _target_actions(self, batch: dict[str, Tensor]) -> Tensor:
        return self.action_normalizer.normalize(batch["action"] - self._reference(batch))

    def _to_absolute(self, actions: Tensor, batch: dict[str, Tensor]) -> Tensor:
        """Invert the target transform; needs no ground truth, unlike ``_reference``."""
        predicted = self.action_normalizer.denormalize(actions)
        state = self._current_state(batch).unsqueeze(1)
        if self.config.delta_mode == "incremental":
            relative = state + predicted.cumsum(dim=1)
        else:
            relative = state + predicted
        return self.delta_mask * relative + (1 - self.delta_mask) * predicted

    # ------------------------------------------------------------------
    # training / inference
    # ------------------------------------------------------------------

    def compute_loss(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        context = self.encode_context(batch)
        targets = self._target_actions(batch)

        weights = torch.ones_like(targets)
        if self.config.mask_loss_for_padding and "action_is_pad" in batch:
            weights = weights * (~batch["action_is_pad"]).unsqueeze(-1).to(weights.dtype)

        def decode(action_input: Tensor, timesteps: Tensor) -> Tensor:
            return self.unet(action_input, timesteps, context)

        loss, metrics = self.objective.loss(decode, targets, weights)
        return loss, {"action_loss": float(loss.detach()), **metrics}

    @torch.no_grad()
    def predict(self, batch: dict[str, Tensor], guidance_weight: float | None = None) -> Tensor:
        """Predicted chunk in raw robot units, [B, horizon, action_dim].

        ``guidance_weight`` overrides the configured one for this call; 1.0 is
        plain conditional sampling.
        """
        weight = self.config.guidance_weight if guidance_weight is None else float(guidance_weight)
        if weight != 1.0 and self.config.cond_dropout <= 0.0:
            raise ValueError(
                "This policy was trained without conditioning dropout, so its unconditional "
                "branch is untrained and guidance would extrapolate from noise. Retrain with "
                "--cond-dropout 0.1 or sample at guidance_weight=1.0."
            )
        context = self.encode_context(batch)
        shape = (context.shape[0], self.config.horizon, self.config.action_dim)
        sampled = self.objective.sample(
            self._decoder_fn(context, weight), shape, context.device, torch.float32
        )
        return self._to_absolute(sampled, batch)

    def parameter_counts(self) -> dict[str, int]:
        """Per-part parameter counts, for the training banner and run.json."""
        encoder = sum(p.numel() for p in self.rgb_encoder.parameters())
        unet = sum(p.numel() for p in self.unet.parameters())
        total = sum(p.numel() for p in self.parameters())
        return {
            "total": total,
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "rgb_encoder": encoder,
            "unet": unet,
        }
