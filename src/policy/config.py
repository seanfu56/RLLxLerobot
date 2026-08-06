"""Configuration for the CNN action-chunk policy.

Defaults are the ones from the Diffusion Policy paper, with one deliberate
exception: ``down_dims``. The reference implementation uses (512, 1024, 2048),
which is a 252M-parameter U-Net. On a bundle of 20 episodes (~3.6k frames) that
is roughly one parameter per seventy thousand training samples in the wrong
direction, so the default here is the (128, 256, 512) variant at 17M.

``objective`` picks how the same network is trained and sampled - ``diffusion``
(DDPM/DDIM) or ``flow`` (rectified flow). Nothing else in the config changes
with it; see ``policy.objectives`` for what each one does.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields

from .objectives import BETA_SCHEDULES, FLOW_TIME_SAMPLINGS, OBJECTIVES

# Recorded in the checkpoint so the run picker in ``policy.checkpoints`` can
# describe a run without importing torch. The value is historical - it dates
# from when a second, DINOv2-based family lived alongside this one - and is kept
# so checkpoints written before that family was removed still describe correctly.
POLICY_KIND = "simple"
VISION = "resnet18"
POOL = "spatial_softmax"

DELTA_MODES = ("incremental", "anchored")
ACTION_REPRS = ("absolute", "delta")
NORM_MODES = ("meanstd", "minmax", "identity")
# Which frame of an episode becomes the goal; ``policy.goal`` implements both
# and re-exports this. It lives here because the config is the leaf module -
# policy.goal imports policy.config, not the other way round.
GOAL_SELECTIONS = ("uniform4", "tail", "future_uniform")

# What the unconditional branch of classifier-free guidance drops.
#
# "full"  the whole conditioning vector, observation and goal together, is
#         replaced by ``null_cond``. Guidance then amplifies everything the
#         policy is conditioned on. Needs cond_dropout > 0.
# "goal"  only the goal feature is replaced, by ``null_goal``; the observation
#         and the measured state stay exactly as they are. Guidance then
#         amplifies the goal alone - "do this rather than whatever you would do
#         from this frame anyway" - which is the knob a goal-conditioned rollout
#         actually wants. Needs goal_dropout > 0, which is the same flag that
#         makes a checkpoint runnable with no goal, so most goal-conditioned runs
#         already support it without retraining.
GUIDANCE_MODES = ("full", "goal")

__all__ = [
    "ACTION_REPRS", "BETA_SCHEDULES", "DELTA_MODES", "FLOW_TIME_SAMPLINGS", "GOAL_SELECTIONS",
    "GUIDANCE_MODES", "NORM_MODES", "OBJECTIVES", "POLICY_KIND", "POOL", "PolicyConfig", "VISION",
    "target_stats",
]


@dataclass
class PolicyConfig:
    # --- task shape ---
    state_dim: int = 7
    action_dim: int = 7
    image_size: int = 224

    # --- horizons ---
    # horizon: actions predicted per forward pass.
    # n_obs_steps: observation frames fed in. One frame cannot tell a stationary
    #   gripper from a moving one; two encode velocity, which is why the paper
    #   uses two and the transformer policy's single frame is a real handicap.
    # n_action_steps: default number executed before re-planning (receding horizon).
    horizon: int = 16
    n_obs_steps: int = 2
    n_action_steps: int = 8

    # --- vision ---
    pretrained_backbone: bool = True
    # BatchNorm's running statistics are wrong under EMA and unreliable at these
    # batch sizes; the paper replaces every BatchNorm with GroupNorm.
    use_group_norm: bool = True
    num_keypoints: int = 32
    vision_feature_dim: int = 64

    # --- U-Net ---
    down_dims: tuple[int, ...] = (128, 256, 512)
    kernel_size: int = 5
    n_groups: int = 8
    diffusion_step_embed_dim: int = 128
    use_film_scale_modulation: bool = True

    # --- generative objective ---
    # "diffusion": DDPM training, epsilon prediction, DDIM sampling.
    # "flow":      rectified flow, velocity prediction, Euler sampling.
    # The network, the conditioning and the action representation are identical
    # either way; only what the decoder regresses and how a sample is drawn change.
    objective: str = "diffusion"
    # Sampler steps. Under diffusion the stride is rounded (see DiffusionObjective);
    # under flow matching N means exactly N decoder calls.
    num_inference_steps: int = 10
    # Diffusion only. Flow matching's time is continuous, so it has neither a
    # bucket count nor a noise schedule.
    num_train_timesteps: int = 100
    beta_schedule: str = "cosine"
    # Flow matching only: the distribution training times are drawn from.
    flow_time_sampling: str = "uniform"

    # --- goal conditioning (implemented in policy.goal) ---
    # When set, the model also sees a goal frame drawn from the episode it is
    # training on, encoded by the same ResNet and appended to the conditioning
    # vector. ``goal_dropout`` replaces that frame with a learned null embedding
    # at training time, which is what makes a goal-conditioned checkpoint
    # runnable with no goal at all.
    #
    # ``goal_selection`` decides which frame. "uniform4" samples goal_frames
    # frames evenly across the episode and takes goal_frame_index of them -
    # the same rule src/diffusion generates video under, so the goal a video
    # model hands over at inference is the kind of picture training used.
    # "tail" is the older rule: a picture of the finished task. In both cases
    # the episode's end is drawn from its last ``goal_window`` frames.
    # "future_uniform" draws uniformly from the first frame after the action
    # chunk through the true final frame. Samples without that much future are
    # removed by GoalChunkDataset.
    goal_conditioned: bool = False
    goal_selection: str = "uniform4"
    goal_frames: int = 4
    goal_frame_index: int = 2
    goal_window: int = 10
    goal_dropout: float = 0.0

    # --- conditioning ---
    # State-only ablations can disable the image encoder entirely. The dataset
    # still carries frames for bundle compatibility, but the policy does not
    # read them or allocate vision parameters.
    use_vision: bool = True
    use_proprio: bool = True
    # Classifier-free guidance. Training replaces the whole conditioning vector
    # with a learned null embedding at rate ``cond_dropout``, which is what gives
    # the model an unconditional branch to extrapolate away from at sampling
    # time. Guidance is only meaningful if that branch was actually trained, so
    # a weight other than 1.0 requires cond_dropout > 0.
    cond_dropout: float = 0.0
    guidance_weight: float = 1.0
    # Which conditioning the guided branch drops; see GUIDANCE_MODES. This is a
    # sampling-time choice, not something training changes - "goal" only needs
    # the null-goal embedding goal_dropout already trains - so it is recorded
    # here as the checkpoint's default and can be overridden per rollout.
    guidance_mode: str = "full"

    # --- action representation (see policy.datasets.compute_stats for the delta conventions) ---
    action_repr: str = "absolute"
    delta_mode: str = "incremental"
    absolute_dims: list[int] = field(default_factory=list)
    # Euler-angle action dimensions in degrees. Deltas on these dimensions are
    # wrapped to [-180, 180] before normalization and after integration.
    angular_dims: list[int] = field(default_factory=list)

    # --- normalisation ---
    # min-max maps targets into [-1, 1], which is what the sampler's clipping
    # assumes. Changing action_norm without disabling clipping silently truncates.
    state_norm: str = "minmax"
    action_norm: str = "minmax"
    clip_sample: bool = True
    mask_loss_for_padding: bool = True
    stats: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.down_dims = tuple(int(value) for value in self.down_dims)
        self.angular_dims = [int(value) for value in self.angular_dims]
        if len(set(self.angular_dims)) != len(self.angular_dims):
            raise ValueError(f"angular_dims has duplicates: {self.angular_dims}")
        if any(not 0 <= index < self.action_dim for index in self.angular_dims):
            raise ValueError(f"angular_dims must be within [0, {self.action_dim})")
        if self.action_repr not in ACTION_REPRS:
            raise ValueError(f"action_repr must be one of {ACTION_REPRS}, got {self.action_repr!r}")
        if self.delta_mode not in DELTA_MODES:
            raise ValueError(f"delta_mode must be one of {DELTA_MODES}, got {self.delta_mode!r}")
        if self.state_norm not in NORM_MODES or self.action_norm not in NORM_MODES:
            raise ValueError(f"Normalisation modes must be one of {NORM_MODES}")
        if self.objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {OBJECTIVES}, got {self.objective!r}")
        if self.beta_schedule not in BETA_SCHEDULES:
            raise ValueError(f"beta_schedule must be one of {BETA_SCHEDULES}, got {self.beta_schedule!r}")
        if self.flow_time_sampling not in FLOW_TIME_SAMPLINGS:
            raise ValueError(
                f"flow_time_sampling must be one of {FLOW_TIME_SAMPLINGS}, got {self.flow_time_sampling!r}"
            )
        if self.num_inference_steps < 1:
            raise ValueError(f"num_inference_steps must be >= 1, got {self.num_inference_steps}")
        if self.num_train_timesteps < 1:
            raise ValueError(f"num_train_timesteps must be >= 1, got {self.num_train_timesteps}")
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {self.horizon}")
        if not 1 <= self.n_obs_steps:
            raise ValueError(f"n_obs_steps must be >= 1, got {self.n_obs_steps}")
        if not 1 <= self.n_action_steps <= self.horizon:
            raise ValueError(f"n_action_steps must be in [1, {self.horizon}], got {self.n_action_steps}")
        if not self.down_dims:
            raise ValueError("down_dims must name at least one U-Net stage")

        # Each stage halves the chunk length, so a horizon that is not a multiple
        # of the total factor cannot be concatenated with its skip connection.
        factor = 2 ** len(self.down_dims)
        if self.horizon % factor:
            raise ValueError(
                f"horizon must be a multiple of {factor} for down_dims={self.down_dims}, "
                f"got {self.horizon}. Use horizon {factor * max(1, self.horizon // factor)} "
                f"or drop a stage from --down-dims."
            )
        for width in self.down_dims:
            if width % self.n_groups:
                raise ValueError(f"down_dims entry {width} is not divisible by n_groups={self.n_groups}")

        if self.action_repr == "delta":
            if len(set(self.absolute_dims)) != len(self.absolute_dims):
                raise ValueError(f"absolute_dims has duplicates: {self.absolute_dims}")
            for index in self.absolute_dims:
                if not 0 <= index < self.action_dim:
                    raise ValueError(f"absolute_dims entry {index} is outside [0, {self.action_dim})")
            if len(self.absolute_dims) == self.action_dim:
                raise ValueError("Every dimension is absolute; use --action-repr absolute instead")
        elif self.absolute_dims:
            raise ValueError("absolute_dims only applies to --action-repr delta")

        if self.goal_window < 1:
            raise ValueError(f"goal_window must be >= 1, got {self.goal_window}")
        if self.goal_selection not in GOAL_SELECTIONS:
            raise ValueError(
                f"goal_selection must be one of {GOAL_SELECTIONS}, got {self.goal_selection!r}"
            )
        if self.goal_frames < 2:
            raise ValueError(f"goal_frames must be >= 2, got {self.goal_frames}")
        if not 0 <= self.goal_frame_index < self.goal_frames:
            raise ValueError(
                f"goal_frame_index must be in [0, {self.goal_frames}), got {self.goal_frame_index}"
            )
        if not 0.0 <= self.goal_dropout < 1.0:
            raise ValueError(f"goal_dropout must be in [0, 1), got {self.goal_dropout}")
        if self.goal_dropout > 0.0 and not self.goal_conditioned:
            raise ValueError("goal_dropout only applies to --goal-conditioned runs")

        if not 0.0 <= self.cond_dropout < 1.0:
            raise ValueError(f"cond_dropout must be in [0, 1), got {self.cond_dropout}")
        if self.guidance_mode not in GUIDANCE_MODES:
            raise ValueError(
                f"guidance_mode must be one of {GUIDANCE_MODES}, got {self.guidance_mode!r}"
            )
        if self.guidance_mode == "goal" and not self.goal_conditioned:
            raise ValueError(
                "guidance_mode='goal' drops the goal from the guided branch, which only a "
                "--goal-conditioned run has."
            )
        if self.guidance_weight < 0.0:
            raise ValueError(f"guidance_weight must be >= 0, got {self.guidance_weight}")
        if self.guidance_weight != 1.0:
            reason = self.guidance_unavailable_reason()
            if reason is not None:
                raise ValueError(f"guidance_weight={self.guidance_weight} {reason}")

        if self.action_norm != "minmax" and self.clip_sample:
            raise ValueError(
                "clip_sample assumes targets normalised into [-1, 1]. Use --action-norm minmax "
                "or pass --no-clip-sample."
            )

    @property
    def chunk_size(self) -> int:
        """Alias for ``horizon``; the shared checkpoint picker asks for this name."""
        return self.horizon

    # ------------------------------------------------------------------
    # classifier-free guidance
    # ------------------------------------------------------------------
    #
    # Both questions below are answered here rather than in the model, because
    # the runner, the CLI and the Streamlit pages all have to ask them - the
    # pages from a run.json, without importing torch at all - and a guidance
    # weight applied against an untrained null embedding produces confident
    # wrong actions rather than an error.

    def guidance_unavailable_reason(self, mode: str | None = None) -> str | None:
        """Why guidance cannot be used in ``mode``, or None when it can.

        The requirement is only ever "was this null embedding trained": "full"
        extrapolates away from ``null_cond``, which ``cond_dropout`` trains, and
        "goal" away from ``null_goal``, which ``goal_dropout`` trains.
        """
        mode = self.guidance_mode if mode is None else mode
        if mode not in GUIDANCE_MODES:
            return f"is not a known guidance mode; use one of {GUIDANCE_MODES}."
        if mode == "goal":
            if not self.goal_conditioned:
                return "needs a --goal-conditioned run; this one has no goal to drop."
            if self.goal_dropout <= 0.0:
                return (
                    "needs a trained null-goal embedding, but goal_dropout is 0. Retrain with "
                    "--goal-dropout 0.1, or guide in the 'full' mode."
                )
            return None
        if self.cond_dropout <= 0.0:
            return (
                "needs an unconditional branch, but cond_dropout is 0. Train with "
                "--cond-dropout 0.1 to use classifier-free guidance."
            )
        return None

    def guidance_available(self, mode: str | None = None) -> bool:
        return self.guidance_unavailable_reason(mode) is None

    @property
    def guidance_modes(self) -> tuple[str, ...]:
        """The modes this checkpoint can actually be guided in, possibly none."""
        return tuple(mode for mode in GUIDANCE_MODES if self.guidance_available(mode))

    def to_dict(self) -> dict:
        """Serialised config, plus the descriptive keys the run picker reads.

        ``policy``/``objective``/``chunk_size`` are what ``policy.checkpoints.RunInfo``
        looks for, so writing them here is what makes these runs show up in the
        Streamlit picker next to the transformer ones.
        """
        payload = asdict(self)
        payload["down_dims"] = list(self.down_dims)
        payload.update(
            policy=POLICY_KIND,
            vision=VISION,
            pool=POOL,
            chunk_size=self.horizon,
        )
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "PolicyConfig":
        known = {entry.name for entry in fields(cls)}
        return cls(**{key: value for key, value in payload.items() if key in known})


def target_stats(config: PolicyConfig, stats: dict) -> dict[str, list[float]] | None:
    """Normalisation statistics for whatever the model actually regresses.

    Under ``action_repr="delta"`` each dimension takes its statistics from the
    matching quantity: delta statistics for the relative dimensions, raw action
    statistics for the ones left absolute.
    """
    if config.action_repr != "delta":
        return stats.get("action")
    relative_key = "increment" if config.delta_mode == "incremental" else "delta"
    action, relative = stats.get("action"), stats.get(relative_key)
    if action is None or relative is None:
        if stats:
            raise ValueError(f"Statistics for {relative_key!r} are missing; rebuild the bundle.")
        return None
    absolute = set(config.absolute_dims)
    return {
        key: [
            (action[key][index] if index in absolute else relative[key][index])
            for index in range(config.action_dim)
        ]
        for key in ("mean", "std", "min", "max")
    }
