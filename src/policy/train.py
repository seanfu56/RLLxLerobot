#!/usr/bin/env python3
"""Train the CNN action-chunk policy on a prepared bundle.

    python src/policy/train.py --bundle pick-bar --run-name S_pick_bar

``--objective`` picks the generative process the same network is trained under:
``diffusion`` (DDPM/DDIM, the default and the published recipe) or ``flow``
(rectified flow, velocity prediction, Euler sampling). Nothing else changes -
same encoder, same U-Net, same conditioning, same action representation.

``--goal-conditioned`` adds a goal frame, drawn from the end of the episode, to
what the policy conditions on; see ``src/policy/goal.py``.

Writes one directory per run - run.json, log.jsonl, best.pt / final.pt /
step_*.pt - which is what ``sweeps/collect_results.py`` and the Streamlit
checkpoint picker read.

Validation reports two error numbers. ``val_mae_ratio`` is the mean absolute
error relative to a policy that just holds the current pose. ``val_mae_bon`` is
the best of ``--eval-samples`` sampled chunks: a diffusion policy is allowed to
commit to one of several valid behaviours, and a plain MAE punishes a correct
but different choice exactly as hard as a wrong one.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from policy.config import (
        ACTION_REPRS, DELTA_MODES, FLOW_TIME_SAMPLINGS, GOAL_SELECTIONS, OBJECTIVES, PolicyConfig,
    )
    from policy.datasets import Bundle, ChunkDataset, compute_stats, default_drop_n_last_frames
    from policy.goal import (
        DEFAULT_GOAL_FRAME_INDEX, DEFAULT_GOAL_FRAMES, DEFAULT_GOAL_WINDOW,
        GoalChunkDataset, GoalChunkPolicy,
    )
    from policy.model import ChunkPolicy
else:
    from .config import (
        ACTION_REPRS, DELTA_MODES, FLOW_TIME_SAMPLINGS, GOAL_SELECTIONS, OBJECTIVES, PolicyConfig,
    )
    from .datasets import Bundle, ChunkDataset, compute_stats, default_drop_n_last_frames
    from .goal import (
        DEFAULT_GOAL_FRAME_INDEX, DEFAULT_GOAL_FRAMES, DEFAULT_GOAL_WINDOW,
        GoalChunkDataset, GoalChunkPolicy,
    )
    from .model import ChunkPolicy

DEFAULT_ABSOLUTE_DIMS = ("gripper.pos",)

# Flags only one objective reads. Passing one of these to the other objective is
# a silent no-op otherwise, which is exactly the sort of thing that costs a run:
# the defaults live here so ``main`` can tell "not set" from "set to the default"
# and warn about the difference.
OBJECTIVE_ONLY_DEFAULTS = {
    "diffusion": {"num_train_timesteps": 100, "beta_schedule": "cosine"},
    "flow": {"flow_time_sampling": "uniform"},
}

# Same idea for the goal-conditioning flags, which do nothing at all without
# --goal-conditioned.
GOAL_ONLY_DEFAULTS = {
    "goal_selection": "uniform4",
    "goal_frames": DEFAULT_GOAL_FRAMES,
    "goal_frame_index": DEFAULT_GOAL_FRAME_INDEX,
    "goal_window": DEFAULT_GOAL_WINDOW,
    "goal_dropout": 0.0,
}


def resolve_absolute_dims(values: list[str] | None, joint_names: list[str]) -> list[int]:
    """Map ``--absolute-dims`` entries (joint names or indices) to dimension indices.

    Names the user typed have to exist; the default is best effort, so a bundle with
    a different joint layout simply gets no absolute dimensions rather than an error.
    """
    explicit = values is not None
    entries = list(values) if explicit else list(DEFAULT_ABSOLUTE_DIMS)
    resolved: list[int] = []
    for entry in entries:
        text = str(entry)
        if text.lstrip("+-").isdigit():
            index = int(text)
            if not 0 <= index < len(joint_names):
                raise SystemExit(f"--absolute-dims {index} is outside [0, {len(joint_names)})")
            resolved.append(index)
        elif text in joint_names:
            resolved.append(joint_names.index(text))
        elif explicit:
            raise SystemExit(f"--absolute-dims {text!r} is not one of {joint_names}")
    return sorted(set(resolved))


class ExponentialMovingAverage:
    """Shadow copy of the trainable weights; almost always better at eval time."""

    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = decay
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module, step: int) -> None:
        # Ramp the decay in so early (noisy) weights are forgotten quickly.
        decay = min(self.decay, (1 + step) / (10 + step))
        for name, parameter in model.named_parameters():
            if name in self.shadow:
                self.shadow[name].lerp_(parameter.detach(), 1 - decay)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {name: value.clone() for name, value in self.shadow.items()}

    def swap_in(self, model: torch.nn.Module) -> dict[str, torch.Tensor]:
        backup = {}
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name in self.shadow:
                    backup[name] = parameter.detach().clone()
                    parameter.copy_(self.shadow[name])
        return backup

    def swap_out(self, model: torch.nn.Module, backup: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name in backup:
                    parameter.copy_(backup[name])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    data = parser.add_argument_group("data")
    data.add_argument("--bundle", required=True, help="Bundle name under --data-root, or a path to a bundle dir")
    data.add_argument("--data-root", type=Path, default=Path("piper-data/dataset"))
    data.add_argument("--output-root", type=Path, default=Path("models/simple"))
    data.add_argument("--run-name", default=None, help="Defaults to <bundle>_simple_h<horizon>")
    data.add_argument("--val-episodes", type=int, default=3, help="Held-out episodes per source (0 disables)")
    data.add_argument("--num-workers", type=int, default=8)
    data.add_argument("--overwrite", action="store_true")

    shape = parser.add_argument_group("horizons")
    shape.add_argument("--horizon", type=int, default=16, help="Actions predicted per forward pass")
    shape.add_argument("--n-obs-steps", type=int, default=1, help="Observation frames fed in (2 encodes velocity)")
    shape.add_argument("--n-action-steps", type=int, default=8, help="Actions executed before re-planning")
    shape.add_argument("--drop-n-last-frames", type=int, default=None,
                       help="Default: horizon - n_action_steps - n_obs_steps + 1")

    arch = parser.add_argument_group("architecture")
    arch.add_argument("--down-dims", type=int, nargs="+", default=[64, 128, 256],
                      help="U-Net stage widths. 17M params at the default, 5M at 64 128 256")
    arch.add_argument("--kernel-size", type=int, default=5)
    arch.add_argument("--n-groups", type=int, default=8)
    arch.add_argument("--num-keypoints", type=int, default=32)
    arch.add_argument("--vision-feature-dim", type=int, default=64)
    arch.add_argument("--diffusion-step-embed-dim", type=int, default=128)
    arch.add_argument("--random-init-backbone", action="store_true", help="Skip ImageNet weights (debug only)")
    arch.add_argument("--no-group-norm", dest="use_group_norm", action="store_false",
                      help="Keep BatchNorm in the ResNet; unreliable at these batch sizes and under EMA")
    arch.add_argument("--no-proprio", dest="use_proprio", action="store_false",
                      help="Condition on images only")
    arch.add_argument("--no-film-scale", dest="use_film_scale_modulation", action="store_false",
                      help="FiLM bias only, no scale")

    objective = parser.add_argument_group("objective")
    objective.add_argument("--objective", choices=OBJECTIVES, default="flow",
                           help="diffusion: DDPM training, epsilon prediction, DDIM sampling. "
                                "flow: rectified flow, velocity prediction, Euler sampling")
    objective.add_argument("--num-inference-steps", type=int, default=10,
                           help="Sampler steps. Exact under flow matching; rounded under diffusion")
    objective.add_argument("--num-train-timesteps", type=int,
                           default=OBJECTIVE_ONLY_DEFAULTS["diffusion"]["num_train_timesteps"],
                           help="Diffusion only; flow matching's time is continuous")
    objective.add_argument("--beta-schedule", choices=("cosine", "linear"),
                           default=OBJECTIVE_ONLY_DEFAULTS["diffusion"]["beta_schedule"],
                           help="Diffusion only; flow matching has no noise schedule")
    objective.add_argument("--flow-time-sampling", choices=FLOW_TIME_SAMPLINGS,
                           default=OBJECTIVE_ONLY_DEFAULTS["flow"]["flow_time_sampling"],
                           help="Flow matching only. logitnormal concentrates training on the "
                                "middle of the noise-to-data path (Stable Diffusion 3's proposal)")
    objective.add_argument("--no-clip-sample", dest="clip_sample", action="store_false")
    objective.add_argument("--cond-dropout", type=float, default=0.0,
                           help="Rate at which the whole observation is replaced by a learned null "
                                "embedding. Required (0.1 is typical) for classifier-free guidance")
    objective.add_argument("--guidance-weight", type=float, default=1.0,
                           help="Guidance used at validation and stored as the checkpoint's default. "
                                "1.0 is plain conditional sampling; needs --cond-dropout above 0")
    objective.add_argument("--action-repr", choices=ACTION_REPRS, default="delta",
                           help="absolute: predict joint targets. delta: predict them relative to the state")
    objective.add_argument("--delta-mode", choices=DELTA_MODES, default="incremental")
    objective.add_argument("--absolute-dims", nargs="*", default=None, metavar="DIM",
                           help="Dimensions kept absolute under --action-repr delta (default: gripper.pos)")
    objective.add_argument("--state-norm", choices=("meanstd", "minmax", "identity"), default="minmax")
    objective.add_argument("--action-norm", choices=("meanstd", "minmax", "identity"), default="minmax")
    objective.add_argument("--no-mask-padding", dest="mask_loss_for_padding", action="store_false",
                           help="Include copy-padded chunk steps in the loss")

    goal = parser.add_argument_group("goal conditioning")
    goal.add_argument("--goal-conditioned", action="store_true",
                      help="Also condition on a goal frame drawn from the episode "
                           "(see src/policy/goal.py)")
    goal.add_argument("--goal-selection", choices=GOAL_SELECTIONS,
                      default=GOAL_ONLY_DEFAULTS["goal_selection"],
                      help="uniform4: sample --goal-frames evenly across the episode and take "
                           "--goal-frame-index of them, the same rule src/diffusion generates "
                           "video under. tail: a picture of the finished task. future_uniform: "
                           "draw uniformly between current+horizon and the final frame")
    goal.add_argument("--goal-frames", type=int, default=GOAL_ONLY_DEFAULTS["goal_frames"],
                      help="Frames sampled across the episode under --goal-selection uniform4")
    goal.add_argument("--goal-frame-index", type=int,
                      default=GOAL_ONLY_DEFAULTS["goal_frame_index"],
                      help="Which of those frames is the goal, counting from 0. The default 2 "
                           "is the third of four, about two thirds of the way through")
    goal.add_argument("--goal-window", type=int, default=GOAL_ONLY_DEFAULTS["goal_window"],
                      help="The episode's end is drawn from its last N frames")
    goal.add_argument("--goal-dropout", type=float, default=GOAL_ONLY_DEFAULTS["goal_dropout"],
                      help="Rate at which the goal is replaced by a learned null embedding. "
                           "Above 0 the checkpoint can also be run with no goal at all, which "
                           "is how to measure what the goal is contributing")

    optimization = parser.add_argument_group("optimisation")
    optimization.add_argument("--steps", type=int, default=6000)
    optimization.add_argument("--batch-size", type=int, default=64)
    optimization.add_argument("--lr", type=float, default=1e-4)
    optimization.add_argument("--min-lr", type=float, default=1e-6)
    optimization.add_argument("--warmup-steps", type=int, default=500)
    optimization.add_argument("--weight-decay", type=float, default=1e-6)
    optimization.add_argument("--grad-clip", type=float, default=1.0)
    optimization.add_argument("--ema-decay", type=float, default=0.999, help="0 disables EMA")
    optimization.add_argument("--no-augment", dest="augment", action="store_false")
    optimization.add_argument("--crop-scale", type=float, default=1.0,
                              help="Lower bound of the random crop. 1.0 disables cropping")
    optimization.add_argument("--color-jitter", type=float, default=0.1)
    optimization.add_argument("--amp", choices=("bf16", "fp16", "off"), default="bf16")
    optimization.add_argument("--seed", type=int, default=1000)
    optimization.add_argument("--device", default="cuda")

    logging_group = parser.add_argument_group("logging")
    logging_group.add_argument("--log-freq", type=int, default=50)
    logging_group.add_argument("--eval-freq", type=int, default=250)
    logging_group.add_argument("--eval-batches", type=int, default=8)
    logging_group.add_argument("--eval-samples", type=int, default=8,
                               help="Chunks sampled per observation for the best-of-N error")
    logging_group.add_argument("--save-freq", type=int, default=5000)
    logging_group.add_argument("--wandb-project", default=None)

    return parser.parse_args(argv)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_bundle_dir(args: argparse.Namespace) -> Path:
    candidate = Path(args.bundle)
    if (candidate / "bundle.json").is_file():
        return candidate
    resolved = args.data_root / args.bundle
    if (resolved / "bundle.json").is_file():
        return resolved
    raise SystemExit(
        f"No bundle at {candidate} or {resolved}. Run python src/policy/data_prep.py --bundles {args.bundle}"
    )


def build_config(args: argparse.Namespace, bundle: Bundle, stats: dict) -> PolicyConfig:
    absolute_dims = (
        resolve_absolute_dims(args.absolute_dims, bundle.joint_names)
        if args.action_repr == "delta"
        else []
    )
    return PolicyConfig(
        state_dim=bundle.state_dim,
        action_dim=bundle.action_dim,
        image_size=bundle.image_size,
        horizon=args.horizon,
        n_obs_steps=args.n_obs_steps,
        n_action_steps=args.n_action_steps,
        pretrained_backbone=not args.random_init_backbone,
        use_group_norm=args.use_group_norm,
        num_keypoints=args.num_keypoints,
        vision_feature_dim=args.vision_feature_dim,
        down_dims=tuple(args.down_dims),
        kernel_size=args.kernel_size,
        n_groups=args.n_groups,
        diffusion_step_embed_dim=args.diffusion_step_embed_dim,
        use_film_scale_modulation=args.use_film_scale_modulation,
        objective=args.objective,
        num_train_timesteps=args.num_train_timesteps,
        num_inference_steps=args.num_inference_steps,
        beta_schedule=args.beta_schedule,
        flow_time_sampling=args.flow_time_sampling,
        goal_conditioned=args.goal_conditioned,
        goal_selection=args.goal_selection,
        goal_frames=args.goal_frames,
        goal_frame_index=args.goal_frame_index,
        goal_window=args.goal_window,
        goal_dropout=args.goal_dropout,
        use_proprio=args.use_proprio,
        cond_dropout=args.cond_dropout,
        guidance_weight=args.guidance_weight,
        action_repr=args.action_repr,
        delta_mode=args.delta_mode,
        absolute_dims=absolute_dims,
        state_norm=args.state_norm,
        action_norm=args.action_norm,
        clip_sample=args.clip_sample,
        mask_loss_for_padding=args.mask_loss_for_padding,
        stats=stats,
    )


def lr_at(step: int, args: argparse.Namespace) -> float:
    if step < args.warmup_steps:
        return args.lr * (step + 1) / max(1, args.warmup_steps)
    progress = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return args.min_lr + (args.lr - args.min_lr) * cosine


def to_device(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


@torch.no_grad()
def evaluate(
    model: ChunkPolicy,
    loader: DataLoader,
    device: str,
    max_batches: int,
    joint_names: list[str],
    autocast,
    eval_samples: int = 1,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    absolute_errors: list[torch.Tensor] = []
    hold_errors: list[torch.Tensor] = []
    best_of_n: list[float] = []

    for index, batch in enumerate(loader):
        if index >= max_batches:
            break
        batch = to_device(batch, device)
        with autocast():
            loss, _ = model.compute_loss(batch)
        losses.append(float(loss))

        valid = (~batch["action_is_pad"]).unsqueeze(-1).float()
        divisor = valid.sum(dim=(0, 1)).clamp_min(1)

        prediction = model.predict(batch)
        error = ((prediction - batch["action"]).abs() * valid).sum(dim=(0, 1)) / divisor
        absolute_errors.append(error.float().cpu())

        # "Hold the current pose" reference. Chunk actions sit a fraction of a
        # degree from the current state, so a raw MAE always looks small; what
        # matters is how much of that gap the policy closes.
        current_state = batch["state"][:, -1].unsqueeze(1)
        hold = ((current_state - batch["action"]).abs() * valid).sum(dim=(0, 1)) / divisor
        hold_errors.append(hold.float().cpu())

        if eval_samples > 1:
            # Per-sample error of the closest of N draws: credits the policy for
            # picking any valid mode rather than the one the operator happened to
            # demonstrate. valid is (B, horizon, 1) and the error is summed over
            # joints too, so the divisor counts every element, not just the steps.
            divisor_per_sample = valid.expand_as(batch["action"]).sum(dim=(1, 2)).clamp_min(1)

            def sample_error(chunk: torch.Tensor) -> torch.Tensor:
                return ((chunk - batch["action"]).abs() * valid).sum(dim=(1, 2)) / divisor_per_sample

            # The chunk already sampled above counts as the first draw; sampling
            # is the expensive part of evaluation.
            draws = [sample_error(prediction)]
            draws += [sample_error(model.predict(batch)) for _ in range(eval_samples - 1)]
            best_of_n.append(float(torch.stack(draws).min(dim=0).values.mean()))

    model.train()
    if not losses:
        return {}

    mean_error = torch.stack(absolute_errors).mean(dim=0)
    mean_hold = torch.stack(hold_errors).mean(dim=0)
    metrics = {
        "val_loss": float(np.mean(losses)),
        "val_mae": float(mean_error.mean()),
        "val_mae_hold": float(mean_hold.mean()),
        # < 1 beats standing still, > 1 is worse than doing nothing
        "val_mae_ratio": float(mean_error.mean() / max(float(mean_hold.mean()), 1e-8)),
    }
    if best_of_n:
        metrics["val_mae_bon"] = float(np.mean(best_of_n))
        metrics["val_mae_bon_ratio"] = float(np.mean(best_of_n) / max(float(mean_hold.mean()), 1e-8))
    for name, value, reference in zip(joint_names, mean_error.tolist(), mean_hold.tolist(), strict=True):
        metrics[f"val_mae/{name}"] = value
        metrics[f"val_mae_hold/{name}"] = reference
    return metrics


def warn_about_ignored_flags(args: argparse.Namespace) -> list[str]:
    """Name the flags this run's settings make inert."""
    ignored = []
    for objective, defaults in OBJECTIVE_ONLY_DEFAULTS.items():
        if objective == args.objective:
            continue
        for name, default in defaults.items():
            if getattr(args, name) != default:
                flag = "--" + name.replace("_", "-")
                ignored.append(flag)
                print(f"warning: {flag} is a {objective} setting and does nothing under "
                      f"--objective {args.objective}")
    if not args.goal_conditioned:
        for name, default in GOAL_ONLY_DEFAULTS.items():
            if getattr(args, name) != default:
                flag = "--" + name.replace("_", "-")
                ignored.append(flag)
                print(f"warning: {flag} does nothing without --goal-conditioned")
    return ignored


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    warn_about_ignored_flags(args)
    set_seed(args.seed)

    bundle_dir = resolve_bundle_dir(args)
    bundle = Bundle(bundle_dir)
    train_episodes, val_episodes = bundle.split_episodes(args.val_episodes, args.seed)
    stats = compute_stats(bundle, train_episodes, args.horizon)

    run_name = args.run_name or f"{bundle.name}_simple_h{args.horizon}"
    output_dir = args.output_root / run_name
    if output_dir.exists() and not args.overwrite:
        raise SystemExit(f"Output already exists: {output_dir}. Pass --run-name or --overwrite.")
    output_dir.mkdir(parents=True, exist_ok=True)

    drop_n_last = (
        args.drop_n_last_frames
        if args.drop_n_last_frames is not None
        else default_drop_n_last_frames(args.horizon, args.n_action_steps, args.n_obs_steps)
    )
    dataset_kwargs = dict(
        horizon=args.horizon,
        n_obs_steps=args.n_obs_steps,
        drop_n_last_frames=drop_n_last,
        seed=args.seed,
    )
    dataset_class = GoalChunkDataset if args.goal_conditioned else ChunkDataset
    if args.goal_conditioned:
        dataset_kwargs.update(
            goal_window=args.goal_window,
            goal_selection=args.goal_selection,
            goal_frames=args.goal_frames,
            goal_frame_index=args.goal_frame_index,
        )

    train_dataset = dataset_class(
        bundle, train_episodes, augment=args.augment, crop_scale=args.crop_scale,
        color_jitter=args.color_jitter, **dataset_kwargs
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=True, drop_last=True, persistent_workers=args.num_workers > 0,
    )
    val_loader = None
    if val_episodes:
        # Validation pins the goal to the final frame: a goal that moves between
        # evaluations makes val_loss noisy, and val_loss is what selects best.pt.
        if args.goal_conditioned:
            dataset_kwargs["random_goal"] = False
        val_dataset = dataset_class(bundle, val_episodes, augment=False, **dataset_kwargs)
        val_loader = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=min(2, args.num_workers), pin_memory=True,
        )

    config = build_config(args, bundle, stats)
    device = args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu"
    model = (GoalChunkPolicy if config.goal_conditioned else ChunkPolicy)(config).to(device)
    counts = model.parameter_counts()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        # The paper's betas: a longer first-moment window than torch's default,
        # which matters because the diffusion loss is noisy at every step.
        betas=(0.95, 0.999),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )
    ema = ExponentialMovingAverage(model, args.ema_decay) if args.ema_decay > 0 else None

    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(args.amp)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp == "fp16" and device.startswith("cuda"))

    def autocast():
        if amp_dtype is None or not device.startswith("cuda"):
            return nullcontext()
        return torch.autocast("cuda", dtype=amp_dtype)

    metadata = {
        "run_name": run_name,
        "bundle": str(bundle_dir),
        "bundle_manifest": bundle.manifest,
        "args": {key: (str(value) if isinstance(value, Path) else value) for key, value in vars(args).items()},
        "config": config.to_dict(),
        "train_episodes": [episode.name + "@" + episode.source for episode in train_episodes],
        "val_episodes": [episode.name + "@" + episode.source for episode in val_episodes],
        "parameters": counts,
        "drop_n_last_frames": drop_n_last,
    }
    (output_dir / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    log_file = (output_dir / "log.jsonl").open("a", buffering=1)

    writer = None
    if args.wandb_project:
        import wandb

        writer = wandb.init(project=args.wandb_project, name=run_name, config=metadata["args"])

    absolute_names = [bundle.joint_names[index] for index in config.absolute_dims]
    action_description = args.action_repr + (
        f"/{args.delta_mode} (absolute: {', '.join(absolute_names) or 'none'})"
        if args.action_repr == "delta"
        else ""
    )
    print(f"run          {run_name}")
    print(f"bundle       {bundle.name}: {len(train_episodes)} train / {len(val_episodes)} val episodes, "
          f"{len(train_dataset)} samples")
    print(f"horizons     predict {args.horizon} | observe {args.n_obs_steps} | execute {args.n_action_steps} "
          f"| drop last {drop_n_last}")
    process = (
        f"flow matching, {args.flow_time_sampling} time / {args.num_inference_steps} Euler steps"
        if args.objective == "flow"
        else f"diffusion {args.num_train_timesteps} train ({args.beta_schedule}) / "
             f"{args.num_inference_steps} DDIM steps"
    )
    print(f"action       {action_description}")
    print(f"objective    {process}")
    print(f"unet         {tuple(args.down_dims)} kernel {args.kernel_size}")
    print(f"guidance     cond dropout {args.cond_dropout} | weight {args.guidance_weight}"
          f"{' (plain conditional)' if args.guidance_weight == 1.0 else ''}")
    if args.goal_conditioned:
        if args.goal_selection == "uniform4":
            goal_rule = (
                f"frame {args.goal_frame_index} of {args.goal_frames} sampled evenly, "
                f"ending in the last {args.goal_window} frames"
            )
        elif args.goal_selection == "future_uniform":
            goal_rule = "uniformly after the action chunk through the final frame"
        else:
            goal_rule = f"the episode's last {args.goal_window} frames"
        print(f"goal         {goal_rule} | "
              f"dropout {args.goal_dropout}"
              f"{' (needs a goal at inference)' if args.goal_dropout == 0.0 else ''}")
    print(f"augment      crop >= {args.crop_scale if args.augment else 1.0}, "
          f"jitter {args.color_jitter if args.augment else 0.0}")
    print(f"parameters   {counts['total']/1e6:.1f}M total "
          f"({counts['rgb_encoder']/1e6:.1f}M encoder + {counts['unet']/1e6:.1f}M U-Net)")
    print(f"output       {output_dir}\n")

    def save(tag: str, step: int) -> None:
        payload = {
            "step": step,
            # Descriptive only; the config below is what actually rebuilds the model.
            "kind": f"simple-{config.objective}" + ("-goal" if config.goal_conditioned else ""),
            "config": config.to_dict(),
            "model": {key: value.cpu() for key, value in model.state_dict().items()},
            "ema": ema.state_dict() if ema else None,
            "joint_names": bundle.joint_names,
            "fps": bundle.fps,
        }
        torch.save(payload, output_dir / f"{tag}.pt")

    def batches():
        while True:
            yield from train_loader

    stream = batches()
    model.train()
    best_val = float("inf")
    running: list[float] = []
    start_time = time.time()

    for step in range(args.steps):
        learning_rate = lr_at(step, args)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

        batch = to_device(next(stream), device)
        with autocast():
            loss, metrics = model.compute_loss(batch)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        if ema:
            ema.update(model, step)

        running.append(float(loss.detach()))
        if (step + 1) % args.log_freq == 0:
            record = {
                "step": step + 1,
                "loss": float(np.mean(running[-args.log_freq :])),
                "lr": learning_rate,
                "elapsed_s": round(time.time() - start_time, 1),
                **metrics,
            }
            log_file.write(json.dumps(record) + "\n")
            if writer:
                writer.log(record, step=step + 1)
            print(f"step {step+1:6d} | loss {record['loss']:.4f} | lr {learning_rate:.2e} | "
                  f"{record['elapsed_s']:.0f}s", flush=True)

        if val_loader and args.eval_freq > 0 and (step + 1) % args.eval_freq == 0:
            backup = ema.swap_in(model) if ema else None
            evaluation = evaluate(
                model, val_loader, device, args.eval_batches, bundle.joint_names, autocast, args.eval_samples
            )
            if ema:
                ema.swap_out(model, backup)
            if evaluation:
                record = {"step": step + 1, **evaluation}
                log_file.write(json.dumps(record) + "\n")
                if writer:
                    writer.log(record, step=step + 1)
                best = (
                    f" | best-of-{args.eval_samples} {evaluation['val_mae_bon']:.3f} "
                    f"(ratio {evaluation['val_mae_bon_ratio']:.2f})"
                    if "val_mae_bon" in evaluation
                    else ""
                )
                print(f"  eval {step+1:6d} | val_loss {evaluation['val_loss']:.4f} | "
                      f"MAE {evaluation['val_mae']:.3f} (hold {evaluation['val_mae_hold']:.3f}, "
                      f"ratio {evaluation['val_mae_ratio']:.2f}){best}", flush=True)
                if evaluation["val_loss"] < best_val:
                    best_val = evaluation["val_loss"]
                    save("best", step + 1)

        if args.save_freq > 0 and (step + 1) % args.save_freq == 0:
            save(f"step_{step+1:06d}", step + 1)

    save("final", args.steps)
    log_file.close()
    if writer:
        writer.finish()
    print(f"\ndone in {(time.time()-start_time)/60:.1f} min | best val_loss {best_val:.4f}")
    print(f"checkpoints: {output_dir}")


if __name__ == "__main__":
    main()
