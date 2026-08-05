#!/usr/bin/env python3
"""Fine-tune the official SmolVLA action expert on Piper bundles.

Example::

    python src/smolvla/train.py --bundle pick-can-all \
        --pretrained lerobot/smolvla_base --horizon 16 --steps 20000

The VLM and vision encoder remain frozen.  The LeRobot SmolVLA forward pass is
the official flow-matching loss; this script only supplies local data and
language tokens.  Install the model stack with ``pip install -e
'path/to/lerobot[smolvla]'``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from policy.datasets import Bundle
from smolvla.dataset import SmolVLADataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", required=True, help="Bundle name or bundle directory")
    p.add_argument("--data-root", type=Path, default=Path("piper-data/dataset"))
    p.add_argument("--pretrained", default="lerobot/smolvla_base")
    p.add_argument("--output-dir", type=Path, default=Path("models/smolvla"))
    p.add_argument("--steps", type=int, default=20_000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--horizon", type=int, default=10, help="Action chunk length")
    p.add_argument("--n-action-steps", type=int, default=None)
    p.add_argument("--val-episodes", type=int, default=3)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--min-lr", type=float, default=2.5e-6)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--decay-steps", type=int, default=30_000)
    p.add_argument("--weight-decay", type=float, default=1e-10)
    p.add_argument("--grad-clip", type=float, default=10.0)
    p.add_argument("--amp", choices=("bf16", "off"), default="bf16")
    p.add_argument("--goal-selection", choices=("future_uniform", "uniform4", "tail"),
                   default="future_uniform")
    p.add_argument("--no-goal", action="store_true",
                   help="Train with current image + state + language only")
    p.add_argument("--goal-window", type=int, default=10)
    p.add_argument("--goal-frames", type=int, default=4)
    p.add_argument("--goal-frame-index", type=int, default=2)
    p.add_argument("--log-freq", type=int, default=100)
    p.add_argument("--eval-freq", type=int, default=500)
    p.add_argument("--eval-batches", type=int, default=32)
    p.add_argument("--persistent-workers", action="store_true", default=True,
                   help="Keep data workers alive between epochs (default: enabled)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--task-prefix", default="")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def bundle_path(root: Path, name: str) -> Path:
    path = Path(name)
    return path if path.is_dir() else root / name


def main() -> None:
    args = parse_args()
    if args.horizon < 1 or args.horizon > 50:
        raise SystemExit("--horizon must be in [1, 50] for the pretrained SmolVLA action expert")
    if args.n_action_steps is not None and not 1 <= args.n_action_steps <= args.horizon:
        raise SystemExit("--n-action-steps must be in [1, horizon]")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir = args.output_dir.expanduser()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise SystemExit(f"{args.output_dir} is not empty; pass --overwrite")

    bundle = Bundle(bundle_path(args.data_root, args.bundle))
    train_eps, val_eps = bundle.split_episodes(args.val_episodes, args.seed)
    dataset_kwargs = {
        "horizon": args.horizon,
        "goal_selection": args.goal_selection,
        "goal_window": args.goal_window,
        "goal_frames": args.goal_frames,
        "goal_frame_index": args.goal_frame_index,
        "task_prefix": args.task_prefix,
        "include_goal": not args.no_goal,
    }
    train_data = SmolVLADataset(bundle, train_eps, **dataset_kwargs)
    val_data = SmolVLADataset(bundle, val_eps, random_goal=False, **dataset_kwargs)
    loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True, drop_last=True,
                        persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, drop_last=False,
                            persistent_workers=args.num_workers > 0)

    try:
        from lerobot.configs.types import FeatureType, PolicyFeature
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
    except ImportError as exc:
        raise SystemExit("SmolVLA needs LeRobot with the smolvla extra installed") from exc

    policy = SmolVLAPolicy.from_pretrained(args.pretrained)
    policy.config.chunk_size = args.horizon
    policy.config.n_action_steps = args.n_action_steps or min(args.horizon, policy.config.n_action_steps)
    policy.config.n_obs_steps = 1
    image_shape = (3, bundle.image_size, bundle.image_size)
    policy.config.input_features = {
        "observation.images.current": PolicyFeature(type=FeatureType.VISUAL, shape=image_shape),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(bundle.state_dim,)),
    }
    if not args.no_goal:
        policy.config.input_features["observation.images.goal"] = PolicyFeature(
            type=FeatureType.VISUAL, shape=image_shape
        )
    policy.config.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(bundle.action_dim,)),
    }
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    policy.to(device)
    tokenizer = policy.model.vlm_with_expert.processor.tokenizer
    token_cache: dict[str, dict[str, torch.Tensor]] = {}

    # Bundle-level aggregate stats are not sufficient after an episode split;
    # calculate training-only mean/std to avoid leaking validation trajectories.
    states = np.concatenate([bundle.states[e.shard][e.start:e.end] for e in train_eps])
    actions = np.concatenate([bundle.actions[e.shard][e.start:e.end] for e in train_eps])
    state_mean, state_std = states.mean(0), np.maximum(states.std(0), 1e-6)
    action_mean, action_std = actions.mean(0), np.maximum(actions.std(0), 1e-6)
    state_mean = torch.tensor(state_mean, dtype=torch.float32, device=device)
    state_std = torch.tensor(state_std, dtype=torch.float32, device=device)
    action_mean = torch.tensor(action_mean, dtype=torch.float32, device=device)
    action_std = torch.tensor(action_std, dtype=torch.float32, device=device)

    trainable = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    policy.config.optimizer_lr = args.lr
    policy.config.scheduler_decay_lr = args.min_lr
    policy.config.scheduler_warmup_steps = args.warmup_steps
    policy.config.scheduler_decay_steps = args.decay_steps
    scheduler = policy.config.get_scheduler_preset().build(optimizer, args.steps)
    use_amp = args.amp == "bf16" and device.type == "cuda"

    def prepare_batch(batch: dict) -> dict:
        texts = batch.pop("task")
        for text in set(texts):
            if text not in token_cache:
                encoded = tokenizer([text], padding=True, truncation=True,
                                    max_length=policy.config.tokenizer_max_length,
                                    return_tensors="pt")
                token_cache[text] = {
                    "tokens": encoded.input_ids,
                    "mask": encoded.attention_mask.bool(),
                }
        batch[OBS_LANGUAGE_TOKENS] = torch.cat(
            [token_cache[text]["tokens"] for text in texts], dim=0
        )
        batch[OBS_LANGUAGE_ATTENTION_MASK] = torch.cat(
            [token_cache[text]["mask"] for text in texts], dim=0
        )
        batch["observation.state"] = (batch["observation.state"].to(device) - state_mean) / state_std
        batch["action"] = (batch["action"].to(device) - action_mean) / action_std
        return {
            key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }

    local_config = {
        "bundle": str(bundle.dir),
        "horizon": args.horizon,
        "n_action_steps": policy.config.n_action_steps,
        "goal_selection": args.goal_selection,
        "goal_window": args.goal_window,
        "goal_frames": args.goal_frames,
        "goal_frame_index": args.goal_frame_index,
        "goal_conditioned": not args.no_goal,
        "image_size": bundle.image_size,
        "action_names": list(bundle.joint_names),
        "state_mean": state_mean.cpu().tolist(),
        "state_std": state_std.cpu().tolist(),
        "action_mean": action_mean.cpu().tolist(),
        "action_std": action_std.cpu().tolist(),
    }

    def save_checkpoint(path: Path, checkpoint_step: int, val_loss: float | None) -> None:
        path.mkdir(parents=True, exist_ok=True)
        policy.save_pretrained(path)
        payload = {**local_config, "step": checkpoint_step, "val_flow_loss": val_loss}
        (path / "smolvla_local_config.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

    @torch.no_grad()
    def evaluate() -> float:
        policy.eval()
        losses: list[float] = []
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(args.seed + 1)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(args.seed + 1)
            for index, val_batch in enumerate(val_loader):
                if index >= args.eval_batches:
                    break
                val_batch = prepare_batch(val_batch)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                    val_loss, _ = policy(val_batch)
                losses.append(float(val_loss))
        policy.train()
        if not losses:
            raise RuntimeError("Validation loader produced no batches")
        return float(np.mean(losses))

    step = 0
    best_val = float("inf")
    policy.train()
    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            batch = prepare_batch(batch)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
                loss, _ = policy(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            optimizer.step()
            scheduler.step()
            step += 1
            if step % args.log_freq == 0 or step == 1:
                current_lr = optimizer.param_groups[0]["lr"]
                print(f"step={step}/{args.steps} loss={loss.item():.6f} lr={current_lr:.3e}",
                      flush=True)
            if args.eval_freq > 0 and (step % args.eval_freq == 0 or step == args.steps):
                val_loss = evaluate()
                improved = val_loss < best_val
                marker = " best" if improved else ""
                print(f"step={step}/{args.steps} val_flow_loss={val_loss:.6f}{marker}", flush=True)
                if improved:
                    best_val = val_loss
                    save_checkpoint(args.output_dir, step, val_loss)

    if best_val < float("inf"):
        final_dir = args.output_dir / "final"
        save_checkpoint(final_dir, step, None)
        print(f"saved best {args.output_dir} (val_flow_loss={best_val:.6f})")
        print(f"saved final {final_dir}")
    else:
        save_checkpoint(args.output_dir, step, None)
        print(f"saved {args.output_dir}")


if __name__ == "__main__":
    main()
