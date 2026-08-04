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
import math
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
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--horizon", type=int, default=16, help="Action chunk length")
    p.add_argument("--n-action-steps", type=int, default=None)
    p.add_argument("--val-episodes", type=int, default=3)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--weight-decay", type=float, default=1e-10)
    p.add_argument("--grad-clip", type=float, default=10.0)
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
    train_data = SmolVLADataset(bundle, train_eps, horizon=args.horizon, task_prefix=args.task_prefix)
    val_data = SmolVLADataset(bundle, val_eps, horizon=args.horizon, random_goal=False, task_prefix=args.task_prefix)
    loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=True, drop_last=True)

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
        "observation.images.goal": PolicyFeature(type=FeatureType.VISUAL, shape=image_shape),
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(bundle.state_dim,)),
    }
    policy.config.output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(bundle.action_dim,)),
    }
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    policy.to(device)
    tokenizer = policy.model.vlm_with_expert.processor.tokenizer

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
    step = 0
    policy.train()
    while step < args.steps:
        for batch in loader:
            if step >= args.steps:
                break
            texts = batch.pop("task")
            encoded = tokenizer(list(texts), padding=True, truncation=True,
                                max_length=policy.config.tokenizer_max_length, return_tensors="pt")
            batch[OBS_LANGUAGE_TOKENS] = encoded.input_ids
            batch[OBS_LANGUAGE_ATTENTION_MASK] = encoded.attention_mask.bool()
            batch["observation.state"] = (batch["observation.state"].to(device) - state_mean) / state_std
            batch["action"] = (batch["action"].to(device) - action_mean) / action_std
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            loss, _ = policy(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            scale = min(1.0, (step + 1) / max(1, args.warmup_steps))
            for group in optimizer.param_groups:
                group["lr"] = args.lr * scale
            optimizer.step()
            step += 1
            if step % 100 == 0 or step == 1:
                print(f"step={step}/{args.steps} loss={loss.item():.6f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    policy.save_pretrained(args.output_dir)
    (args.output_dir / "smolvla_local_config.json").write_text(json.dumps({
        "bundle": str(bundle.dir), "horizon": args.horizon,
        "n_action_steps": policy.config.n_action_steps,
        "action_names": list(bundle.joint_names),
        "state_mean": state_mean.cpu().tolist(), "state_std": state_std.cpu().tolist(),
        "action_mean": action_mean.cpu().tolist(), "action_std": action_std.cpu().tolist(),
    }, indent=2), encoding="utf-8")
    print(f"saved {args.output_dir}")


if __name__ == "__main__":
    main()
