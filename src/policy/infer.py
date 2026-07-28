#!/usr/bin/env python3
"""Run a trained checkpoint offline, before it ever reaches the arm.

Two modes, neither of which touches hardware:

  Latency only - what a control step will cost.

      python src/policy/infer.py --checkpoint models/simple/S_pick_bar/best.pt

  Replay - drive the runner frame by frame through a held-out episode of a
  prepared bundle and compare its commands with what the operator did. This is
  the closed-loop control path (observation history, action queue, re-planning
  every --action-steps) run on recorded pixels, so it catches normalisation and
  history bugs that a training-time validation pass cannot.

      python src/policy/infer.py --checkpoint models/simple/S_pick_bar/best.pt \
          --bundle pick-bar --episode 0

The commanded joint targets are compared against the recording and against a
policy that just holds the current pose. A ratio below 1 beats standing still.

A goal-conditioned checkpoint (``--goal-conditioned``, see ``policy/goal.py``)
replays against the last frame of the episode unless ``--goal-image`` names
another one. If it was trained with ``--goal-dropout``, ``--no-goal`` scores the
same checkpoint with no goal at all, and the gap between the two is what the
goal is actually worth.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from policy.datasets import Bundle  # noqa: E402
from policy.goal import load_runner  # noqa: E402
from policy.inference import PolicyRunner  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", required=True, help="A trained .pt, usually best.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-ema", dest="use_ema", action="store_false")
    parser.add_argument("--action-steps", type=int, default=None,
                        help="Actions executed per prediction. Default: the checkpoint's n_action_steps")
    parser.add_argument("--num-inference-steps", type=int, default=None,
                        help="Sampler steps. Fewer means lower latency and a coarser action")
    parser.add_argument("--guidance-weight", type=float, nargs="+", default=None, metavar="W",
                        help="Classifier-free guidance weight(s). 1.0 is plain conditional sampling. "
                             "Several values replay the episode once per weight and rank them, which "
                             "is how to pick one without touching the arm")

    goal = parser.add_argument_group("goal conditioning")
    goal.add_argument("--goal-image", type=Path, default=None, metavar="PATH",
                      help="Goal frame for a --goal-conditioned checkpoint. Default in replay: "
                           "the last frame of the replayed episode, which is what validation used")
    goal.add_argument("--no-goal", action="store_true",
                      help="Run a goal-conditioned checkpoint with no goal, against its learned "
                           "null embedding. Needs --goal-dropout above 0 at training time. Compare "
                           "with the same run's goal score to see what the goal is worth")

    replay = parser.add_argument_group("replay")
    replay.add_argument("--bundle", default=None, help="Bundle name under --data-root, or a path")
    replay.add_argument("--data-root", type=Path, default=Path("piper-data/dataset"))
    replay.add_argument("--episode", type=int, default=0, help="Episode index within the bundle")
    replay.add_argument("--max-steps", type=int, default=0, help="Stop early (0 = the whole episode)")
    replay.add_argument("--json", type=Path, default=None, help="Write the per-step trace here")

    parser.add_argument("--latency-steps", type=int, default=20,
                        help="Predictions timed in latency mode")
    return parser.parse_args(argv)


def read_goal_image(path: Path) -> np.ndarray:
    """A goal frame off disk, in the raw BGR the runner's own crop expects."""
    import cv2

    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise SystemExit(f"Could not read --goal-image {path}")
    return frame


def episode_goal(bundle: Bundle, episode_index: int) -> np.ndarray:
    """The last frame of an episode, converted back to camera BGR.

    Bundle frames are already RGB at the model's resolution; the runner expects
    what a camera hands it, so the channels are swapped back here and its centre
    crop is a no-op on an already square frame. This is the goal validation used
    - ``GoalChunkDataset(random_goal=False)`` pins the same frame.
    """
    episode = bundle.episodes[episode_index]
    frame_rgb = np.array(bundle.frames[episode.shard][episode.end - 1])
    return frame_rgb[:, :, ::-1].copy()


def resolve_bundle_dir(name: str, data_root: Path) -> Path:
    candidate = Path(name)
    if (candidate / "bundle.json").is_file():
        return candidate
    resolved = data_root / name
    if (resolved / "bundle.json").is_file():
        return resolved
    raise SystemExit(f"No bundle at {candidate} or {resolved}")


def measure_latency(runner: PolicyRunner, steps: int) -> dict[str, float]:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    state = np.zeros(runner.config.state_dim, dtype=np.float32)
    runner.warmup()
    timings: list[float] = []
    for _ in range(steps):
        runner.reset()
        started = time.perf_counter()
        runner.predict_chunk(frame, state)
        timings.append((time.perf_counter() - started) * 1000.0)
    runner.reset()
    ordered = sorted(timings)
    return {
        "mean_ms": statistics.fmean(timings),
        "p50_ms": ordered[len(ordered) // 2],
        "p95_ms": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
        "max_ms": ordered[-1],
    }


def replay_episode(runner: PolicyRunner, bundle: Bundle, episode_index: int,
                   max_steps: int) -> tuple[dict[str, float], list[dict]]:
    """Step the runner through recorded frames and score its commands.

    Frames in a bundle are already RGB at the model's resolution, while the
    runner expects raw BGR from the camera and does its own centre crop. The
    channel swap below undoes the first conversion; the crop is a no-op on an
    already square frame.
    """
    if not 0 <= episode_index < len(bundle.episodes):
        raise SystemExit(f"--episode {episode_index} is outside [0, {len(bundle.episodes)})")
    episode = bundle.episodes[episode_index]
    frames = bundle.frames[episode.shard]
    states = bundle.states[episode.shard]
    actions = bundle.actions[episode.shard]

    length = len(episode) if max_steps <= 0 else min(len(episode), max_steps)
    runner.reset()

    trace: list[dict] = []
    policy_error: list[np.ndarray] = []
    hold_error: list[np.ndarray] = []
    latencies: list[float] = []

    for offset in range(length):
        absolute = episode.start + offset
        frame_rgb = np.array(frames[absolute])
        frame_bgr = frame_rgb[:, :, ::-1].copy()
        state = states[absolute].astype(np.float32)
        target = actions[absolute].astype(np.float32)

        queued = runner.queued_actions
        started = time.perf_counter()
        command = runner.select_action(frame_bgr, state)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if queued == 0:
            latencies.append(elapsed_ms)

        policy_error.append(np.abs(command - target))
        hold_error.append(np.abs(state - target))
        trace.append(
            {
                "step": offset,
                "replanned": queued == 0,
                "latency_ms": round(elapsed_ms, 3),
                "state": [round(float(value), 4) for value in state],
                "command": [round(float(value), 4) for value in command],
                "recorded": [round(float(value), 4) for value in target],
            }
        )

    mean_policy = np.stack(policy_error).mean(axis=0)
    mean_hold = np.stack(hold_error).mean(axis=0)
    summary = {
        "steps": length,
        "predictions": len(latencies),
        "mae": float(mean_policy.mean()),
        "mae_hold": float(mean_hold.mean()),
        "mae_ratio": float(mean_policy.mean() / max(float(mean_hold.mean()), 1e-8)),
        "prediction_mean_ms": statistics.fmean(latencies) if latencies else 0.0,
        "per_joint_mae": {
            name: float(value) for name, value in zip(bundle.joint_names, mean_policy, strict=True)
        },
    }
    return summary, trace


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    weights = args.guidance_weight
    runner = load_runner(
        args.checkpoint,
        device=args.device,
        use_ema=args.use_ema,
        action_steps=args.action_steps,
        num_inference_steps=args.num_inference_steps,
        guidance_weight=weights[0] if weights else None,
    )
    goal_conditioned = runner.config.goal_conditioned
    if args.goal_image is not None and not goal_conditioned:
        raise SystemExit(f"{runner.run_name} was not trained with --goal-conditioned.")
    if args.no_goal and not goal_conditioned:
        raise SystemExit(f"{runner.run_name} has no goal to leave out.")
    if args.no_goal and runner.config.goal_dropout <= 0.0:
        raise SystemExit(
            f"{runner.run_name} was trained with --goal-dropout 0, so its null-goal embedding "
            "is untrained and --no-goal would condition on noise. Retrain with "
            "--goal-dropout 0.1 to make this comparison."
        )
    if args.goal_image is not None and args.no_goal:
        raise SystemExit("--goal-image and --no-goal ask for opposite things.")

    description = runner.describe()
    print(f"checkpoint   {description['checkpoint']} (step {description['step']})")
    print(f"policy       simple/{description['objective']} · predict {description['chunk_size']} · "
          f"observe {description['n_obs_steps']} · execute {description['action_steps']} · "
          f"{description['num_inference_steps']} sampler steps")
    print(f"unet         {tuple(description['down_dims'])} | device {description['device']} | "
          f"ema {description['use_ema']}")
    print(f"guidance     weight {description['guidance_weight']} "
          f"(trained with cond dropout {description['cond_dropout']})"
          + ("" if runner.supports_guidance else " - guidance unavailable"))
    print(f"action       {description['action_repr']}"
          + (f"/{description['delta_mode']}" if description["action_repr"] == "delta" else ""))

    if goal_conditioned:
        if args.no_goal:
            source = "none (learned null embedding)"
        elif args.goal_image is not None:
            runner.set_goal(read_goal_image(args.goal_image))
            source = str(args.goal_image)
        elif args.bundle is None:
            # Latency mode feeds the model a blank observation, so a blank goal
            # keeps the pair consistent. It costs the same either way.
            runner.set_goal(np.zeros((480, 640, 3), dtype=np.uint8))
            source = "blank (latency mode)"
        else:
            source = "the replayed episode's last frame"  # set once the bundle is open
        print(f"goal         {source} | trained on the last "
              f"{description['goal_window']} frames, dropout {description['goal_dropout']}")

    if args.bundle is None:
        timing = measure_latency(runner, args.latency_steps)
        print(f"\nprediction   mean {timing['mean_ms']:.1f} ms · p50 {timing['p50_ms']:.1f} · "
              f"p95 {timing['p95_ms']:.1f} · max {timing['max_ms']:.1f}")
        budget_ms = 1000.0 / runner.fps
        share = timing["p95_ms"] / budget_ms
        print(f"control      {runner.fps:.0f} Hz leaves {budget_ms:.0f} ms per step; "
              f"a prediction uses {share * 100:.0f}% of one")
        if share > 1.0:
            print("             The step it re-plans on will run long. Execute more actions per "
                  "prediction or lower the sampler steps.")
        return

    bundle = Bundle(resolve_bundle_dir(args.bundle, args.data_root))
    if not 0 <= args.episode < len(bundle.episodes):
        raise SystemExit(f"--episode {args.episode} is outside [0, {len(bundle.episodes)})")
    if goal_conditioned and args.goal_image is None and not args.no_goal:
        runner.set_goal(episode_goal(bundle, args.episode))

    runner.warmup()
    episode = bundle.episodes[args.episode]

    results = []
    for weight in weights or [runner.guidance_weight]:
        runner.set_guidance_weight(weight)
        summary, trace = replay_episode(runner, bundle, args.episode, args.max_steps)
        results.append((weight, summary, trace))

        print(f"\nreplay       {episode.name}@{episode.source} · {summary['steps']} steps, "
              f"{summary['predictions']} predictions · guidance {weight}")
        print(f"error        MAE {summary['mae']:.3f} vs hold {summary['mae_hold']:.3f} "
              f"(ratio {summary['mae_ratio']:.2f})")
        print("             " + "  ".join(
            f"{name.replace('.pos', '')}:{value:.3f}" for name, value in summary["per_joint_mae"].items()
        ))
        print(f"prediction   mean {summary['prediction_mean_ms']:.1f} ms")

    if len(results) > 1:
        print("\nguidance sweep, best first:")
        for weight, summary, _ in sorted(results, key=lambda entry: entry[1]["mae_ratio"]):
            print(f"  w={weight:<5.2f} ratio {summary['mae_ratio']:.3f}  "
                  f"MAE {summary['mae']:.3f}  {summary['prediction_mean_ms']:.1f} ms")
        print("One episode is a weak basis for choosing. Repeat over the held-out episodes "
              "before committing a weight to hardware.")

    best = min(results, key=lambda entry: entry[1]["mae_ratio"])[1]
    if best["mae_ratio"] >= 1.0:
        print("\nThis checkpoint is no better than holding the current pose on this episode. "
              "Do not run it on the arm.")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                [
                    {"guidance_weight": weight, "summary": summary, "trace": trace}
                    for weight, summary, trace in results
                ],
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"trace        {args.json}")


if __name__ == "__main__":
    main()
