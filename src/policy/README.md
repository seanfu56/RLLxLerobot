# policy

A small CNN action-chunk policy: ImageNet ResNet-18 trained end to end, a
spatial softmax bottleneck, and a conditional 1D U-Net over the action chunk.
This is the canonical Diffusion Policy recipe (Chi et al., RSS 2023) sized for a
bundle of ~20 episodes rather than for a large multi-task dataset.

That network is trained under one of two generative objectives, picked with
`--objective`: **diffusion** (DDPM/DDIM, the default) or **flow** (rectified
flow). See [Objectives](#objectives). `--goal-conditioned` additionally feeds it
a picture of how the episode should end; see
[Goal conditioning](#goal-conditioning).

## Why this architecture

A DINOv2 transformer policy used to live here too. Its sweeps on `pick-bar`
(20 episodes, 3601 frames) landed in a very narrow band:

```
$ python sweeps/collect_results.py --root models/sweeps --prefix C_
C_gcaux_flow_c8_ssm_d44    ratio 0.370
...
C_bc_flow_c8_ssm_d44       ratio 0.441
```

Sixteen runs across three architecture axes, all within 0.07 of each other. The
trunk was not the limiting factor, so that family was removed in favour of one
that changes what actually differs from the published recipe:

| | removed transformer policy | here |
|---|---|---|
| encoder | DINOv2 ViT-L/14, **frozen**, 304M | ResNet-18, **trained**, 11.2M |
| decoder | transformer, 8.6M | 1D conv U-Net, 17.3M |
| observation | 1 frame | 2 frames (encodes velocity) |
| chunk | predict 8 | predict 16, execute 8 |
| random crop | off (`--crop-scale 1.0`) | on (`--crop-scale 0.9`) |
| total | 320.7M | 28.5M |

A frozen encoder cannot adapt its notion of "nearby" to the task. That matters
more than decoder capacity here: at this data scale a diffusion policy behaves
largely as a lookup table keyed by the nearest training image in latent space
([arXiv:2505.05787](https://arxiv.org/abs/2505.05787)), so the embedding's
neighbourhood structure is what determines whether it retrieves the right chunk.

The `models/sweeps/` checkpoints from that family can no longer be loaded -
nothing in the tree can build the architecture. Their `run.json` and `log.jsonl`
still read fine, so `sweeps/collect_results.py` continues to report their
metrics, and the checkpoint picker refuses them with an explanation.

## Prepare the data

Raw teleop recordings live in `piper-data/raw/<task>/episode_XXX/`. Decode them
once into `piper-data/dataset/` — training reads that, never the MP4s:

```bash
python src/policy/data_prep.py --list          # what is on disk
python src/policy/data_prep.py                 # one bundle per task directory
python src/policy/data_prep.py --bundles pick-can-m1
```

Each task directory becomes a *shard* (the decoded pixels, under `_shards/`) and
a *bundle* (a manifest pointing at shards). A new recording session needs no
code change; only a bundle that **merges** sessions does, via `BUNDLE_SPECS` in
`data_prep.py`:

```python
BUNDLE_SPECS = {"pick-can-all": ("pick-can-m1", "pick-can-m2")}
```

Merged bundles are a few kilobytes — they reference the shards rather than
copying them, and re-running the script reuses any shard already decoded at the
same `--image-size`. Pass `--force` to re-decode.

## Train

```bash
python src/policy/train.py --bundle pick-can-m1 --run-name S_pick_can
```

Defaults are the published recipe (horizon 16, 2 observation frames, execute 8,
100 training timesteps, 10 DDIM steps at inference, EMA, random crop) with one
deliberate change: `--down-dims 128 256 512` instead of `512 1024 2048`. The
reference width is a 252M-parameter U-Net, which is not a sensible size for 3.6k
frames.

| `--down-dims` | U-Net |
|---|---|
| `64 128 256` | 4.9M |
| `128 256 512` (default) | 17.3M |
| `512 1024 2048` (reference) | 251.8M |

Two knobs are worth sweeping before anything else:

```bash
# capacity at this data scale
python src/policy/train.py --bundle pick-bar --run-name S_small --down-dims 64 128 256

# absolute joint targets vs. targets relative to the measured state
python src/policy/train.py --bundle pick-bar --run-name S_delta \
    --action-repr delta --delta-mode incremental
```

`--horizon` must be a multiple of `2 ** len(down_dims)`; the error message says
what to use if it is not.

## Objectives

`--objective` chooses how the network is trained and sampled. Everything else -
the encoder, the U-Net, the conditioning, the action representation, the
normalisation, guidance - is identical, so a run of each is a controlled
comparison of the generative process alone.

| | `diffusion` (default) | `flow` |
|---|---|---|
| forward process | add noise on a cosine schedule | interpolate `(1-t)·noise + t·action` |
| decoder predicts | the noise | the velocity `action − noise` |
| training time | one of `--num-train-timesteps` buckets | continuous on [0, 1] |
| sampler | DDIM | Euler |
| `--num-inference-steps` | rounded by the stride: 3 runs 4 | exact: 3 runs 3 |
| ignores | — | `--num-train-timesteps`, `--beta-schedule` |

```bash
python src/policy/train.py --bundle pick-bar --run-name S_flow --objective flow
```

The reason to try flow matching here is latency, not accuracy. The path from
noise to data is straight by construction, so it degrades more gracefully as the
step count drops — and the step count is the binding constraint on this arm: a
re-plan at the default 10 steps already costs more than one 20 Hz control step.
Sampler cost is very nearly linear in the step count for both objectives - one
decoder call each - so cutting steps is the only lever that matters. The same
flow checkpoint, one session, one GPU:

```
10 Euler steps   p50 126 ms     2 Euler steps   p50 28 ms
```

Read that as a ratio, not an absolute: the numbers in [Cost](#cost) below were
measured on an idle GPU and are roughly half these. Whether 2 steps is
*good enough* is an empirical question about your data.
Compare the two objectives at the same step count before assuming either way;
`sweeps/sweep_policy.sh` has lines for exactly that.

Flow matching adds one knob, `--flow-time-sampling`. `uniform` is the plain
rectified-flow recipe and the default; `logitnormal` (Stable Diffusion 3's
proposal) concentrates training on the middle of the path, where the velocity is
genuinely ambiguous, instead of the two ends, where it is nearly determined by
the endpoint. Passing a diffusion-only flag to a flow run — or the reverse —
prints a warning rather than silently doing nothing.

The default stays diffusion. It is the published recipe, every checkpoint in
`models/simple` was trained under it, and flow matching on 20 episodes is not a
combination with much published evidence behind it.

## Goal conditioning

`--goal-conditioned` adds one more image to what the policy sees: a picture of
how the episode should end. It is encoded by the same ResNet and appended to the
conditioning vector — the U-Net, the objective, the action representation and
the sampler are untouched. The implementation is in
[`goal.py`](goal.py), separately from the plain policy.

```bash
python src/policy/train.py --bundle pick-can-all --run-name G_pick_can \
    --goal-conditioned --goal-dropout 0.1
```

**Which frame is the goal.** `--goal-selection` supports three rules. `uniform4`
takes the third of four frames spread across the episode, matching the generated
video pipeline. `tail` draws from the final `--goal-window` frames.
`future_uniform` draws uniformly from `current + horizon` through the true final
frame, and removes late samples that cannot put their whole target action chunk
before the goal. Nothing needs labelling — every rule reads hindsight goals from
the demonstrations. Validation makes the selected goal deterministic so a moving
goal does not add noise to checkpoint comparison.

Under augmentation the goal shares the observation stack's crop box, for the
same reason the two observation frames do: the camera is static, so cropping the
goal separately would shift the target relative to what the policy is looking at.

**Whether it will help you.** Often not. Every episode of `pick-can-m1` ends in
nearly the same picture, so the goal carries almost nothing the policy could not
infer from the task itself, and it will learn to ignore it — at the cost of one
extra ResNet pass per prediction. Goal conditioning starts paying when one
bundle holds several end states: two objects, two placements, `pick-can-all`
rather than one session. Train with `--goal-dropout 0.1` and measure the
difference rather than assuming it:

```bash
python src/policy/infer.py --checkpoint models/simple/G_pick_can/best.pt \
    --bundle pick-can-all --episode 0            # goal = the episode's last frame
python src/policy/infer.py --checkpoint models/simple/G_pick_can/best.pt \
    --bundle pick-can-all --episode 0 --no-goal  # same weights, null embedding
```

If the two ratios match, the goal is decoration. `--goal-dropout` is what makes
that comparison possible at all: it trains a null-goal embedding, so the same
checkpoint can run with no goal — and that same embedding is what goal-only
classifier-free guidance extrapolates away from (`--guidance-mode goal`, below).
It is separate from `--cond-dropout`, which drops the *whole* conditioning
vector, goal included, and is what the default `full` guidance needs.

Both of those numbers are optimistic in the same way: validation and replay hand
the policy a goal taken from the very episode it is being scored on, which is
the cleanest goal it will ever get. On the arm you supply a photo from a
different attempt, lit slightly differently, with the object somewhere slightly
else. Treat the offline gap as an upper bound on what the goal buys you.

**On the arm.** A goal-conditioned checkpoint needs a goal frame at every step,
so the Streamlit page grows a **Goal image** field next to the checkpoint
picker — point it at a photo of the finished task taken from the same camera; a
frame grabbed from a successful rollout is exactly right. The goal is fixed for
the rollout and survives `reset`, since it describes the task rather than the
attempt. Loading a goal-conditioned checkpoint through the plain `PolicyRunner`
fails with an explanation rather than a state-dict error; `policy.goal.load_runner`
picks the right class from the config.

### Reading the validation output

```
eval  4000 | val_loss 0.0121 | MAE 2.104 (hold 7.518, ratio 0.28) | best-of-8 1.702 (ratio 0.23)
```

`ratio` is the error relative to a policy that just holds the current pose, so
below 1 beats standing still. `best-of-8` is the closest of eight sampled chunks:
a diffusion policy may legitimately commit to one of several valid behaviours,
and a plain MAE scores a correct-but-different choice exactly as harshly as a
wrong one. A large gap between the two means the policy is multimodal rather
than inaccurate.

Neither number is a success rate. Use them to screen checkpoints, not to decide
that a policy works.

## Classifier-free guidance

Off by default. Guidance needs an unconditional branch, which only exists if the
model was trained with conditioning dropout — so it is a training decision, not
just an inference flag:

```bash
python src/policy/train.py --bundle pick-bar --run-name S_cfg --cond-dropout 0.1
```

That replaces the whole observation (vision *and* state together) with a learned
null embedding on 10% of samples. At sampling time the decoder's output becomes
`out_uncond + w · (out_cond − out_uncond)`. That extrapolation is linear in the
output, so it is the same rule under either objective — guiding a velocity field
works exactly like guiding an epsilon prediction, and `--cond-dropout` combines
with `--objective flow` unchanged.

A checkpoint trained without `--cond-dropout` **refuses** any weight other than
1.0, at load time. This is deliberate: guiding against an untrained null
embedding does not raise, it produces confident wrong actions, which on a robot
is the worst possible failure mode.

### What the guided branch drops: `--guidance-mode`

There are two useful "unconditional" branches on a goal-conditioned policy, and
the mode picks which one the weight extrapolates away from:

| mode | the guided branch is | needs | the weight scales |
|---|---|---|---|
| `full` (default) | `null_cond` — no observation, no goal | `--cond-dropout` | everything the policy conditions on |
| `goal` | this frame and this joint state, `null_goal` in place of the goal | `--goal-dropout` | the goal alone |

`goal` is the mode a goal-conditioned rollout usually wants. Under `full`, a
weight of 2.0 also sharpens how literally the policy reads the frame in front of
it, which is not the question being asked; under `goal` the difference between
the branches is exactly *"with this goal"* minus *"from this frame, no goal in
particular"*, so the weight is a dial on how hard the arm is pushed towards the
picture it was aimed at, with the scene it has to act in left alone.

The requirement is only ever *was this null embedding trained* — which means
**`--goal-dropout 0.1` already buys goal-only guidance**, and any existing
goal-conditioned run with dropout can be guided this way without retraining:

```bash
python src/policy/infer.py --checkpoint models/simple/G_pick_can/best.pt \
    --bundle pick-can-all --episode 0 \
    --guidance-mode goal --guidance-weight 1.0 1.5 2.0 3.0
```

Goal-only guidance needs an actual goal: with none, both branches are the null
goal, the extrapolation cancels, and the weight would quietly do nothing — so
the runner refuses instead. `--guidance-mode` is a sampling-time choice, and
`train.py` takes the same flag only to record the default a checkpoint starts
from.

Pick the weight offline, never on the arm:

```bash
python src/policy/infer.py --checkpoint models/simple/S_cfg/best.pt \
    --bundle pick-bar --episode 0 --guidance-weight 1.0 1.5 2.0 3.0
```

That replays the episode once per weight through the real control path and ranks
them. Note that the reference Diffusion Policy does **not** use guidance, and
with 20 episodes the unconditional branch is learned from very little data —
treat any gain as something to verify on hardware, not as a given. Robotics
weights are modest; 1.5–3.0, not the 7.5 common in text-to-image. High weights
also interact badly with `clip_sample`, which clamps each sampler step to
[-1, 1] and will saturate on aggressive extrapolation. (Under flow matching the
clamp applies to the endpoint each Euler step extrapolates to, which is the same
guarantee: a guided velocity cannot walk the chunk out of the action range one
step at a time.)

### Cost

Guidance doubles the U-Net work but not the vision encoder: the conditional
context is encoded once and the other branch is built out of it — a learned
constant under `full`, the same vector with its goal slot swapped under `goal` —
so both go through the U-Net as one batch of 2 and the ResNet still runs once. Measured at the default config
(224px, 2 observation steps, `down_dims=(128,256,512)`, 10 sampler steps):

| | RTX 3090 p50 | RTX 3080 p50 |
|---|---|---|
| guidance 1.0 | 58.2 ms | 58.7 ms |
| guidance 2.0 | 59.5 ms | 59.4 ms |

Effectively free on GPU — a batch of 2 fits inside the parallel slack. On CPU it
is not free (34 ms → 91 ms), so offline replay runs noticeably slower.

Independently of guidance, **a re-planning step costs ~58 ms against a 50 ms
budget at 20 Hz.** With `n_action_steps=8` that is one long step in eight, and
the UI's telemetry panel flags it. `--num-inference-steps 5` halves it to ~28 ms
if you want the loop to stay inside budget.

## Inference

Offline first — this never touches the robot:

```bash
# what a control step will cost
python src/policy/infer.py --checkpoint models/simple/S_pick_bar/best.pt

# drive the real control path (history, action queue, re-planning) through a
# recorded episode and compare against what the operator did
python src/policy/infer.py --checkpoint models/simple/S_pick_bar/best.pt \
    --bundle pick-bar --episode 0 --json /tmp/trace.json
```

On the arm, use the Streamlit page, which lists both policy families:

```bash
bash scripts/9_policy_ui.sh
```

The checkpoint picker defaults to scanning `models/sweeps models/simple`. Pick a
run, load it, and start a rollout; the page reads `n_action_steps` from the
checkpoint so the execution horizon defaults to what the policy was trained for.

For a guidance-capable checkpoint a **Guidance** section appears in the sidebar
once the policy is loaded, offering only the modes that checkpoint trained a
null embedding for. Moving the slider and pressing Apply retunes the weight in
place — no reload, no CUDA context rebuild, no warm-up — so trials at different
weights are a few seconds apart rather than a few minutes; switching mode works
the same way. Either drops any queued chunk, so actions planned under the old
setting never reach the arm, and both are refused outright while a rollout is
running. Each recorded rollout's `meta.json` stores the weight *and* the mode
that produced it. A checkpoint with neither `--cond-dropout` nor
`--goal-dropout` shows no section at all.

The goal-conditioned page (`scripts/9_goal_policy_ui.sh`) is where goal-only
guidance belongs: generate the video, aim at a frame, then turn the weight up to
push the rollout harder towards it.

**Run the control loop at the rate the demonstrations were recorded at.** The
policy conditions on the last two frames, and the motion between them is the
velocity signal it learned from. At a different control rate that signal means
something else.

## Layout

```
data_prep.py     raw recordings -> frame/state/action bundles (run once)
config.py        PolicyConfig, its validation, and target_stats
model.py         RgbEncoder, ConditionalUnet1d, ChunkPolicy
datasets.py      Bundle, ChunkDataset (observation history), compute_stats
backbones.py     ImageNormalizer, SpatialSoftmax, BatchNorm -> GroupNorm
modules.py       Normalizer, the timestep embedding
objectives.py    DiffusionObjective (DDPM/DDIM), FlowMatchingObjective (rectified flow)
goal.py          goal-conditioned dataset, policy and runner
train.py         training loop, EMA, validation                [CLI]
infer.py         offline replay and latency measurement        [CLI]
inference.py     PolicyRunner, the on-robot interface
checkpoints.py   run/checkpoint discovery, no torch required
tests/
```

Both CLIs run directly - `python src/policy/train.py …` - and put `src` on the
path themselves, so no `PYTHONPATH` is needed.

## Tests

```bash
PYTHONPATH=src python -m unittest discover -s src/policy/tests -t src
```
