# Cosmos3-Nano post-training on the Piper demonstrations

LoRA supervised fine-tuning (SFT) of NVIDIA **Cosmos3-Nano** at **256×256** on
`piper-data/dataset/pick-can-all`, plus three ways to run the result.

Cosmos3-Nano is an 8B joint text+vision transformer. Only the generation
pathway's attention projections (`add_q_proj`, `add_k_proj`, `add_v_proj`,
`to_add_out`) are adapted — 30.7M trainable parameters — while the
understanding (text) pathway and the Wan VAE stay frozen. This matches the
official Cosmos3 LoRA SFT recipe.

```text
data.py       bundle -> raw MP4 -> 256x256 clips + captions
train.py      LoRA SFT
runtime.py    CosmosRunner: load once, generate from a camera frame
sample.py     offline sampling against held-out episodes
serve.py      HTTP server (cosmos3 env)
client.py     HTTP client (stdlib + numpy + cv2, runs in `piper`)
```

## Environment

**Cosmos3 must not be installed into `piper`.** `lerobot` 0.4.4 pins
`diffusers<0.36.0` and `huggingface-hub<0.36.0`; Cosmos3 needs diffusers from
git and `transformers>=5`, which drags `huggingface-hub>=1.5`. Upgrading in
place would leave `lerobot` — recording, teleop, policy training — installed
against versions it declares itself incompatible with.

What `piper` (Python 3.10.20, torch 2.10.0+cu128) is missing:

| Package | In `piper` | Needed |
| --- | --- | --- |
| `transformers` | **absent** | `>=5.10` — Qwen3-VL classes, v5 chat-template API |
| `diffusers` | 0.35.2 | **git `main`** — `Cosmos3OmniPipeline` is in no tagged release |
| `peft` | **absent** | `>=0.19` |
| `huggingface_hub` | 0.35.3 | `>=1.5,<2` (transitive from transformers 5) |
| `tensorboard` | **absent** | default `--report-to`; `--report-to wandb` avoids it |

Create a separate environment from `requirements.txt`:

```bash
conda create -n cosmos3 python=3.12 -y
conda activate cosmos3
pip install torch==2.12.1 torchvision==0.27.1 \
    --index-url https://download.pytorch.org/whl/cu130
pip install -r src/cosmos/requirements.txt
```

torch goes in first, from the CUDA index — the RTX PRO 6000 is Blackwell
(sm_120) and needs a cu128 or newer build. `requirements.txt` pins the exact
versions this code was verified against, including the **diffusers commit**:
the transformer's forward return type changed from a 2-tuple to
`Cosmos3OmniTransformerOutput` between 0.39 and 0.40, so tracking `main`
unpinned can break `train.py` without warning.

`cosmos_guardrail` is **not** needed — every script here passes
`enable_safety_checker=False`, and diffusers ships an import stub for that case.

The `nvidia/Cosmos3-Nano` weights are already in the local HF cache (~33 GB).

### What still runs under `piper`

Two things, deliberately:

- `data.py` needs only torch, numpy, OpenCV and Pillow, so the dataset,
  captions and clip geometry can be checked without the model stack:
  `python src/cosmos/train.py --dry-run`
- `client.py` needs only the standard library, numpy and cv2, so robot code in
  `piper` can call the model without ever importing diffusers.

## Data

`bundle.json` points at 224×224 policy arrays, but training goes back to each
episode's original 480×640 MP4 under `piper-data/raw/` — upsampling 224→256
would throw away detail the source has. Every frame is center-cropped square
before the resize, the same crop `src/diffusion` and `src/policy` use:

```text
640 pixels wide = 80 removed + 480 retained + 80 removed
480 pixels high =                   480 retained
```

`pick-can-all` is 60 episodes over two shards (`pick-can-m1`, `pick-can-m2`),
114–169 frames each at 20 FPS. Six episodes are held out for validation,
stratified as three from each shard. Training samples are source-balanced every
epoch, so neither motion family can dominate through episode count.

Two clip modes:

- `episode` (training default) — the whole episode uniformly resampled to 93
  frames. It always uses the true first frame, matching first-frame-conditioned
  inference, and includes the complete demonstrated motion. Playback is
  retimed, so the reported FPS is scaled to preserve the real duration
  (a 167-frame episode becomes 93 frames at 11.1 FPS, still 8.35 s).
- `window` — a contiguous 93-frame window at the native 20 FPS. It remains
  available for ablations, but random later windows can reveal which motion is
  already under way and therefore do not match true-first-frame inference.

`num_frames` must satisfy `(num_frames - 1) % 4 == 0` (Wan VAE temporal
compression) and the resolution must be a multiple of 16. At 256×256×93 the
joint sequence carries 1,536 vision tokens.

Text conditioning comes from `captions.json`, keyed by the episode `task`
string. Captions describe what the video shows, not the label, and one caption
covers both recording modes of a bundle (`*-m1` and `*-m2`), since the two modes
differ only in how the demonstration is executed.

Note that the recorded task label in `piper-data/raw/*/episode_*/meta.json` —
and therefore in the generated `shard.json` — is `"pick up cube"` for every
bundle, a stale default from the recording script. `captions.json` is keyed by
the real tasks (`"pick up can"`, `"pick up scissors"`), so the shards must be
relabelled before training, otherwise `load_episodes` looks up `"pick up cube"`
and the run aborts with `No caption for task(s) ['pick up cube']`. Alternatively
override every caption with `--caption "..."`.

## Train

Run from the repository root:

```bash
python src/cosmos/train.py --bundle piper-data/dataset/pick-can-all
```

`Cosmos3OmniTransformer` packs one text+vision sequence per forward call and
has no batch dimension, so a step is one clip and `--grad-accum` (default 8)
provides the effective batch size. Defaults: 1000 optimizer steps, LoRA rank
32, LR 1e-4 with 20 warmup steps decaying to 0.1×, 10% caption dropout for
classifier-free guidance, and bf16. LoRA covers the generation attention,
generation-specific feed-forward layers, and vision input/output projections;
the separate understanding/text pathway remains frozen.

For both GPUs:

```bash
accelerate launch --num_processes 2 src/cosmos/train.py
```

Add `--gradient-checkpointing` if memory is tight, and `--report-to wandb` if
tensorboard is unavailable.

The training step is a single random-timestep rectified-flow step: latent frame
0 is held clean as the image condition and the loss is the MSE between the
predicted and true velocity over the remaining frames. Noise levels are drawn
from the Waver video-time schedule with flow shift 5.0. Joint-sequence packing
(mRoPE ids, sequence indexes, timestep splicing) and VAE latent normalization
are delegated to the pipeline's own helpers rather than reimplemented, so
training stays aligned with what inference does.

Validation loss is measured on the held-out clips at three fixed noise levels
(0.25/0.5/0.75) with one shared noise draw, so the number moves with the model
rather than with the sampler. Step 0 is logged before any update — LoRA is
zero-initialized there, so it is the base model's loss and the reference point
for whether fine-tuning helped.

## Inference

All three paths go through `CosmosRunner` in `runtime.py`, so they load the
model, merge the adapter and preprocess frames identically. Construction is the
expensive part (~15 s and ~35 GB of VRAM), so build one runner and keep it.

Frames follow the repository's camera convention: a numpy argument is **BGR**
uint8, what `cv2.VideoCapture` gives and what
`policy.inference.PolicyRunner.select_action` expects. It is center-cropped
square and resized exactly the way training preprocessed the recordings.

### 1. Offline sampling

Use every captured first frame in `snapshots/pick-can` in timestamp order while
loading the model only once:

```bash
python src/cosmos/sample.py \
  --adapter models/cosmos3_nano_lora/pick-can-all/adapter \
  --images-dir snapshots/pick-can
```

Each 640×480 snapshot is center-cropped before resizing, matching training.
Its filename stem and sampling seed are retained in the generated MP4 name.

Without `--image` or `--images-dir`, the sampler uses the held-out dataset:

```bash
python src/cosmos/sample.py --adapter models/cosmos3_nano_lora/pick-can-all/adapter
```

Reproduces the training split (keep `--val-episodes` / `--split-seed` in sync),
conditions on each held-out episode's first frame, and writes four independent
Flow-Matching trajectories next to the ground-truth clip they should match.
The sampler chooses a fresh random base seed on each invocation; pass
`--seed 1000` for a reproducible seed sequence or
`--samples-per-condition 1` for the previous single-output behavior. Pass
`--adapter ""` for the base model — running both with the same explicit seed is
the comparison worth making.

```text
samples/cosmos3-nano/<shard>_<episode>_lora_seed<N>.mp4   with the adapter
samples/cosmos3-nano/<shard>_<episode>_base_seed<N>.mp4   without it
samples/cosmos3-nano/<shard>_<episode>_reference.mp4      ground truth
```

Why seeds matter: the rectified Flow-Matching model transports an initial
Gaussian latent through a deterministic ODE. Reusing one seed therefore
reuses one trajectory; it cannot expose the conditional distribution's mode
coverage. Each variant now starts from an independent Gaussian latent.

### What decides which mode you get

One caption covers both recording modes, and the two modes' first frames are
indistinguishable (mean absolute difference across modes 0.0103, versus 0.0105
and 0.0101 within them), so nothing in the conditioning says which motion to
produce. The initial latent decides it, and because `generate` seeds the noise
from the seed alone, **the seed is the unit of variation, not the conditioning
frame** — the same seed lands in the same basin for every held-out episode.
Estimating mode coverage therefore needs many seeds, not many episodes.

Measured on `pick-can-all`, 320 clips, 32 seeds x 2 held-out episodes,
classified against the 60 real episodes by a motion signature that scores 100%
leave-one-out:

| `--guidance-scale` | conditioning FPS | mode-1 (top-down) rate |
| ------------------ | ---------------- | ---------------------- |
| 1                  | 20.0             | 0.0%  (0/64)           |
| 1                  | 12.78 (trained)  | 21.9% (14/64)          |
| 2                  | 12.78            | 21.9% (14/64)          |
| 3                  | 12.78            | 23.4% (15/64)          |
| 6                  | 12.78            | 20.3% (13/64)          |

Two things follow. **The conditioning FPS is decisive**: it scales the temporal
component of the vision tokens' mRoPE ids (`fps=20` packs the 24 latent frames
into a span of 27.6 instead of 43.2) and is written into the prompt's duration
template ("4.7 seconds long and is of 20 FPS" instead of "7.3 seconds"). Asking
for a rate the adapter never saw tells the model the clip covers the wrong
duration, and the faster of the two motions wins every time. This is why `--fps`
now defaults to the rate recovered from the run rather than the camera's 20.

**Classifier-free guidance is inert here**, 1 through 6, contrary to the usual
diversity/adherence tradeoff — so the official Cosmos value of 6 remains the
default. `--flow-shift` was also inert across 5 and 10 (six paired settings,
identical rates) and stays at 10. Neither was measured for visual quality, only
for which motion appears.

The remaining gap matters: ~22% against the 50% the training data contains. That
residual skew is a property of the fine-tuned velocity field, not of the
sampler, and closing it needs a training-side change.

### DiverseFlow inference

`sample.py` also implements the training-free coupled inference method from
**DiverseFlow: Sample-Efficient Diverse Mode Coverage in Flows**. At every
Euler step it predicts the clean endpoint of all `K` trajectories, builds a
DPP kernel over their trajectory features, and adds the normalized gradient of
the DPP log-likelihood to the reverse-time flow. The first image-conditioning
latent is masked out of both the feature and the gradient, then restored after
every step.

Run the low-memory latent-feature version on all supplied first frames:

```bash
python src/cosmos/sample.py \
  --adapter models/cosmos3_nano_lora/pick-can-all/adapter \
  --images-dir snapshots/pick-can \
  --sampler diverse-flow \
  --samples-per-condition 4 \
  --seed 1000 \
  --guidance-scale 3 \
  --diversity-feature latent \
  --diversity-scale 1
```

Outputs include the sampler in the name, for example
`snapshot_..._lora_diverse_seed1000.mp4`, so they do not overwrite the UniPC
baseline. Use `--diversity-scale 0` for an IID Euler solver control with the
same seeds.

For the paper-aligned visual objective, use
`--diversity-feature dino`. It decodes several ordered generated frames and
backpropagates through the frozen VAE and DINOv2 model. This is much more
expensive than latent guidance: each guided Euler step needs a differentiable
VAE/DINO pass for every particle. The implementation recomputes those paths
one particle at a time so their activation graphs do not all occupy VRAM at
once. `--diversity-every 2` or `4` trades some guidance fidelity for runtime.

The Gaussian-source quality constraint is enabled at the 99.5th percentile by
default and excludes the clean condition latent. Pass `--quality-percentile 0`
to ablate it. The image paper used a diversity scale of 20; video is much
higher-dimensional, so this implementation starts conservatively at 1 and
expects the scale to be swept against task-success and physical-quality
metrics.

### 2. Direct, in-process

For a script that already lives in the cosmos3 environment:

```python
from cosmos.runtime import CosmosRunner

runner = CosmosRunner(adapter="models/cosmos3_nano_lora/pick-can-all/adapter")
runner.warmup()                              # optional; first call is slower

frames = runner.generate(frame_bgr)          # 93 RGB PIL frames at 256x256
goal = runner.goal_frame(frame_bgr)          # just the last one
```

`generate` accepts a raw BGR camera frame or an RGB `PIL.Image`, and takes
per-call `prompt`, `num_frames`, `fps`, `steps`, `guidance_scale` and `seed`
overrides.

### 3. Server + client

This is the path for a real robot. The model cannot share an environment with
`lerobot`, so it runs behind a socket: the robot process stays in `piper`, and
the 35 GB of weights stay resident across episodes instead of reloading per
rollout.

Start the server in the cosmos3 environment:

```bash
python src/cosmos/serve.py --adapter models/cosmos3_nano_lora/pick-can-all/adapter
```

It binds `127.0.0.1:8501`, warms up once so the first robot request is not the
slow one, and serves:

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | liveness plus the runner's configuration |
| `POST /generate` | `{"image": "<b64 png>", ...}` → `goal` / `frames` / `video` |

There is **no authentication**, so `--host 0.0.0.0` exposes the model to your
whole network; leave it on loopback unless the robot is on another host.

Then from the robot process, in `piper`:

```python
from cosmos.client import CosmosClient

cosmos = CosmosClient()                          # http://127.0.0.1:8501
goal_bgr = cosmos.goal_frame(frame_bgr)          # (256, 256, 3) uint8 BGR
runner.set_goal(goal_bgr)                        # policy/goal.py
```

`goal_frame` is the one to use inside a rollout: a goal-conditioned policy
consumes only the predicted end state, so shipping just that frame instead of
all 93 costs 1/93rd of the bytes. `set_goal` re-crops what it is handed, and
cropping a 256×256 frame to a square is a no-op, so it passes through intact.
`generate` returns the full rollout as BGR arrays and `video` returns MP4 bytes
when you want to look at the whole prediction.

As a CLI:

```bash
python src/cosmos/client.py --health
python src/cosmos/client.py --image frame.png --output video --output-dir samples/live
```

Images cross the wire as base64 **BGR** PNGs — the camera's own layout — so
neither side has a channel swap to get wrong. Generation is serialized behind a
lock: concurrent requests queue rather than fighting over one GPU, and
`/health` stays responsive while they do. A failed request returns HTTP 400/500
with a message and leaves the server up.

## Outputs

```text
models/cosmos3_nano_lora/pick-can-all/
├── adapter/                  final PEFT LoRA adapter
├── checkpoint-000200/        periodic adapters (--save-freq)
├── logs/                     tensorboard / wandb
└── train_args.json
```

`models/` is gitignored.

## Measured on this machine

One RTX PRO 6000 (96 GB), 256×256 × 93 frames, LoRA rank 32:

| | |
| --- | --- |
| Trainable parameters | 30.7M |
| Training | ~1.2 s per clip, ~9 s per optimizer step at `--grad-accum 8` |
| | ~2.5 h for the default 1000 steps on one GPU |
| Peak training memory | ~43 GB (no gradient checkpointing) |
| Model load | 15.2 s from the local HF cache, plus 2.3 s warmup |
| Generation | ~4.8 denoising steps/s; ~7 s for 93 frames at 30 steps |
| Server round trip | 5.2 s for a goal frame at 20 steps, from `piper`, including PNG encode/decode |
