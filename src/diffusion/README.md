# Piper video diffusion

This directory trains a two-stage video generator on the demonstrations in
`piper-data/dataset/pick-can-all`:

1. a first-frame-conditioned video DDPM that generates three future frames at
   **56×56**;
2. a **2D image DDPM** that independently super-resolves each generated frame
   from **56×56 to 224×224**.

There is intentionally no 480×480 generation stage. The final 224×224 output
is the policy input resolution.

## Input preprocessing

The bundle itself points at 224×224 policy arrays, but diffusion training
resolves the `raw_root` stored in each shard and decodes the original MP4 for
every trajectory sample. Each original 480×640 frame is center-cropped first:

```text
640 pixels wide = 80 removed + 480 retained + 80 removed
480 pixels high =                   480 retained
```

That 480×480 crop is then independently resized to 56×56 for base training. For
super-resolution training, the same source frame supplies a paired 56×56
condition and 224×224 target. No random spatial crop is used, so motion cannot
be introduced by inconsistent crops from one frame to the next.

Each episode contributes exactly four frames:

```text
frame indices = [0, round(last/3), round(2*last/3), last]
```

Index `0` is always fixed. During training, `last` is sampled uniformly from
the episode's last 10 frames each time the episode is loaded. The two interior
indices are recalculated from that sampled endpoint. Validation is deterministic
and uses the actual final frame.

The base DDPM receives frame 1 as its condition and diffuses only frames 2–4.
For super-resolution, the three future frames are flattened into three
independent image pairs. The 2D DDPM receives one 56×56 RGB image and predicts
its corresponding 224×224 image; it contains no 3D convolution and cannot mix
information across frames. The first frame is never generated or changed.

## Train

Run these commands from the repository root in the `piper` conda environment.
Train the base model first:

```bash
python src/diffusion/train.py \
  --stage base \
  --bundle piper-data/dataset/pick-can-all
```

Then train the super-resolution model:

```bash
python src/diffusion/train.py \
  --stage superres \
  --bundle piper-data/dataset/pick-can-all
```

The default checkpoints are written to:

```text
models/video_diffusion/pick-can-all/base/{best,latest,final}.pt
models/video_diffusion/pick-can-all/superres/{best,latest,final}.pt
```

Defaults use one four-frame trajectory sample per episode per epoch, 1,000
diffusion time steps, a cosine noise schedule, EMA weights, and Min-SNR
weighting. The base 3D U-Net downsamples only the spatial axes; temporal 3×3×3
convolutions retain cross-frame coherence among its three generated frames. The
super-resolution U-Net uses only 2D convolutions. It is deliberately smaller
and enables gradient checkpointing because 224×224 image activations are
expensive.

The base now defaults to 32 initial channels (about 13M parameters) rather than
64 (about 50M), and uses 500 warmup steps. This is a better fit for 60 episodes
and means the step-1,000 preview has spent half its updates at or near the full
learning rate. Step 1,000 is still an early-training diagnostic, not a
converged sample; use the later previews before judging generation quality.

Resume an interrupted run by preserving the original architecture flags:

```bash
python src/diffusion/train.py --stage base \
  --resume models/video_diffusion/pick-can-all/base/latest.pt
```

Useful first-run diagnostics:

```bash
# Verify data and the full training path cheaply.
python src/diffusion/train.py --stage base --device cpu \
  --steps 2 --batch-size 1 --num-workers 0 --val-videos 1 \
  --base-channels 8 --channel-multipliers 1 2 \
  --output-root /tmp/video-diffusion-smoke --run-name base

# Reduce GPU memory if the super-resolution defaults still do not fit.
python src/diffusion/train.py --stage superres \
  --base-channels 24 --batch-size 1 --grad-accumulation 8 \
  --gradient-checkpointing
```

Validation holds out complete episode videos, so an episode never appears in
both training and validation.

Every evaluation also generates held-out MP4 previews with the EMA weights:

```text
models/video_diffusion/pick-can-all/base/eval/step_0001000/sample_00_56.mp4
models/video_diffusion/pick-can-all/base/eval/step_0001000/sample_00_56_frames.png
models/video_diffusion/pick-can-all/base/eval/step_0001000/reference_00_56.mp4
models/video_diffusion/pick-can-all/base/eval/step_0001000/condition_00_56.png
models/video_diffusion/pick-can-all/superres/eval/step_0001000/sample_00_224.mp4
```

Base evaluation conditions on the held-out episode's fixed first frame and
samples its three future frames. Super-resolution evaluation independently
samples the three 224×224 images from the held-out 56×56 images, then prepends
the unchanged first frame. `samples.json` beside each preview records the four
source indices. The deterministic validation loss is retained for selecting
`best.pt`; evaluation is therefore both visual and quantitative.

Each MP4 is decoded immediately after writing. Training stops with an error if
it does not contain four frames, has the wrong resolution, omits the condition,
or swaps RGB/BGR channels. The condition is also saved separately as PNG, and
the real held-out four-frame sequence is saved as `reference_*.mp4`. Playback
defaults to 1 FPS because these are four sparse trajectory states, making the
fixed first frame visible for a full second.

The four generated frames are also concatenated left-to-right into
`sample_*_frames.png` for viewing the entire trajectory at once.

Control preview generation with:

```bash
python src/diffusion/train.py --stage base \
  --eval-freq 1000 --eval-samples 2 --eval-inference-steps 50
```

## Generate 224×224 MP4s

After both stages finish:

```bash
python src/diffusion/sample.py \
  --base-checkpoint models/video_diffusion/pick-can-all/base/best.pt \
  --superres-checkpoint models/video_diffusion/pick-can-all/superres/best.pt \
  --condition piper-data/raw/pick-can-m1/episode_000/video.mp4 \
  --output samples/pick_can.mp4 \
  --save-low-resolution
```

`--condition` accepts an image, a video, or an episode directory containing
`video.mp4`; for a video it reads frame zero. Sampling uses 50 DDIM steps per
stage and EMA weights. Each of the three 56×56 generated frames is passed
through the 2D super-resolution sampler as a separate image. The output always
has exactly four frames: the unchanged condition followed by three generated
224×224 frames.

Dependencies are PyTorch, NumPy, and OpenCV, all already present in the
project's `piper` environment.
