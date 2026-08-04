# SmolVLA fine-tuning

This adapter trains the official LeRobot `SmolVLAPolicy` on the local
`piper-data/dataset` bundles. Each sample contains exactly two visual inputs:

* `observation.images.current`: the frame at the action start;
* `observation.images.goal`: a frame sampled uniformly from the strict future
  interval `(current, episode_end]` (the final transition uses its own frame).

The pretrained VLM/vision encoder is frozen and the action expert is trained
with SmolVLA's rectified-flow matching loss. `--horizon` changes both the local
action chunk and the expert's `chunk_size` (up to the pretrained model limit of
50). The state and action statistics are calculated from training episodes only.

Install dependencies from the LeRobot checkout:

```bash
pip install -e ".[smolvla]"  # run from the LeRobot checkout
```

Run from this repository root:

```bash
python src/smolvla/train.py --bundle pick-can-all \
  --pretrained lerobot/smolvla_base --horizon 16 --steps 20000
```
