"""A small CNN action-chunk policy for the Piper teleoperation datasets.

The architecture is the one Chi et al. (RSS 2023) recommend, sized for ~20
episodes: an ImageNet ResNet-18 trained end to end, a spatial-softmax
bottleneck, and a conditional 1D U-Net over the action chunk.

That network is trained under one of two generative objectives, chosen with
``--objective``: diffusion (DDPM training, DDIM sampling - the default) or flow
matching (rectified flow, Euler sampling). Classifier-free guidance is optional
and works with either.

    data_prep.py   raw recordings -> compact frame/state/action bundles (once)
    train.py       bundle -> models/simple/<run>/{run.json,log.jsonl,*.pt}
    infer.py       replay a checkpoint offline, before it reaches the arm
    inference.py   PolicyRunner: the on-robot interface the Streamlit page drives

This package is deliberately decoupled from ``lerobot``: bundles are read
directly, so training never depends on the recording stack.
"""
