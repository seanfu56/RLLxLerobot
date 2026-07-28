from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from diffusion.data import center_square_crop, resolve_video_paths
from diffusion.diffusion import GaussianDiffusion
from diffusion.model import VideoUNet, VideoUNetConfig


def small_model(*, conditional: bool = False) -> VideoUNet:
    return VideoUNet(
        VideoUNetConfig(
            channels=3,
            condition_channels=3 if conditional else 0,
            base_channels=8,
            channel_multipliers=(1, 2),
            blocks_per_level=1,
            time_embedding_dim=32,
        )
    )


class DiffusionTests(unittest.TestCase):
    def test_center_square_crop_removes_80_pixels_from_both_edges(self) -> None:
        frame = np.broadcast_to(np.arange(640)[None, :, None], (480, 640, 3)).copy()
        cropped = center_square_crop(frame)
        self.assertEqual(cropped.shape, (480, 480, 3))
        np.testing.assert_array_equal(cropped, frame[:, 80:560])

    def test_bundle_resolves_raw_episode_videos_after_move(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            piper_data = Path(directory) / "piper-data"
            bundle_dir = piper_data / "dataset" / "pick-can-all"
            shard_dir = piper_data / "dataset" / "_shards" / "pick-can-m1"
            video = piper_data / "raw" / "pick-can-m1" / "episode_000" / "video.mp4"
            bundle_dir.mkdir(parents=True)
            shard_dir.mkdir(parents=True)
            video.parent.mkdir(parents=True)
            video.touch()
            (bundle_dir / "bundle.json").write_text(
                json.dumps({"shards": ["_shards/pick-can-m1"]}), encoding="utf-8"
            )
            (shard_dir / "shard.json").write_text(
                json.dumps(
                    {
                        "source": "pick-can-m1",
                        "raw_root": "/a/stale/absolute/path",
                        "episodes": [{"name": "episode_000"}],
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(resolve_video_paths(bundle_dir), [video.resolve()])

    def test_unet_preserves_video_shape(self) -> None:
        model = small_model()
        video = torch.randn(1, 3, 2, 8, 8)
        output = model(video, torch.tensor([3]))
        self.assertEqual(output.shape, video.shape)

    def test_superres_unet_resizes_condition_frame_by_frame(self) -> None:
        model = small_model(conditional=True)
        video = torch.randn(1, 3, 2, 8, 8)
        condition = torch.randn(1, 3, 2, 4, 4)
        output = model(video, torch.tensor([3]), condition)
        self.assertEqual(output.shape, video.shape)

    def test_diffusion_loss_and_ddim_sample_are_finite(self) -> None:
        model = small_model()
        diffusion = GaussianDiffusion(timesteps=8)
        clean = torch.randn(1, 3, 2, 8, 8).clamp(-1, 1)
        loss = diffusion.training_loss(model, clean)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(bool(torch.isfinite(loss)))
        sampled = diffusion.sample(
            model,
            tuple(clean.shape),
            inference_steps=2,
            generator=torch.Generator().manual_seed(1),
        )
        self.assertEqual(sampled.shape, clean.shape)
        self.assertTrue(bool(torch.isfinite(sampled).all()))
        self.assertGreaterEqual(float(sampled.min()), -1)
        self.assertLessEqual(float(sampled.max()), 1)


if __name__ == "__main__":
    unittest.main()
