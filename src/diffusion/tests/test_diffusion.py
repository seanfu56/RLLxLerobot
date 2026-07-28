from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from diffusion.data import (
    VideoClipDataset,
    VideoInfo,
    center_square_crop,
    four_frame_indices,
    resolve_video_paths,
)
from diffusion.diffusion import GaussianDiffusion
from diffusion.image_model import ImageUNet, ImageUNetConfig
from diffusion.model import VideoUNet, VideoUNetConfig
from diffusion.train import stage_tensors
from diffusion.video_io import write_mp4


def small_model(*, condition_channels: int = 0) -> VideoUNet:
    return VideoUNet(
        VideoUNetConfig(
            channels=3,
            condition_channels=condition_channels,
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

    def test_four_frame_indices_fix_first_and_evenly_space_middle(self) -> None:
        self.assertEqual(four_frame_indices(frame_count=167, last_index=160), (0, 53, 107, 160))
        self.assertEqual(four_frame_indices(frame_count=167, last_index=166), (0, 55, 111, 166))

    def test_training_endpoint_is_always_one_of_last_ten_frames(self) -> None:
        dataset = VideoClipDataset.__new__(VideoClipDataset)
        dataset.videos = [VideoInfo(Path("unused.mp4"), 167, 640, 480, 20.0)]
        dataset.tail_frames = 10
        dataset.random_tail = True
        dataset.seed = 7
        dataset._draws = 0
        for _ in range(50):
            indices = dataset.sample_indices(0)
            self.assertEqual(indices[0], 0)
            self.assertGreaterEqual(indices[-1], 157)
            self.assertLessEqual(indices[-1], 166)
            self.assertEqual(indices, four_frame_indices(167, indices[-1]))

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

    def test_base_unet_repeats_fixed_first_frame_condition(self) -> None:
        model = small_model(condition_channels=3)
        video = torch.randn(1, 3, 3, 8, 8)
        condition = torch.randn(1, 3, 1, 4, 4)
        output = model(video, torch.tensor([3]), condition)
        self.assertEqual(output.shape, video.shape)

    def test_base_training_uses_first_frame_only_as_condition(self) -> None:
        values = torch.arange(4, dtype=torch.float32)[None, None, :, None, None]
        batch = {"video_56": values.expand(1, 3, 4, 2, 2).clone()}
        target, condition = stage_tensors(batch, "base", torch.device("cpu"))
        self.assertEqual(condition.shape, (1, 3, 1, 2, 2))
        self.assertEqual(target.shape, (1, 3, 3, 2, 2))
        torch.testing.assert_close(condition[:, :, 0], torch.zeros(1, 3, 2, 2))
        torch.testing.assert_close(
            target[:, 0, :, 0, 0], torch.tensor([[1.0, 2.0, 3.0]])
        )

    def test_superres_unet_is_strictly_2d(self) -> None:
        model = ImageUNet(
            ImageUNetConfig(
                base_channels=8,
                channel_multipliers=(1, 2),
                blocks_per_level=1,
                time_embedding_dim=32,
                gradient_checkpointing=False,
            )
        )
        image = torch.randn(3, 3, 8, 8)
        condition = torch.randn(3, 3, 4, 4)
        output = model(image, torch.tensor([3, 4, 5]), condition)
        self.assertEqual(output.shape, image.shape)
        self.assertTrue(any(isinstance(module, torch.nn.Conv2d) for module in model.modules()))
        self.assertFalse(any(isinstance(module, torch.nn.Conv3d) for module in model.modules()))

    def test_superres_training_flattens_three_frames_into_images(self) -> None:
        batch = {
            "video_56": torch.randn(2, 3, 4, 56, 56),
            "video_224": torch.randn(2, 3, 4, 224, 224),
        }
        target, condition = stage_tensors(batch, "superres", torch.device("cpu"))
        self.assertEqual(target.shape, (6, 3, 224, 224))
        self.assertEqual(condition.shape, (6, 3, 56, 56))

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

    def test_ddim_sampler_recovers_clean_data_from_oracle_noise(self) -> None:
        diffusion = GaussianDiffusion(timesteps=20)

        class OracleNoise(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(()))

            def forward(
                self,
                noisy: torch.Tensor,
                timesteps: torch.Tensor,
                clean: torch.Tensor,
            ) -> torch.Tensor:
                alpha = diffusion.alpha_bars.gather(0, timesteps).reshape(
                    timesteps.shape[0], *((1,) * (noisy.ndim - 1))
                )
                return (noisy - alpha.sqrt() * clean) / (1 - alpha).sqrt()

        clean = torch.rand(1, 3, 3, 8, 8).mul(1.6).sub(0.8)
        sampled = diffusion.sample(
            OracleNoise(),
            tuple(clean.shape),
            condition=clean,
            inference_steps=5,
            generator=torch.Generator().manual_seed(4),
        )
        torch.testing.assert_close(sampled, clean, atol=2e-5, rtol=2e-5)

    def test_mp4_keeps_first_frame_and_rgb_channel_order(self) -> None:
        video = torch.zeros(3, 4, 16, 16)
        video[:, 0] = torch.tensor([1.0, -1.0, -1.0])[:, None, None]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "color.mp4"
            write_mp4(path, video, fps=4.0)
            capture = cv2.VideoCapture(str(path))
            ok, first_bgr = capture.read()
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            capture.release()
        self.assertTrue(ok)
        self.assertEqual(frame_count, 4)
        first_rgb = cv2.cvtColor(first_bgr, cv2.COLOR_BGR2RGB)
        means = first_rgb.mean(axis=(0, 1))
        self.assertGreater(means[0], 200)
        self.assertLess(means[1], 30)
        self.assertLess(means[2], 30)


if __name__ == "__main__":
    unittest.main()
