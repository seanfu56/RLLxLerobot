from __future__ import annotations

import unittest
from unittest.mock import patch

import torch
from torch import nn

from cosmos.diverse_flow import (
    DinoTrajectoryFeatures,
    DiverseFlowConfig,
    diversity_gradients,
    dpp_kernel,
    dpp_log_likelihood,
    gaussian_source_quality,
    latent_trajectory_features,
    sample_diverse_latents,
)
from cosmos.flow import build_condition_mask
from cosmos.sample import parse_args


class DPPObjectiveTests(unittest.TestCase):
    def test_separated_features_have_higher_likelihood(self) -> None:
        close = torch.tensor([[1.0, 0.0], [1.0, 0.01]])
        separated = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        close_kernel, _ = dpp_kernel(close, distance_scale=1.0)
        separated_kernel, _ = dpp_kernel(separated, distance_scale=1.0)
        self.assertGreater(
            float(dpp_log_likelihood(separated_kernel)),
            float(dpp_log_likelihood(close_kernel)),
        )

    def test_near_duplicate_particles_remain_finite(self) -> None:
        features = torch.tensor([[1.0, 2.0], [1.0, 2.0 + 1e-7]], requires_grad=True)
        kernel, _ = dpp_kernel(features)
        objective = dpp_log_likelihood(kernel, jitter=1e-4)
        (gradient,) = torch.autograd.grad(objective, features)
        self.assertTrue(bool(torch.isfinite(objective)))
        self.assertTrue(bool(torch.isfinite(gradient).all()))

    def test_quality_ignores_the_clean_condition_frame(self) -> None:
        mask = build_condition_mask(3, device="cpu", dtype=torch.float32)
        source = torch.zeros(2, 1, 3, 1, 1)
        source[0, :, 0] = 1e6
        source[1, :, 1:] = 1e6
        quality = gaussian_source_quality(
            source,
            mask,
            percentile=0.9,
            floor=1e-3,
            strength=1.0,
        )
        assert quality is not None
        self.assertAlmostEqual(float(quality[0]), 1.0)
        self.assertAlmostEqual(float(quality[1]), 1e-3)


class DiverseGradientTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.mask = build_condition_mask(4, device="cpu", dtype=torch.float32)
        first = torch.randn(1, 2, 4, 3, 3)
        second = first.clone()
        second[:, :, 1:] += 0.02 * torch.randn_like(second[:, :, 1:])
        self.particles = torch.cat([first, second], dim=0)
        self.velocities = torch.zeros_like(self.particles)
        self.config = DiverseFlowConfig(
            steps=2,
            feature="latent",
            feature_frames=3,
            quality_percentile=0,
            diversity_scale=1.0,
        )

    def _objective(self, particles: torch.Tensor, distance_scale: float) -> float:
        features = latent_trajectory_features(
            particles,
            self.mask,
            temporal_features=self.config.feature_frames,
        )
        kernel, _ = dpp_kernel(
            features,
            bandwidth=self.config.dpp_bandwidth,
            distance_scale=distance_scale,
        )
        return float(dpp_log_likelihood(kernel, jitter=self.config.dpp_jitter))

    def test_gradient_is_joint_normalized_and_masks_the_condition(self) -> None:
        gradient, _ = diversity_gradients(
            self.particles,
            self.velocities,
            0.5,
            self.mask,
            self.config,
        )
        torch.testing.assert_close(gradient[:, :, 0], torch.zeros_like(gradient[:, :, 0]))
        self.assertAlmostEqual(float(gradient.square().sum().sqrt()), 1.0, places=5)

    def test_cosmos_reverse_time_update_ascends_dpp_likelihood(self) -> None:
        sigma = 0.5
        gradient, stats = diversity_gradients(
            self.particles,
            self.velocities,
            sigma,
            self.mask,
            self.config,
        )
        delta_sigma = -1e-3
        guided_velocity = self.velocities - sigma * gradient
        updated = self.particles + delta_sigma * guided_velocity
        before = self._objective(self.particles, stats.distance_scale)
        after = self._objective(updated, stats.distance_scale)
        self.assertGreater(after, before)


class DinoTrajectoryFeatureTests(unittest.TestCase):
    def test_generated_frames_are_ordered_and_differentiable(self) -> None:
        class TinyBackbone(nn.Module):
            def forward_features(self, frames: torch.Tensor) -> dict[str, torch.Tensor]:
                return {"x_norm_clstoken": frames.mean(dim=(2, 3))}

        extractor = DinoTrajectoryFeatures.__new__(DinoTrajectoryFeatures)
        extractor.variant = "test"
        extractor.frames = 2
        extractor.device = torch.device("cpu")
        extractor.image_size = 14
        extractor.embed_dim = 3
        extractor.model = TinyBackbone()
        extractor.mean = torch.zeros(1, 3, 1, 1)
        extractor.std = torch.ones(1, 3, 1, 1)

        video = torch.randn(1, 3, 5, 16, 16, requires_grad=True)
        features = extractor(video)
        self.assertEqual(features.shape, (1, 6))
        (gradient,) = torch.autograd.grad(features.sum(), video)
        torch.testing.assert_close(gradient[:, :, 0], torch.zeros_like(gradient[:, :, 0]))
        self.assertGreater(float(gradient[:, :, 1:].abs().sum()), 0.0)


class DiverseCLIArgumentTests(unittest.TestCase):
    def test_positive_diversity_rejects_one_particle(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least two|>= 2"):
            parse_args(
                [
                    "--images-dir",
                    "snapshots/pick-can",
                    "--sampler",
                    "diverse-flow",
                    "--samples-per-condition",
                    "1",
                    "--seed",
                    "1",
                ]
            )

    def test_diverse_options_are_exposed(self) -> None:
        args = parse_args(
            [
                "--images-dir",
                "snapshots/pick-can",
                "--sampler",
                "diverse-flow",
                "--samples-per-condition",
                "2",
                "--seed",
                "10",
                "--diversity-feature",
                "latent",
                "--diversity-scale",
                "2.5",
                "--diversity-every",
                "3",
            ]
        )
        self.assertEqual(args.sampler, "diverse-flow")
        self.assertEqual(args.sample_seeds, [10, 11])
        self.assertEqual(args.diversity_feature, "latent")
        self.assertEqual(args.diversity_scale, 2.5)
        self.assertEqual(args.diversity_every, 3)


class _FakeTransformer(nn.Module):
    @property
    def dtype(self) -> torch.dtype:
        return torch.float32


class _FakePipe:
    def __init__(self) -> None:
        self.transformer = _FakeTransformer()
        self.vae = nn.Identity()


class CoupledSamplerTests(unittest.TestCase):
    def test_euler_loop_preserves_the_condition_latent(self) -> None:
        condition = torch.zeros(1, 2, 4, 2, 2)
        condition[:, :, 0] = 3.0
        config = DiverseFlowConfig(
            steps=2,
            cfg_scale=1.0,
            diversity_scale=1.0,
            feature="latent",
            quality_percentile=0,
            log_every=1,
        )

        def zero_velocity(_pipe, _transformer, latent, *_args, **_kwargs):
            return torch.zeros_like(latent)

        with patch("cosmos.diverse_flow.predict_velocity", side_effect=zero_velocity):
            result = sample_diverse_latents(
                _FakePipe(),
                condition,
                "pick the can",
                "",
                [10, 11],
                config,
                device=torch.device("cpu"),
                fps=20.0,
                num_frames=13,
                resolution=32,
            )

        self.assertEqual(len(result), 2)
        for particle in result:
            torch.testing.assert_close(particle[:, :, 0], condition[:, :, 0])
        self.assertFalse(torch.equal(result[0][:, :, 1:], result[1][:, :, 1:]))


if __name__ == "__main__":
    unittest.main()
