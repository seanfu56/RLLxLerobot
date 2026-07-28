from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from policy.datasets import Bundle, ChunkDataset, compute_stats, default_drop_n_last_frames
from policy.tests.fixtures import write_bundle
from policy.config import PolicyConfig
from policy.inference import PolicyRunner
from policy.model import ChunkPolicy
from policy.objectives import DiffusionObjective, FlowMatchingObjective
from policy.train import main, parse_args

# Small enough to train in a couple of seconds, still exercising every stage:
# two U-Net levels means one downsample and one skip connection.
TINY = ["--down-dims", "16", "32", "--horizon", "8", "--n-action-steps", "4",
        "--random-init-backbone", "--num-keypoints", "4", "--vision-feature-dim", "8",
        "--diffusion-step-embed-dim", "16"]


def tiny_config(**overrides) -> PolicyConfig:
    defaults = dict(
        state_dim=7,
        action_dim=7,
        image_size=64,
        horizon=8,
        n_obs_steps=2,
        n_action_steps=4,
        pretrained_backbone=False,
        num_keypoints=4,
        vision_feature_dim=8,
        down_dims=(16, 32),
        diffusion_step_embed_dim=16,
        num_inference_steps=2,
    )
    defaults.update(overrides)
    return PolicyConfig(**defaults)


class ConfigTests(unittest.TestCase):
    def test_horizon_must_fit_the_downsampling_factor(self) -> None:
        # Three stages halve the chunk three times over in the reference check,
        # so 12 is not a legal horizon there but 16 is.
        with self.assertRaises(ValueError) as error:
            tiny_config(horizon=12, down_dims=(16, 32, 64))
        self.assertIn("multiple of 8", str(error.exception))
        tiny_config(horizon=16, down_dims=(16, 32, 64))

    def test_stage_widths_must_divide_the_group_count(self) -> None:
        with self.assertRaises(ValueError):
            tiny_config(down_dims=(12, 32))

    def test_clipping_requires_min_max_normalised_actions(self) -> None:
        with self.assertRaises(ValueError) as error:
            tiny_config(action_norm="meanstd")
        self.assertIn("clip_sample", str(error.exception))
        tiny_config(action_norm="meanstd", clip_sample=False)

    def test_execution_horizon_cannot_exceed_the_predicted_one(self) -> None:
        with self.assertRaises(ValueError):
            tiny_config(n_action_steps=16)

    def test_absolute_dimensions_only_apply_to_delta_targets(self) -> None:
        with self.assertRaises(ValueError):
            tiny_config(action_repr="absolute", absolute_dims=[6])
        tiny_config(action_repr="delta", absolute_dims=[6])

    def test_serialised_config_carries_what_the_run_picker_reads(self) -> None:
        payload = tiny_config().to_dict()
        self.assertEqual(payload["policy"], "simple")
        self.assertEqual(payload["objective"], "diffusion")
        self.assertEqual(payload["chunk_size"], payload["horizon"])

    def test_the_objective_reaches_the_run_picker(self) -> None:
        # The picker reads this string straight out of run.json, so a flow run
        # has to advertise itself as one rather than inheriting the default.
        self.assertEqual(tiny_config(objective="flow").to_dict()["objective"], "flow")
        with self.assertRaises(ValueError):
            tiny_config(objective="ddim")

    def test_round_trips_through_its_dictionary(self) -> None:
        for overrides in (
            dict(action_repr="delta", absolute_dims=[6], n_obs_steps=3),
            dict(objective="flow", flow_time_sampling="logitnormal"),
        ):
            with self.subTest(**overrides):
                original = tiny_config(**overrides)
                restored = PolicyConfig.from_dict(original.to_dict())
                self.assertEqual(restored.to_dict(), original.to_dict())

    def test_a_checkpoint_written_before_flow_matching_still_reads_as_diffusion(self) -> None:
        # objective used to be injected by to_dict rather than stored as a field.
        # Old configs therefore carry the key already, but a hand-written one may
        # not, and it must not become a flow run by accident.
        payload = tiny_config().to_dict()
        payload.pop("objective")
        self.assertEqual(PolicyConfig.from_dict(payload).objective, "diffusion")


class DatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.bundle = Bundle(write_bundle(root, "tiny", {"tiny": [20, 20]}, image_size=32))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_shapes_match_the_configured_horizons(self) -> None:
        dataset = ChunkDataset(self.bundle, self.bundle.episodes, horizon=8, n_obs_steps=3)
        sample = dataset[5]
        self.assertEqual(tuple(sample["image"].shape), (3, 3, 32, 32))
        self.assertEqual(tuple(sample["state"].shape), (3, 7))
        self.assertEqual(tuple(sample["action"].shape), (8, 7))
        self.assertEqual(sample["image"].dtype, torch.uint8)

    def test_history_is_clamped_at_the_episode_start(self) -> None:
        dataset = ChunkDataset(self.bundle, self.bundle.episodes, horizon=4, n_obs_steps=3)
        sample = dataset[0]
        # Every history step repeats frame 0 rather than reaching into the
        # previous episode, which would be a different demonstration entirely.
        self.assertTrue(torch.equal(sample["state"][0], sample["state"][-1]))
        self.assertTrue(torch.equal(sample["image"][0], sample["image"][-1]))

    def test_the_newest_observation_and_the_first_action_share_an_instant(self) -> None:
        dataset = ChunkDataset(self.bundle, self.bundle.episodes, horizon=4, n_obs_steps=2)
        sample = dataset[6]
        episode = self.bundle.episodes[0]
        states = self.bundle.states[episode.shard]
        actions = self.bundle.actions[episode.shard]
        np.testing.assert_allclose(sample["state"][-1].numpy(), states[episode.start + 6])
        np.testing.assert_allclose(sample["action"][0].numpy(), actions[episode.start + 6])
        # The fixture's action leads the state by exactly one step.
        np.testing.assert_allclose(sample["action"][0].numpy(), sample["state"][-1].numpy() + 1.0)

    def test_padding_is_flagged_at_the_episode_end(self) -> None:
        dataset = ChunkDataset(self.bundle, self.bundle.episodes, horizon=8, n_obs_steps=2)
        sample = dataset[len(self.bundle.episodes[0]) - 1]
        self.assertFalse(bool(sample["action_is_pad"][0]))
        self.assertTrue(bool(sample["action_is_pad"][-1]))

    def test_dropping_the_tail_shortens_the_index(self) -> None:
        full = ChunkDataset(self.bundle, self.bundle.episodes, horizon=8, n_obs_steps=2)
        dropped = ChunkDataset(
            self.bundle, self.bundle.episodes, horizon=8, n_obs_steps=2, drop_n_last_frames=3
        )
        self.assertEqual(len(full) - len(dropped), 3 * len(self.bundle.episodes))

    def test_the_default_drop_matches_the_reference_rule(self) -> None:
        self.assertEqual(default_drop_n_last_frames(16, 8, 2), 7)
        self.assertEqual(default_drop_n_last_frames(8, 8, 2), 0)

    def test_augmentation_uses_one_crop_for_the_whole_stack(self) -> None:
        # A per-frame crop would look like camera motion between the two
        # observation steps and corrupt the velocity the history encodes.
        torch.manual_seed(0)  # _rng seeds off torch's global state
        plain = ChunkDataset(self.bundle, self.bundle.episodes, horizon=4, n_obs_steps=2)
        cropped = ChunkDataset(
            self.bundle, self.bundle.episodes, horizon=4, n_obs_steps=2,
            augment=True, crop_scale=0.5, color_jitter=0.0,
        )

        # At each episode's first offset the history repeats one source frame,
        # so a shared crop box leaves the two observation steps byte-identical.
        episode_starts = []
        cursor = 0
        for episode in self.bundle.episodes:
            episode_starts.append(cursor)
            cursor += len(episode)
        for item in episode_starts:
            sample = cropped[item]
            self.assertTrue(torch.equal(sample["image"][0], sample["image"][1]))

        # A crop scale below 1 has to actually change the pixels. Any single
        # draw can land on the full frame, so this checks that not all of them do.
        changed = sum(
            not torch.equal(cropped[item]["image"][0], plain[item]["image"][0])
            for item in range(5)
        )
        self.assertGreater(changed, 0)


class ModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.bundle = Bundle(write_bundle(root, "tiny", {"tiny": [20, 20]}, image_size=64))
        self.stats = compute_stats(self.bundle, self.bundle.episodes, 8)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _batch(self, config: PolicyConfig, size: int = 2) -> dict[str, torch.Tensor]:
        dataset = ChunkDataset(
            self.bundle, self.bundle.episodes, horizon=config.horizon, n_obs_steps=config.n_obs_steps
        )
        samples = [dataset[index] for index in range(size)]
        return {key: torch.stack([sample[key] for sample in samples]) for key in samples[0]}

    def test_loss_and_prediction_have_the_expected_shapes(self) -> None:
        config = tiny_config(stats=self.stats)
        model = ChunkPolicy(config)
        batch = self._batch(config)

        loss, metrics = model.compute_loss(batch)
        self.assertEqual(loss.shape, ())
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("action_loss", metrics)

        prediction = model.predict(batch)
        self.assertEqual(tuple(prediction.shape), (2, config.horizon, config.action_dim))
        self.assertTrue(torch.isfinite(prediction).all())

    def test_conditioning_is_one_flat_vector_per_sample(self) -> None:
        config = tiny_config(stats=self.stats, n_obs_steps=2)
        model = ChunkPolicy(config)
        context = model.encode_context(self._batch(config))
        expected = (config.vision_feature_dim + config.state_dim) * config.n_obs_steps
        self.assertEqual(tuple(context.shape), (2, expected))

    def test_the_target_transform_is_invertible(self) -> None:
        # Whatever the model regresses has to map back to raw joint targets
        # exactly, or a correct prediction still commands the wrong pose.
        for action_repr, delta_mode, absolute_dims in (
            ("absolute", "incremental", []),
            ("delta", "incremental", [6]),
            ("delta", "anchored", [6]),
        ):
            with self.subTest(action_repr=action_repr, delta_mode=delta_mode):
                config = tiny_config(
                    stats=self.stats, action_repr=action_repr,
                    delta_mode=delta_mode, absolute_dims=absolute_dims,
                )
                model = ChunkPolicy(config)
                batch = self._batch(config)
                recovered = model._to_absolute(model._target_actions(batch), batch)
                torch.testing.assert_close(recovered, batch["action"], rtol=1e-4, atol=1e-3)

    def test_conditioning_reads_the_newest_observation_step(self) -> None:
        config = tiny_config(stats=self.stats, action_repr="delta", absolute_dims=[6])
        model = ChunkPolicy(config)
        batch = self._batch(config)
        torch.testing.assert_close(model._current_state(batch), batch["state"][:, -1])

    def test_padded_chunk_steps_are_excluded_from_the_loss(self) -> None:
        config = tiny_config(stats=self.stats)
        model = ChunkPolicy(config)
        batch = self._batch(config)

        torch.manual_seed(0)
        masked, _ = model.compute_loss({**batch, "action_is_pad": torch.ones_like(batch["action_is_pad"])})
        # Every step padded means every weight is zero, so the loss must vanish
        # rather than quietly averaging over nothing.
        self.assertAlmostEqual(float(masked), 0.0, places=6)


class FlowMatchingObjectiveTests(unittest.TestCase):
    """The objective on its own, with a stand-in decoder instead of the U-Net."""

    def test_the_regressed_velocity_is_the_straight_path_between_noise_and_data(self) -> None:
        # x_t = (1-t)*noise + t*action means action - x_t = (1-t)*(action - noise),
        # so a decoder handed x_t and t can reconstruct the training target
        # exactly. If the interpolation or the target were anything else, this
        # loss could not reach zero.
        torch.manual_seed(0)
        objective = FlowMatchingObjective(num_inference_steps=4)
        actions = torch.randn(8, 6, 7)

        def perfect(interpolated: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
            time = (times / objective.time_scale).view(-1, 1, 1)
            return (actions - interpolated) / (1 - time)

        loss, _ = objective.loss(perfect, actions, torch.ones_like(actions))
        self.assertAlmostEqual(float(loss), 0.0, places=5)

    def test_sampling_integrates_the_velocity_over_the_whole_unit_interval(self) -> None:
        # A constant velocity field must move the sample by exactly that
        # velocity: the Euler steps have to cover t=0 to t=1 and no further.
        torch.manual_seed(0)
        objective = FlowMatchingObjective(num_inference_steps=5, clip_sample=False)
        velocity = torch.full((2, 6, 7), 0.25)

        seen: list[torch.Tensor] = []

        def constant(sample: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
            seen.append(sample.clone())
            return velocity

        result = objective.sample(constant, (2, 6, 7), torch.device("cpu"))
        torch.testing.assert_close(result, seen[0] + velocity, rtol=1e-5, atol=1e-6)

    def test_the_step_count_is_exact(self) -> None:
        # Unlike the DDIM sampler, whose stride is integer division, N Euler
        # steps means N decoder calls for any N.
        for steps in (1, 3, 7, 10):
            calls = []
            objective = FlowMatchingObjective(num_inference_steps=steps)
            objective.sample(
                lambda sample, times: (calls.append(times) or torch.zeros_like(sample)),
                (1, 6, 7), torch.device("cpu"),
            )
            self.assertEqual(len(calls), steps)
            # The first call is at t=0, pure noise, and none of them reach t=1.
            self.assertAlmostEqual(float(calls[0][0]), 0.0)
            self.assertLess(float(calls[-1][0]), objective.time_scale)

    def test_clipping_keeps_the_sample_inside_the_normalised_range(self) -> None:
        # min-max normalisation puts every real action in [-1, 1]; a large
        # velocity - which is what an aggressive guidance weight produces - must
        # not be able to walk the sample out of it one Euler step at a time.
        objective = FlowMatchingObjective(num_inference_steps=5, clip_sample=True)
        result = objective.sample(
            lambda sample, times: torch.full_like(sample, 50.0), (2, 6, 7), torch.device("cpu")
        )
        self.assertLessEqual(float(result.max()), 1.0 + 1e-6)
        self.assertGreaterEqual(float(result.min()), -1.0 - 1e-6)

    def test_training_times_stay_on_the_open_unit_interval(self) -> None:
        torch.manual_seed(0)
        for sampling in ("uniform", "logitnormal"):
            with self.subTest(sampling=sampling):
                times = FlowMatchingObjective(time_sampling=sampling).sample_times(
                    512, torch.device("cpu")
                )
                self.assertGreaterEqual(float(times.min()), 0.0)
                self.assertLess(float(times.max()), 1.0)

    def test_an_unknown_time_distribution_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            FlowMatchingObjective(time_sampling="beta")


class FlowMatchingPolicyTests(unittest.TestCase):
    """The same network under the flow objective, end to end through the policy."""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.bundle = Bundle(write_bundle(root, "tiny", {"tiny": [20, 20]}, image_size=64))
        self.stats = compute_stats(self.bundle, self.bundle.episodes, 8)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _batch(self, config: PolicyConfig, size: int = 2) -> dict[str, torch.Tensor]:
        dataset = ChunkDataset(
            self.bundle, self.bundle.episodes, horizon=config.horizon, n_obs_steps=config.n_obs_steps
        )
        samples = [dataset[index] for index in range(size)]
        return {key: torch.stack([sample[key] for sample in samples]) for key in samples[0]}

    def test_the_config_selects_the_objective(self) -> None:
        self.assertIsInstance(ChunkPolicy(tiny_config()).objective, DiffusionObjective)
        self.assertIsInstance(ChunkPolicy(tiny_config(objective="flow")).objective, FlowMatchingObjective)

    def test_only_the_objective_differs_between_the_two_runs(self) -> None:
        # Same encoder, same U-Net, same conditioning: switching the objective
        # must not quietly change the model's size or its weight names.
        diffusion = ChunkPolicy(tiny_config(stats=self.stats))
        flow = ChunkPolicy(tiny_config(stats=self.stats, objective="flow"))
        self.assertEqual(diffusion.parameter_counts(), flow.parameter_counts())
        trainable = {name for name, _ in diffusion.named_parameters()}
        self.assertEqual(trainable, {name for name, _ in flow.named_parameters()})

    def test_loss_and_prediction_have_the_expected_shapes(self) -> None:
        config = tiny_config(stats=self.stats, objective="flow")
        model = ChunkPolicy(config)
        batch = self._batch(config)

        loss, metrics = model.compute_loss(batch)
        self.assertEqual(loss.shape, ())
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("action_loss", metrics)
        loss.backward()
        self.assertTrue(any(p.grad is not None for p in model.unet.parameters()))

        prediction = model.predict(batch)
        self.assertEqual(tuple(prediction.shape), (2, config.horizon, config.action_dim))
        self.assertTrue(torch.isfinite(prediction).all())

    def test_padded_chunk_steps_are_excluded_from_the_loss(self) -> None:
        config = tiny_config(stats=self.stats, objective="flow")
        model = ChunkPolicy(config)
        batch = self._batch(config)
        torch.manual_seed(0)
        masked, _ = model.compute_loss({**batch, "action_is_pad": torch.ones_like(batch["action_is_pad"])})
        self.assertAlmostEqual(float(masked), 0.0, places=6)

    def test_guidance_works_the_same_way_under_flow_matching(self) -> None:
        # Guiding a velocity is the same linear extrapolation as guiding an
        # epsilon prediction, so the policy hands both objectives one decoder.
        config = tiny_config(stats=self.stats, objective="flow", cond_dropout=0.1)
        model = ChunkPolicy(config).eval()
        context = model.encode_context(self._batch(config))
        action = torch.randn(2, config.horizon, config.action_dim)
        timesteps = torch.zeros(2)

        conditional = model._decoder_fn(context, 1.0)(action, timesteps)
        unconditional = model.unet(action, timesteps, model.unconditional_context(2, context.dtype))
        torch.testing.assert_close(
            model._decoder_fn(context, 2.5)(action, timesteps),
            unconditional + 2.5 * (conditional - unconditional),
            rtol=1e-4, atol=1e-5,
        )

    def test_the_vision_encoder_runs_once_and_the_unet_once_per_step(self) -> None:
        config = tiny_config(stats=self.stats, objective="flow", num_inference_steps=3)
        model = ChunkPolicy(config).eval()
        calls = {"encoder": 0, "unet": 0}
        original_encoder, original_unet = model.rgb_encoder.forward, model.unet.forward
        model.rgb_encoder.forward = lambda *a, **k: (calls.__setitem__("encoder", calls["encoder"] + 1)
                                                     or original_encoder(*a, **k))
        model.unet.forward = lambda *a, **k: (calls.__setitem__("unet", calls["unet"] + 1)
                                              or original_unet(*a, **k))
        try:
            model.predict(self._batch(config))
        finally:
            model.rgb_encoder.forward, model.unet.forward = original_encoder, original_unet
        self.assertEqual(calls, {"encoder": 1, "unet": 3})


class GuidanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.bundle = Bundle(write_bundle(root, "tiny", {"tiny": [20, 20]}, image_size=64))
        self.stats = compute_stats(self.bundle, self.bundle.episodes, 8)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _batch(self, config: PolicyConfig, size: int = 2) -> dict[str, torch.Tensor]:
        dataset = ChunkDataset(
            self.bundle, self.bundle.episodes, horizon=config.horizon, n_obs_steps=config.n_obs_steps
        )
        samples = [dataset[index] for index in range(size)]
        return {key: torch.stack([sample[key] for sample in samples]) for key in samples[0]}

    def test_a_weight_without_conditioning_dropout_is_rejected(self) -> None:
        # The whole hazard of guidance: an untrained null embedding produces
        # confident nonsense rather than an error, so this has to be refused.
        with self.assertRaises(ValueError) as error:
            tiny_config(guidance_weight=2.0)
        self.assertIn("cond_dropout", str(error.exception))
        tiny_config(guidance_weight=2.0, cond_dropout=0.1)

    def test_predict_refuses_a_weight_the_policy_cannot_support(self) -> None:
        config = tiny_config(stats=self.stats)
        model = ChunkPolicy(config)
        with self.assertRaises(ValueError):
            model.predict(self._batch(config), guidance_weight=2.0)

    def test_conditioning_is_dropped_only_while_training(self) -> None:
        config = tiny_config(stats=self.stats, cond_dropout=0.5)
        model = ChunkPolicy(config)
        batch = self._batch(config, size=32)

        def dropped_rows(context: torch.Tensor) -> int:
            return int((context == model.null_cond).all(dim=-1).sum())

        torch.manual_seed(0)
        model.train()
        rows = dropped_rows(model.encode_context(batch))
        self.assertGreater(rows, 0)   # some samples see the null embedding
        self.assertLess(rows, 32)     # and some still see the observation

        # At eval time the real observation must always get through, whatever
        # the dropout rate was during training.
        model.eval()
        self.assertEqual(dropped_rows(model.encode_context(batch)), 0)

    def test_guidance_extrapolates_away_from_the_unconditional_branch(self) -> None:
        config = tiny_config(stats=self.stats, cond_dropout=0.1)
        model = ChunkPolicy(config).eval()
        context = model.encode_context(self._batch(config))
        action = torch.randn(2, config.horizon, config.action_dim)
        timesteps = torch.zeros(2)

        conditional = model._decoder_fn(context, 1.0)(action, timesteps)
        unconditional_context = model.unconditional_context(2, context.dtype)
        unconditional = model.unet(action, timesteps, unconditional_context)
        guided = model._decoder_fn(context, 2.5)(action, timesteps)

        # eps = uncond + w * (cond - uncond), exactly.
        torch.testing.assert_close(
            guided, unconditional + 2.5 * (conditional - unconditional), rtol=1e-4, atol=1e-5
        )

    def test_a_weight_of_one_reproduces_plain_conditional_sampling(self) -> None:
        config = tiny_config(stats=self.stats, cond_dropout=0.1)
        model = ChunkPolicy(config).eval()
        context = model.encode_context(self._batch(config))
        action = torch.randn(2, config.horizon, config.action_dim)
        timesteps = torch.zeros(2)
        torch.testing.assert_close(
            model._decoder_fn(context, 1.0)(action, timesteps),
            model.unet(action, timesteps, context),
        )

    def test_the_vision_encoder_runs_once_regardless_of_guidance(self) -> None:
        # Guidance doubles the U-Net work, not the whole policy: the conditional
        # context is encoded once and the unconditional one is a constant.
        # A divisor of num_train_timesteps, so the sampler's stride arithmetic
        # lands on exactly this many steps.
        config = tiny_config(stats=self.stats, cond_dropout=0.1, num_inference_steps=4)
        model = ChunkPolicy(config).eval()
        batch = self._batch(config)

        calls = {"encoder": 0, "unet": 0}
        original_encoder, original_unet = model.rgb_encoder.forward, model.unet.forward
        model.rgb_encoder.forward = lambda *a, **k: (calls.__setitem__("encoder", calls["encoder"] + 1)
                                                     or original_encoder(*a, **k))
        model.unet.forward = lambda *a, **k: (calls.__setitem__("unet", calls["unet"] + 1)
                                              or original_unet(*a, **k))
        try:
            model.predict(batch, guidance_weight=2.0)
        finally:
            model.rgb_encoder.forward, model.unet.forward = original_encoder, original_unet

        self.assertEqual(calls["encoder"], 1)
        self.assertEqual(calls["unet"], config.num_inference_steps)


class TrainAndInferenceTests(unittest.TestCase):
    """Train a few steps, then drive the checkpoint the way the robot loop does."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        root = Path(cls.temporary.name)
        write_bundle(root / "data", "tiny", {"tiny": [16, 16, 16]}, image_size=64)
        cls.output_root = root / "runs"
        main([
            "--bundle", "tiny", "--data-root", str(root / "data"),
            "--output-root", str(cls.output_root), "--run-name", "run",
            "--steps", "4", "--batch-size", "2", "--num-workers", "0",
            "--val-episodes", "1", "--eval-freq", "4", "--eval-batches", "1",
            "--eval-samples", "2", "--save-freq", "0", "--log-freq", "4",
            "--num-inference-steps", "2", "--device", "cpu", "--amp", "off",
            *TINY,
        ])
        cls.run_dir = cls.output_root / "run"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_the_run_directory_has_the_shared_layout(self) -> None:
        for name in ("run.json", "log.jsonl", "best.pt", "final.pt"):
            self.assertTrue((self.run_dir / name).is_file(), name)
        metadata = json.loads((self.run_dir / "run.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["config"]["policy"], "simple")
        self.assertEqual(metadata["config"]["chunk_size"], 8)
        self.assertGreater(metadata["parameters"]["unet"], 0)

    def test_validation_reports_the_best_of_n_error(self) -> None:
        records = [
            json.loads(line)
            for line in (self.run_dir / "log.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        evaluations = [record for record in records if "val_mae" in record]
        self.assertTrue(evaluations)
        self.assertIn("val_mae_bon", evaluations[-1])
        # The closest of several draws can never be worse than a single draw's
        # average, so this ordering is a real invariant rather than a threshold.
        self.assertLessEqual(evaluations[-1]["val_mae_bon"], evaluations[-1]["val_mae"] + 1e-6)

    def test_the_run_is_discoverable_by_the_checkpoint_picker(self) -> None:
        # The picker reads run.json without torch, so the config has to carry
        # the descriptive keys RunInfo looks for.
        from policy.checkpoints import discover_runs

        # Other tests in this class write extra run directories under the same
        # root, so pick this one out by name rather than assuming it is alone.
        runs = {run.name: run for run in discover_runs(self.output_root)}
        self.assertIn("run", runs)
        self.assertEqual(runs["run"].policy, "simple")
        self.assertEqual(runs["run"].chunk_size, 8)
        self.assertIn("best", [entry.tag for entry in runs["run"].checkpoints])

    def _runner(self, **kwargs) -> PolicyRunner:
        return PolicyRunner(self.run_dir / "final.pt", device="cpu", **kwargs)

    def test_the_runner_defaults_to_the_trained_execution_horizon(self) -> None:
        runner = self._runner()
        self.assertEqual(runner.action_steps, 4)
        self.assertEqual(runner.config.horizon, 8)

    def test_actions_are_replanned_once_per_execution_horizon(self) -> None:
        runner = self._runner(action_steps=3)
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        state = np.zeros(7, dtype=np.float32)

        queued = []
        for _ in range(7):
            queued.append(runner.queued_actions)
            action = runner.select_action(frame, state)
            self.assertEqual(action.shape, (7,))
        # A prediction refills the queue with three actions, so it runs dry
        # every third call rather than on every step.
        self.assertEqual(queued, [0, 2, 1, 0, 2, 1, 0])

    def test_every_frame_is_recorded_even_when_an_action_is_queued(self) -> None:
        # The observation history has to span consecutive control steps. If the
        # runner only looked at frames on the steps where it re-plans, the two
        # frames would be a whole execution horizon apart.
        runner = self._runner(action_steps=4)
        state = np.zeros(7, dtype=np.float32)
        for index in range(3):
            runner.select_action(np.full((48, 64, 3), index * 40, dtype=np.uint8), state)
        newest, previous = runner.frames[-1], runner.frames[-2]
        self.assertFalse(torch.equal(newest, previous))

    def test_the_history_is_filled_on_the_first_observation(self) -> None:
        runner = self._runner()
        runner.select_action(np.zeros((48, 64, 3), dtype=np.uint8), np.zeros(7, dtype=np.float32))
        self.assertEqual(len(runner.frames), runner.config.n_obs_steps)
        self.assertTrue(torch.equal(runner.frames[0], runner.frames[-1]))

    def test_reset_clears_the_queue_and_the_history(self) -> None:
        runner = self._runner()
        runner.select_action(np.zeros((48, 64, 3), dtype=np.uint8), np.zeros(7, dtype=np.float32))
        runner.reset()
        self.assertEqual(runner.queued_actions, 0)
        self.assertEqual(len(runner.frames), 0)

    def test_the_sampler_step_count_can_be_overridden(self) -> None:
        self.assertEqual(self._runner(num_inference_steps=3).num_inference_steps, 3)
        with self.assertRaises(ValueError):
            self._runner(num_inference_steps=0)

    def test_actions_are_named_after_the_bundle_joints(self) -> None:
        runner = self._runner()
        named = runner.action_dict(np.arange(7, dtype=np.float32))
        self.assertEqual(list(named), list(runner.joint_names))
        self.assertEqual(named["gripper.pos"], 6.0)
        with self.assertRaises(ValueError):
            runner.action_dict(np.zeros(3, dtype=np.float32))

    def test_warmup_leaves_no_dummy_observation_behind(self) -> None:
        runner = self._runner()
        self.assertGreater(runner.warmup(frame_shape=(48, 64)), 0.0)
        self.assertEqual(runner.queued_actions, 0)
        self.assertEqual(len(runner.frames), 0)

    def test_describe_covers_every_field_the_runtime_reads(self) -> None:
        # These are the keys src/ui/policy_runtime.py pulls out of describe() to
        # build its PolicyInfo. Dropping one breaks loading only at runtime.
        required = {
            "checkpoint", "run_name", "step", "policy", "objective", "chunk_size",
            "action_steps", "num_inference_steps", "image_size", "vision", "pool",
            "action_repr", "delta_mode", "absolute_dims", "joint_names", "fps",
            "device", "use_ema", "n_obs_steps", "down_dims", "guidance_weight", "cond_dropout",
        }
        description = self._runner().describe()
        self.assertFalse(required - set(description))
        self.assertEqual(description["policy"], "simple")
        self.assertEqual(description["n_obs_steps"], 2)

    def test_ema_weights_are_used_by_default(self) -> None:
        self.assertTrue(self._runner().use_ema)
        self.assertFalse(self._runner(use_ema=False).use_ema)

    def test_a_checkpoint_predating_guidance_still_loads(self) -> None:
        # null_cond was added with classifier-free guidance. A run trained before
        # it must keep working at weight 1.0 rather than failing to load.
        payload = torch.load(self.run_dir / "final.pt", map_location="cpu", weights_only=False)
        payload["model"] = {k: v for k, v in payload["model"].items() if k != "null_cond"}
        payload["config"].pop("cond_dropout", None)
        payload["config"].pop("guidance_weight", None)
        legacy = self.run_dir.parent / "legacy"
        legacy.mkdir(parents=True, exist_ok=True)
        torch.save(payload, legacy / "final.pt")

        runner = PolicyRunner(legacy / "final.pt", device="cpu")
        self.assertFalse(runner.supports_guidance)
        self.assertEqual(runner.guidance_weight, 1.0)
        self.assertEqual(runner.predict_chunk(
            np.zeros((48, 64, 3), dtype=np.uint8), np.zeros(7, dtype=np.float32)
        ).shape, (8, 7))

    def test_a_genuinely_mismatched_checkpoint_is_still_rejected(self) -> None:
        payload = torch.load(self.run_dir / "final.pt", map_location="cpu", weights_only=False)
        payload["model"] = {k: v for k, v in payload["model"].items() if not k.startswith("unet.")}
        broken = self.run_dir.parent / "broken"
        broken.mkdir(parents=True, exist_ok=True)
        torch.save(payload, broken / "final.pt")
        with self.assertRaises(ValueError):
            PolicyRunner(broken / "final.pt", device="cpu")

    def test_the_runner_refuses_guidance_this_checkpoint_cannot_support(self) -> None:
        # This run trained without --cond-dropout, so guidance must fail at load
        # time rather than silently on the arm.
        runner = self._runner()
        self.assertFalse(runner.supports_guidance)
        self.assertEqual(runner.guidance_weight, 1.0)
        with self.assertRaises(ValueError) as error:
            self._runner(guidance_weight=2.0)
        self.assertIn("cond_dropout", str(error.exception))
        with self.assertRaises(ValueError):
            runner.set_guidance_weight(1.5)


class GuidedTrainingTests(unittest.TestCase):
    """A run trained with conditioning dropout can actually be guided."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        root = Path(cls.temporary.name)
        write_bundle(root / "data", "tiny", {"tiny": [16, 16, 16]}, image_size=64)
        cls.run_dir = root / "runs" / "guided"
        main([
            "--bundle", "tiny", "--data-root", str(root / "data"),
            "--output-root", str(root / "runs"), "--run-name", "guided",
            "--steps", "4", "--batch-size", "2", "--num-workers", "0",
            "--val-episodes", "1", "--eval-freq", "0", "--save-freq", "0", "--log-freq", "4",
            "--num-inference-steps", "2", "--device", "cpu", "--amp", "off",
            "--cond-dropout", "0.1", "--guidance-weight", "1.5",
            *TINY,
        ])

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_the_checkpoint_carries_its_guidance_settings(self) -> None:
        config = json.loads((self.run_dir / "run.json").read_text(encoding="utf-8"))["config"]
        self.assertAlmostEqual(config["cond_dropout"], 0.1)
        self.assertAlmostEqual(config["guidance_weight"], 1.5)

    def test_the_runner_defaults_to_the_trained_weight_and_accepts_others(self) -> None:
        runner = PolicyRunner(self.run_dir / "final.pt", device="cpu")
        self.assertTrue(runner.supports_guidance)
        self.assertAlmostEqual(runner.guidance_weight, 1.5)

        override = PolicyRunner(self.run_dir / "final.pt", device="cpu", guidance_weight=3.0)
        self.assertAlmostEqual(override.guidance_weight, 3.0)
        override.set_guidance_weight(1.0)
        self.assertAlmostEqual(override.guidance_weight, 1.0)
        with self.assertRaises(ValueError):
            override.set_guidance_weight(-1.0)

    def test_guidance_changes_the_commanded_action(self) -> None:
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        state = np.zeros(7, dtype=np.float32)
        commands = []
        for weight in (1.0, 3.0):
            runner = PolicyRunner(self.run_dir / "final.pt", device="cpu", guidance_weight=weight)
            torch.manual_seed(0)  # same sampler noise, so only guidance differs
            commands.append(runner.predict_chunk(frame, state))
        self.assertFalse(np.allclose(commands[0], commands[1]))

    def test_describe_reports_guidance_to_the_ui(self) -> None:
        description = PolicyRunner(self.run_dir / "final.pt", device="cpu").describe()
        self.assertAlmostEqual(description["guidance_weight"], 1.5)
        self.assertAlmostEqual(description["cond_dropout"], 0.1)


class FlowTrainingTests(unittest.TestCase):
    """A flow-matching run trains, saves, and loads through the robot interface."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        root = Path(cls.temporary.name)
        write_bundle(root / "data", "tiny", {"tiny": [16, 16, 16]}, image_size=64)
        cls.run_dir = root / "runs" / "flow"
        main([
            "--bundle", "tiny", "--data-root", str(root / "data"),
            "--output-root", str(root / "runs"), "--run-name", "flow",
            "--steps", "4", "--batch-size", "2", "--num-workers", "0",
            "--val-episodes", "1", "--eval-freq", "4", "--eval-batches", "1",
            "--eval-samples", "2", "--save-freq", "0", "--log-freq", "4",
            "--num-inference-steps", "3", "--device", "cpu", "--amp", "off",
            "--objective", "flow", "--flow-time-sampling", "logitnormal",
            "--cond-dropout", "0.1",
            *TINY,
        ])

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_the_run_records_its_objective(self) -> None:
        config = json.loads((self.run_dir / "run.json").read_text(encoding="utf-8"))["config"]
        self.assertEqual(config["objective"], "flow")
        self.assertEqual(config["flow_time_sampling"], "logitnormal")

    def test_the_checkpoint_picker_reads_it_as_a_flow_run(self) -> None:
        from policy.checkpoints import discover_runs

        runs = {run.name: run for run in discover_runs(self.run_dir.parent)}
        self.assertEqual(runs["flow"].objective, "flow")

    def test_the_runner_rebuilds_the_flow_objective(self) -> None:
        runner = PolicyRunner(self.run_dir / "final.pt", device="cpu")
        self.assertIsInstance(runner.policy.objective, FlowMatchingObjective)
        self.assertEqual(runner.describe()["objective"], "flow")
        # No stride rounding here, so the runner reports what it was asked for.
        self.assertEqual(runner.num_inference_steps, 3)

    def test_it_drives_the_control_loop_like_any_other_checkpoint(self) -> None:
        runner = PolicyRunner(self.run_dir / "final.pt", device="cpu", action_steps=2)
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        state = np.zeros(7, dtype=np.float32)
        for _ in range(3):
            self.assertEqual(runner.select_action(frame, state).shape, (7,))

    def test_guidance_is_available_under_flow_matching_too(self) -> None:
        runner = PolicyRunner(self.run_dir / "final.pt", device="cpu", guidance_weight=2.5)
        self.assertTrue(runner.supports_guidance)
        commands = []
        for weight in (1.0, 2.5):
            runner.set_guidance_weight(weight)
            torch.manual_seed(0)  # same sampler noise, so only guidance differs
            commands.append(runner.predict_chunk(np.zeros((48, 64, 3), dtype=np.uint8),
                                                 np.zeros(7, dtype=np.float32)))
            runner.reset()
        self.assertFalse(np.allclose(commands[0], commands[1]))


class ArgumentTests(unittest.TestCase):
    def test_the_defaults_are_the_published_recipe(self) -> None:
        args = parse_args(["--bundle", "x"])
        self.assertEqual((args.horizon, args.n_obs_steps, args.n_action_steps), (16, 2, 8))
        self.assertEqual(args.down_dims, [128, 256, 512])
        self.assertEqual((args.num_train_timesteps, args.num_inference_steps), (100, 10))
        # Diffusion stays the default: it is the published recipe, and every
        # existing run in models/simple was trained under it.
        self.assertEqual(args.objective, "diffusion")
        # Random cropping is on by default: it is the single most effective
        # regulariser at this data scale, and the transformer sweeps had it off.
        self.assertTrue(args.augment)
        self.assertLess(args.crop_scale, 1.0)

    def test_flags_belonging_to_the_other_objective_are_flagged(self) -> None:
        # Silently ignoring --beta-schedule on a flow run would cost a whole
        # training run before anyone noticed it did nothing.
        from policy.train import warn_about_ignored_flags

        quiet = parse_args(["--bundle", "x", "--objective", "flow"])
        self.assertEqual(warn_about_ignored_flags(quiet), [])

        noisy = parse_args(["--bundle", "x", "--objective", "flow", "--beta-schedule", "linear"])
        self.assertEqual(warn_about_ignored_flags(noisy), ["--beta-schedule"])

        reversed_ = parse_args(["--bundle", "x", "--flow-time-sampling", "logitnormal"])
        self.assertEqual(warn_about_ignored_flags(reversed_), ["--flow-time-sampling"])


if __name__ == "__main__":
    unittest.main()
