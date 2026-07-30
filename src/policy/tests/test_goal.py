from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from policy.config import PolicyConfig
from policy.datasets import Bundle
from policy.goal import (
    GoalChunkDataset,
    GoalChunkPolicy,
    GoalPolicyRunner,
    goal_offset_range,
    is_goal_conditioned,
    load_runner,
    uniform_frame_offsets,
)
from policy.inference import PolicyRunner
from policy.model import ChunkPolicy
from policy.tests.fixtures import write_bundle
from policy.train import main, parse_args

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
        goal_conditioned=True,
    )
    defaults.update(overrides)
    return PolicyConfig(**defaults)


class GoalWindowTests(unittest.TestCase):
    def test_the_goal_comes_from_the_last_n_frames(self) -> None:
        self.assertEqual(goal_offset_range(100, 10), (90, 100))

    def test_a_short_episode_offers_every_frame_it_has(self) -> None:
        # Six frames cannot yield a ten-frame window; the alternative is an
        # empty or negative range, which would index the wrong episode.
        self.assertEqual(goal_offset_range(6, 10), (0, 6))

    def test_a_window_of_one_is_the_final_frame_alone(self) -> None:
        self.assertEqual(goal_offset_range(100, 1), (99, 100))

    def test_degenerate_arguments_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            goal_offset_range(0, 10)
        with self.assertRaises(ValueError):
            goal_offset_range(100, 0)


class GoalDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        write_bundle(root, "tiny", {"tiny": [30, 30]}, image_size=32)
        self.bundle = Bundle(root / "tiny")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _dataset(self, **kwargs) -> GoalChunkDataset:
        options = dict(horizon=8, n_obs_steps=2, goal_window=10)
        options.update(kwargs)
        return GoalChunkDataset(self.bundle, self.bundle.episodes, **options)

    def test_every_sample_carries_a_goal_frame(self) -> None:
        sample = self._dataset()[0]
        # One frame, not a stack: the goal has no history to express.
        self.assertEqual(sample["goal_image"].shape, (3, 32, 32))
        self.assertEqual(sample["goal_image"].dtype, torch.uint8)
        self.assertEqual(sample["image"].shape, (2, 3, 32, 32))

    def test_a_pinned_tail_goal_is_the_final_frame_of_the_episode(self) -> None:
        dataset = self._dataset(goal_selection="tail", random_goal=False)
        episode = self.bundle.episodes[0]
        expected = np.array(self.bundle.frames[episode.shard][episode.end - 1])
        np.testing.assert_array_equal(
            dataset[0]["goal_image"].permute(1, 2, 0).numpy(), expected
        )

    def test_a_random_tail_goal_stays_inside_the_window(self) -> None:
        dataset = self._dataset(goal_selection="tail", goal_window=10)
        episode = self.bundle.episodes[0]
        tail = {
            bytes(np.array(self.bundle.frames[episode.shard][episode.end - offset]).data)
            for offset in range(1, 11)
        }
        drawn = {
            bytes(dataset[0]["goal_image"].permute(1, 2, 0).contiguous().numpy().data)
            for _ in range(40)
        }
        self.assertTrue(drawn <= tail)
        # 40 draws from 10 frames landing on one of them would be a stuck index.
        self.assertGreater(len(drawn), 1)

    def test_a_pinned_uniform4_goal_is_two_thirds_through_the_episode(self) -> None:
        dataset = self._dataset(random_goal=False)
        episode = self.bundle.episodes[0]
        # 30 frames, so the pinned end is offset 29 and the four sampled offsets
        # are 0, 10, 19, 29. The third of them is the goal.
        self.assertEqual(uniform_frame_offsets(29), [0, 10, 19, 29])
        expected = np.array(self.bundle.frames[episode.shard][episode.start + 19])
        np.testing.assert_array_equal(
            dataset[0]["goal_image"].permute(1, 2, 0).numpy(), expected
        )

    def test_a_random_uniform4_goal_moves_with_the_sampled_endpoint(self) -> None:
        """The endpoint comes from the tail window, so the goal moves a little.

        With a 30-frame episode and a 10-frame window the endpoint is one of
        20..29, which puts the two-thirds frame at offsets 13..19 - well before
        the end, and never in the tail the old rule drew from.
        """
        dataset = self._dataset(goal_window=10)
        episode = self.bundle.episodes[0]
        reachable = {
            bytes(np.array(self.bundle.frames[episode.shard][episode.start + offset]).data)
            for offset in range(13, 20)
        }
        drawn = {
            bytes(dataset[0]["goal_image"].permute(1, 2, 0).contiguous().numpy().data)
            for _ in range(40)
        }
        self.assertTrue(drawn <= reachable)
        self.assertGreater(len(drawn), 1)

    def test_the_uniform_rule_is_the_one_the_video_model_generates_under(self) -> None:
        """src/diffusion generates frames 1-3 of four; the goal is its frame 3.

        These two expressions have to agree exactly or the policy is trained on
        one frame of the episode and handed another at inference.
        """
        from diffusion.data import four_frame_indices

        for last in range(3, 400):
            self.assertEqual(tuple(uniform_frame_offsets(last, 4)), four_frame_indices(last + 1, last))

    def test_the_goal_selection_rule_must_be_one_of_the_two(self) -> None:
        with self.assertRaises(ValueError):
            self._dataset(goal_selection="whatever")

    def test_the_goal_frame_index_must_be_inside_the_sampled_frames(self) -> None:
        with self.assertRaises(ValueError):
            self._dataset(goal_frames=4, goal_frame_index=4)

    def test_the_goal_is_redrawn_between_epochs_but_pinned_for_validation(self) -> None:
        random = self._dataset()
        pinned = self._dataset(random_goal=False)
        as_bytes = lambda sample: bytes(sample["goal_image"].numpy().data)  # noqa: E731
        self.assertEqual(len({as_bytes(pinned[0]) for _ in range(8)}), 1)
        self.assertGreater(len({as_bytes(random[0]) for _ in range(20)}), 1)

    def test_the_goal_is_cropped_with_the_observation_not_separately(self) -> None:
        """A separate crop would shift the goal relative to what the policy sees.

        The camera is static, so every frame of one sample has to share a crop
        box. Photometric jitter is disabled here because it is redrawn per frame
        by design, which would mask the crop the test is actually checking.
        """
        episode = self.bundle.episodes[0]
        # Make the whole episode one repeated image, so any surviving difference
        # between the observation and the goal can only come from the crop. The
        # shard is a read-only memmap, so swap in a writable copy.
        frames = np.array(self.bundle.frames[episode.shard])
        frames[episode.start : episode.end] = frames[episode.start]
        self.bundle.frames[episode.shard] = frames

        dataset = self._dataset(
            random_goal=False, augment=True, crop_scale=0.6, color_jitter=0.0
        )
        for _ in range(10):
            sample = dataset[0]
            np.testing.assert_array_equal(
                sample["goal_image"].numpy(), sample["image"][-1].numpy()
            )

    def test_the_window_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            self._dataset(goal_window=0)


class GoalPolicyTests(unittest.TestCase):
    def _batch(self, config: PolicyConfig, size: int = 2, goal: bool = True) -> dict:
        generator = torch.Generator().manual_seed(0)
        shape = (size, config.n_obs_steps, 3, config.image_size, config.image_size)
        batch = {
            "image": torch.randint(0, 256, shape, dtype=torch.uint8, generator=generator),
            "state": torch.randn(size, config.n_obs_steps, config.state_dim, generator=generator),
            "action": torch.randn(size, config.horizon, config.action_dim, generator=generator),
            "action_is_pad": torch.zeros(size, config.horizon, dtype=torch.bool),
        }
        if goal:
            batch["goal_image"] = torch.randint(
                0, 256, (size, 3, config.image_size, config.image_size),
                dtype=torch.uint8, generator=generator,
            )
        return batch

    def test_the_conditioning_vector_grows_by_exactly_one_image_feature(self) -> None:
        plain = ChunkPolicy(tiny_config(goal_conditioned=False))
        goal = GoalChunkPolicy(tiny_config())
        self.assertEqual(
            goal.global_cond_dim, plain.global_cond_dim + plain.rgb_encoder.feature_dim
        )

    def test_the_goal_encoder_is_shared_with_the_observation_encoder(self) -> None:
        # A second ResNet would double the vision parameters to learn the same
        # thing twice from the same frames.
        policy = GoalChunkPolicy(tiny_config())
        encoders = {name.split(".")[0] for name, _ in policy.named_parameters()}
        self.assertIn("rgb_encoder", encoders)
        self.assertNotIn("goal_encoder", encoders)

    def test_loss_and_prediction_have_the_expected_shapes(self) -> None:
        config = tiny_config()
        policy = GoalChunkPolicy(config)
        batch = self._batch(config)

        loss, metrics = policy.compute_loss(batch)
        self.assertEqual(loss.shape, ())
        self.assertIn("action_loss", metrics)
        loss.backward()
        # The goal reaches the loss through the shared encoder, so its gradient
        # is the encoder's. null_goal gets none here and should not: with
        # goal_dropout=0 it is never used.
        self.assertIsNotNone(policy.rgb_encoder.project.weight.grad)
        self.assertIsNone(policy.null_goal.grad)

        prediction = policy.predict(batch)
        self.assertEqual(prediction.shape, (2, config.horizon, config.action_dim))

    def test_the_goal_changes_the_predicted_chunk(self) -> None:
        """A goal the model ignores is worse than no goal at all - it costs a
        ResNet pass to change nothing. This is the untrained version of that
        check: the goal has to reach the sampler."""
        config = tiny_config()
        policy = GoalChunkPolicy(config).eval()
        batch = self._batch(config)

        torch.manual_seed(0)
        first = policy.predict(batch)
        other = dict(batch, goal_image=torch.zeros_like(batch["goal_image"]))
        torch.manual_seed(0)
        second = policy.predict(other)
        self.assertFalse(torch.allclose(first, second))

    def test_a_missing_goal_is_refused_unless_the_null_branch_was_trained(self) -> None:
        config = tiny_config()
        policy = GoalChunkPolicy(config)
        with self.assertRaises(ValueError) as error:
            policy.compute_loss(self._batch(config, goal=False))
        self.assertIn("goal_dropout", str(error.exception))

    def test_a_dropout_trained_policy_falls_back_to_the_null_embedding(self) -> None:
        config = tiny_config(goal_dropout=0.1)
        policy = GoalChunkPolicy(config).eval()
        prediction = policy.predict(self._batch(config, goal=False))
        self.assertEqual(prediction.shape, (2, config.horizon, config.action_dim))

    def test_dropping_the_observation_drops_the_goal_with_it(self) -> None:
        """cond_dropout has to blank the goal too, or the 'unconditional' branch
        classifier-free guidance extrapolates away from would still know where
        the episode ends."""
        config = tiny_config(cond_dropout=0.5)
        policy = GoalChunkPolicy(config).train()
        batch = self._batch(config, size=16)
        elsewhere = dict(batch, goal_image=torch.zeros_like(batch["goal_image"]))

        # Same seed, so the same rows are dropped in both passes.
        torch.manual_seed(0)
        first = policy.encode_context(batch)
        torch.manual_seed(0)
        second = policy.encode_context(elsewhere)

        null = policy.null_cond.expand(first.shape[0], -1)
        dropped = torch.isclose(first, null).all(dim=-1)
        self.assertTrue(dropped.any() and not dropped.all(), "expected a mix of dropped rows")
        torch.testing.assert_close(first[dropped], second[dropped])
        self.assertFalse(torch.allclose(first[~dropped], second[~dropped]))

    def test_goal_dropout_only_applies_while_training(self) -> None:
        config = tiny_config(goal_dropout=0.5)
        policy = GoalChunkPolicy(config)
        batch = self._batch(config, size=16)
        feature_dim = policy.rgb_encoder.feature_dim

        def null_goal_rows(context: torch.Tensor) -> int:
            null = policy.null_goal.expand(context.shape[0], -1)
            return int(torch.isclose(context[:, -feature_dim:], null).all(dim=-1).sum())

        policy.train()
        torch.manual_seed(0)
        self.assertGreater(null_goal_rows(policy.encode_context(batch)), 0)
        policy.eval()
        self.assertEqual(null_goal_rows(policy.encode_context(batch)), 0)

    def test_a_plain_config_cannot_build_a_goal_policy(self) -> None:
        with self.assertRaises(ValueError):
            GoalChunkPolicy(tiny_config(goal_conditioned=False))

    def test_goal_dropout_without_goal_conditioning_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            tiny_config(goal_conditioned=False, goal_dropout=0.1)


class GoalTrainingTests(unittest.TestCase):
    """Train a goal-conditioned run end to end, then drive it like the robot loop."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        root = Path(cls.temporary.name)
        cls.data_root = root / "data"
        write_bundle(cls.data_root, "tiny", {"tiny": [16, 16, 16]}, image_size=64)
        cls.output_root = root / "runs"
        main([
            "--bundle", "tiny", "--data-root", str(cls.data_root),
            "--output-root", str(cls.output_root), "--run-name", "goal",
            "--goal-conditioned", "--goal-window", "4", "--goal-dropout", "0.2",
            "--steps", "4", "--batch-size", "2", "--num-workers", "0",
            "--val-episodes", "1", "--eval-freq", "4", "--eval-batches", "1",
            "--eval-samples", "2", "--save-freq", "0", "--log-freq", "4",
            "--num-inference-steps", "2", "--device", "cpu", "--amp", "off",
            *TINY,
        ])
        cls.run_dir = cls.output_root / "goal"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_the_run_records_how_it_was_conditioned(self) -> None:
        metadata = json.loads((self.run_dir / "run.json").read_text(encoding="utf-8"))
        self.assertTrue(metadata["config"]["goal_conditioned"])
        self.assertEqual(metadata["config"]["goal_window"], 4)
        self.assertEqual(metadata["config"]["goal_dropout"], 0.2)

    def test_validation_ran_with_a_goal(self) -> None:
        records = [
            json.loads(line)
            for line in (self.run_dir / "log.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertTrue([record for record in records if "val_mae" in record])

    def test_the_checkpoint_is_recognisable_without_loading_the_weights(self) -> None:
        self.assertTrue(is_goal_conditioned(self.run_dir / "final.pt"))

    def test_the_plain_runner_refuses_it_with_an_explanation(self) -> None:
        with self.assertRaises(ValueError) as error:
            PolicyRunner(self.run_dir / "final.pt", device="cpu")
        self.assertIn("goal", str(error.exception).lower())

    def test_load_runner_picks_the_goal_runner(self) -> None:
        runner = load_runner(self.run_dir / "final.pt", device="cpu")
        self.assertIsInstance(runner, GoalPolicyRunner)
        self.assertIsInstance(runner.policy, GoalChunkPolicy)

    def _runner(self, **kwargs) -> GoalPolicyRunner:
        return GoalPolicyRunner(self.run_dir / "final.pt", device="cpu", **kwargs)

    def test_the_control_loop_runs_once_a_goal_is_set(self) -> None:
        runner = self._runner()
        runner.set_goal(np.zeros((48, 64, 3), dtype=np.uint8))
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        state = np.zeros(7, dtype=np.float32)
        for _ in range(6):
            self.assertEqual(runner.select_action(frame, state).shape, (7,))

    def test_the_goal_survives_a_reset(self) -> None:
        # reset() ends an episode; the goal describes the task, not the episode.
        runner = self._runner(goal=np.zeros((48, 64, 3), dtype=np.uint8))
        runner.select_action(np.zeros((48, 64, 3), np.uint8), np.zeros(7, np.float32))
        runner.reset()
        self.assertTrue(runner.has_goal)
        self.assertEqual(runner.queued_actions, 0)

    def test_changing_the_goal_drops_the_queued_chunk(self) -> None:
        # Actions planned for the old goal must not reach the arm afterwards.
        runner = self._runner(goal=np.zeros((48, 64, 3), dtype=np.uint8))
        runner.select_action(np.zeros((48, 64, 3), np.uint8), np.zeros(7, np.float32))
        self.assertGreater(runner.queued_actions, 0)
        runner.set_goal(np.full((48, 64, 3), 255, dtype=np.uint8))
        self.assertEqual(runner.queued_actions, 0)

    def test_warmup_needs_no_goal_and_leaves_none_behind(self) -> None:
        runner = self._runner()
        runner.warmup(frame_shape=(48, 64))
        self.assertFalse(runner.has_goal)
        self.assertEqual(runner.queued_actions, 0)

    def test_this_checkpoint_can_also_run_with_no_goal(self) -> None:
        # It was trained with --goal-dropout 0.2, so the null branch is real.
        runner = self._runner()
        action = runner.select_action(np.zeros((48, 64, 3), np.uint8), np.zeros(7, np.float32))
        self.assertEqual(action.shape, (7,))

    def test_describe_reports_the_goal_settings(self) -> None:
        runner = self._runner()
        description = runner.describe()
        self.assertTrue(description["goal_conditioned"])
        self.assertEqual(description["goal_window"], 4)
        self.assertFalse(description["has_goal"])

    def test_offline_replay_supplies_the_goal_from_the_episode(self) -> None:
        from policy.infer import main as infer_main

        infer_main([
            "--checkpoint", str(self.run_dir / "final.pt"), "--device", "cpu",
            "--bundle", "tiny", "--data-root", str(self.data_root),
            "--episode", "0", "--max-steps", "4",
        ])

    def test_a_goal_free_run_is_scored_the_same_way(self) -> None:
        from policy.infer import main as infer_main

        infer_main([
            "--checkpoint", str(self.run_dir / "final.pt"), "--device", "cpu",
            "--bundle", "tiny", "--data-root", str(self.data_root),
            "--episode", "0", "--max-steps", "4", "--no-goal",
        ])


class GoalTrainingWithoutDropoutTests(unittest.TestCase):
    """A checkpoint trained with goal_dropout=0 must not be run without a goal."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        root = Path(cls.temporary.name)
        write_bundle(root / "data", "tiny", {"tiny": [16, 16]}, image_size=64)
        main([
            "--bundle", "tiny", "--data-root", str(root / "data"),
            "--output-root", str(root / "runs"), "--run-name", "strict",
            "--goal-conditioned", "--steps", "2", "--batch-size", "2",
            "--num-workers", "0", "--val-episodes", "0", "--eval-freq", "0",
            "--save-freq", "0", "--log-freq", "2", "--num-inference-steps", "2",
            "--device", "cpu", "--amp", "off", *TINY,
        ])
        cls.checkpoint = root / "runs" / "strict" / "final.pt"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_stepping_without_a_goal_fails_before_it_commands_anything(self) -> None:
        runner = GoalPolicyRunner(self.checkpoint, device="cpu")
        with self.assertRaises(RuntimeError) as error:
            runner.select_action(np.zeros((48, 64, 3), np.uint8), np.zeros(7, np.float32))
        self.assertIn("set_goal", str(error.exception))


class GoalArgumentTests(unittest.TestCase):
    def test_goal_conditioning_is_off_by_default(self) -> None:
        args = parse_args(["--bundle", "tiny"])
        self.assertFalse(args.goal_conditioned)
        self.assertEqual(args.goal_window, 10)
        self.assertEqual(args.goal_dropout, 0.0)

    def test_the_default_goal_is_the_third_of_four_sampled_frames(self) -> None:
        args = parse_args(["--bundle", "tiny", "--goal-conditioned"])
        self.assertEqual(args.goal_selection, "uniform4")
        self.assertEqual((args.goal_frames, args.goal_frame_index), (4, 2))

    def test_goal_flags_without_goal_conditioning_are_flagged(self) -> None:
        from policy.train import warn_about_ignored_flags

        args = parse_args(["--bundle", "tiny", "--goal-window", "5"])
        self.assertIn("--goal-window", warn_about_ignored_flags(args))

        args = parse_args(["--bundle", "tiny", "--goal-conditioned", "--goal-window", "5"])
        self.assertEqual(warn_about_ignored_flags(args), [])


if __name__ == "__main__":
    unittest.main()
