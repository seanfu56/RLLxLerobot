"""Regression tests for the corrected source-balanced training process."""

from __future__ import annotations

import unittest
from collections import Counter
from pathlib import Path

from cosmos.data import Episode, PiperClipDataset, SourceBalancedSampler, split_episodes
from cosmos.train import LORA_TARGET_MODULES, parse_args


def make_episode(source: str, index: int, *, frames: int = 121) -> Episode:
    return Episode(
        path=Path(f"/unused/{source}/episode{index:03d}/video.mp4"),
        source=source,
        name=f"episode{index:03d}",
        task="pick up the can",
        frames=frames,
        fps=20.0,
    )


class StratifiedSplitTests(unittest.TestCase):
    def test_equal_sources_receive_equal_validation_quotas(self) -> None:
        episodes = [make_episode("m1", i) for i in range(30)] + [
            make_episode("m2", i) for i in range(30)
        ]

        train, validation = split_episodes(episodes, val_episodes=6, seed=42)

        self.assertEqual(Counter(ep.source for ep in train), {"m1": 27, "m2": 27})
        self.assertEqual(
            Counter(ep.source for ep in validation), {"m1": 3, "m2": 3}
        )
        train_again, validation_again = split_episodes(
            episodes, val_episodes=6, seed=42
        )
        self.assertEqual(train, train_again)
        self.assertEqual(validation, validation_again)

    def test_single_episode_source_is_kept_for_training(self) -> None:
        episodes = [make_episode("rare", 0)] + [
            make_episode("common", i) for i in range(5)
        ]

        train, validation = split_episodes(episodes, val_episodes=2, seed=0)

        self.assertEqual(Counter(ep.source for ep in train), {"common": 3, "rare": 1})
        self.assertEqual(Counter(ep.source for ep in validation), {"common": 2})


class SourceBalancedSamplerTests(unittest.TestCase):
    def test_smaller_source_is_resampled_to_equalize_training(self) -> None:
        episodes = [make_episode("m1", i) for i in range(2)] + [
            make_episode("m2", i) for i in range(3)
        ]
        sampler = SourceBalancedSampler(episodes, seed=7)

        indexes = list(sampler)

        self.assertEqual(len(indexes), 6)
        self.assertEqual(
            Counter(episodes[index].source for index in indexes), {"m1": 3, "m2": 3}
        )
        self.assertEqual(indexes, list(SourceBalancedSampler(episodes, seed=7)))


class CorrectedTrainingDefaultsTests(unittest.TestCase):
    def test_episode_mode_starts_at_true_first_frame_and_covers_motion(self) -> None:
        episode = make_episode("m1", 0)
        dataset = PiperClipDataset(
            [episode], {episode.task: episode.task}, clip_mode="episode"
        )

        indices, output_fps = dataset.clip_plan(0)

        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 120)
        self.assertAlmostEqual(output_fps, 93 * 20.0 / 121)

    def test_training_defaults_to_episode_mode(self) -> None:
        args = parse_args(["--bundle", "/unused/pick-can-all"])

        self.assertEqual(args.clip_mode, "episode")

    def test_lora_targets_only_the_broader_generation_pathway(self) -> None:
        expected = {
            "add_q_proj",
            "add_k_proj",
            "add_v_proj",
            "to_add_out",
            "mlp_moe_gen.gate_proj",
            "mlp_moe_gen.up_proj",
            "mlp_moe_gen.down_proj",
            "proj_in",
            "proj_out",
        }

        self.assertEqual(set(LORA_TARGET_MODULES), expected)
        self.assertNotIn("to_q", LORA_TARGET_MODULES)
        self.assertNotIn("gate_proj", LORA_TARGET_MODULES)


if __name__ == "__main__":
    unittest.main()
