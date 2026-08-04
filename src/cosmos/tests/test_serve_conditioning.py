"""Regression tests for how the HTTP server resolves conditioning FPS.

The robot path reaches the model through ``serve.py``, so a rate that drifts
from the one the adapter was trained at collapses mode coverage there too --
silently, because nothing in the response used to say which rate was used.
"""

from __future__ import annotations

import unittest

from cosmos.serve import GenerationService


class StubRunner:
    """Just the attributes GenerationService reads off a CosmosRunner."""

    def __init__(self, fps: float = 12.78, training_fps: float | None = 12.78):
        self.fps = fps
        self.training_fps = training_fps
        self.num_frames = 93
        self.resolution = 256
        self.guidance_scale = 1.0


class ResolveFpsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = GenerationService(StubRunner())

    def test_a_request_without_fps_uses_the_trained_rate(self) -> None:
        self.assertAlmostEqual(self.service.resolve_fps({}), 12.78)

    def test_an_explicit_null_is_not_treated_as_a_rate(self) -> None:
        # client.py strips None, but a hand-rolled client can send {"fps": null}
        # and float(None) would otherwise raise a 500.
        self.assertAlmostEqual(self.service.resolve_fps({"fps": None}), 12.78)

    def test_an_explicit_rate_is_honoured(self) -> None:
        self.assertAlmostEqual(self.service.resolve_fps({"fps": 20.0}), 20.0)

    def test_a_drifting_rate_is_warned_about_once(self) -> None:
        with self.assertLogs("cosmos3-serve", level="WARNING") as captured:
            self.service.resolve_fps({"fps": 20.0})
        self.assertIn("trained at 12.78", captured.output[0])

        # The robot polls this server every rollout; one warning per distinct
        # rate is enough to notice, and repeating it would drown the log.
        with self.assertNoLogs("cosmos3-serve", level="WARNING"):
            self.service.resolve_fps({"fps": 20.0})

    def test_a_rate_matching_training_is_not_warned_about(self) -> None:
        with self.assertNoLogs("cosmos3-serve", level="WARNING"):
            self.service.resolve_fps({"fps": 12.8})

    def test_a_nonpositive_rate_is_a_client_error(self) -> None:
        with self.assertRaises(ValueError):
            self.service.resolve_fps({"fps": 0})

    def test_an_uninspectable_run_skips_the_comparison(self) -> None:
        service = GenerationService(StubRunner(fps=20.0, training_fps=None))
        with self.assertNoLogs("cosmos3-serve", level="WARNING"):
            self.assertAlmostEqual(service.resolve_fps({"fps": 20.0}), 20.0)


if __name__ == "__main__":
    unittest.main()
