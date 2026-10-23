# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""GPU integration checks: uv run -m unittest scripts.tests.test_sand_cuda_graphs."""

import unittest

import numpy as np
import warp as wp

import newton.viewer
from newton.examples.mpm.example_mpm_bottle_drop import Example as BottleExample
from newton.examples.mpm.example_mpm_sand_bucket import Example as PourExample


@unittest.skipUnless(wp.is_cuda_available(), "CUDA graph integration requires a GPU")
class TestSandCudaGraphs(unittest.TestCase):
    def test_bottle_phase_capture(self):
        """Match eager motion through all four captured phases with a contained deep bed."""
        results = []
        with wp.ScopedDevice("cuda:0"):
            for eager in (True, False):
                args = BottleExample.create_parser().parse_args([])
                args.no_cuda_graph = eager
                # A smaller resolved bed and shorter waits cover every phase without
                # repeating the full six-case experiment in this regression test.
                args.bed_width = args.bed_length = 0.4
                args.voxel_size = args.cap_radius = 0.02
                args.pre_settle_seconds = 0.05
                args.impact_settle_seconds = 1.0
                args.lift_speed = 1.0
                args.substeps = 8
                example = BottleExample(newton.viewer.ViewerNull(), args)
                for _ in range(180):
                    example.step()
                example.test_final()
                self.assertEqual(example.phase, "settling crater")
                self.assertEqual(len(example.graphs), 0 if eager else 4)
                self.assertEqual(example.model.particle_count, 48000)
                results.append((example.state_0.particle_q.numpy(), example.state_0.body_q.numpy()))
                del example
        for eager, captured in zip(*results, strict=True):
            # Atomic contact reductions accumulate small trajectory differences;
            # bound these below 5% of the 2 cm voxel (1 mm for positions).
            np.testing.assert_allclose(captured, eager, atol=0.001, rtol=0)

    def test_pour_capture_state_swaps(self):
        """Match eager pour motion across alternating input and output state captures."""
        results = []
        with wp.ScopedDevice("cuda:0"):
            for eager in (True, False):
                args = PourExample.create_parser().parse_args([])
                args.no_cuda_graph = eager
                example = PourExample(newton.viewer.ViewerNull(), args)
                for _ in range(30):
                    example.step()
                self.assertEqual(len(example.graphs), 0 if eager else 2)
                results.append(example.state_0.particle_q.numpy())
                del example
        np.testing.assert_allclose(results[1], results[0], atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
