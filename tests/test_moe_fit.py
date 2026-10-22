#!/usr/bin/env python3
"""Unit tests for scripts/moe_fit.py (placement policy and GGUF size math).

Run: python3 -I tests/test_moe_fit.py
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))

import moe_fit  # noqa: E402

GIB = moe_fit.GIB


class TensorBytes(unittest.TestCase):
    def test_dense_types(self):
        self.assertEqual(moe_fit.tensor_bytes([2880, 2880], 0), 2880 * 2880 * 4)
        self.assertEqual(moe_fit.tensor_bytes([2880, 2880], 1), 2880 * 2880 * 2)

    def test_quantised_types_round_like_ggml(self):
        # Q8_0: 32 values in 34 bytes
        self.assertEqual(moe_fit.tensor_bytes([32, 10], 8), 10 * 34)
        # MXFP4: 32 values in 17 bytes (1 exponent + 16 nibble bytes)
        self.assertEqual(moe_fit.tensor_bytes([32, 10], 39), 10 * 17)
        self.assertEqual(moe_fit.tensor_bytes([2880, 2880], 39), 2880 * 2880 // 32 * 17)

    def test_unknown_type_is_refused(self):
        with self.assertRaises(SystemExit):
            moe_fit.tensor_bytes([32], 999)


class PickPlacement(unittest.TestCase):
    """A synthetic 4-layer model with 1.5 GiB of experts per layer."""

    def setUp(self):
        self.expert = {i: int(1.5 * GIB) for i in range(4)}
        self.n_layer = 4
        self.nonexpert = 5 * GIB
        self.kv = 1 * GIB
        self.buffers = 0.5 * GIB

    def pick(self, budget, avail_host=100 * GIB, host_margin=0.5 * GIB):
        return moe_fit.pick_placement(
            self.expert, self.n_layer, self.nonexpert, self.kv, self.buffers,
            budget, avail_host, host_margin)

    def test_vram_decides_when_host_has_room(self):
        # ncmoe=2 -> 5 + 3.0 + 1 + 0.5 = 9.5 <= 10; ncmoe=1 -> 11 > 10
        ncmoe, vram_need, host_need, host_ok, notes = self.pick(10 * GIB)
        self.assertEqual(ncmoe, 2)
        self.assertAlmostEqual(vram_need / GIB, 9.5, places=6)
        self.assertAlmostEqual(host_need / GIB, 3.5, places=6)
        self.assertTrue(host_ok)
        self.assertEqual(notes, [])

    def test_everything_fits_on_gpu(self):
        ncmoe, vram_need, _, _, _ = self.pick(100 * GIB)
        self.assertEqual(ncmoe, 0)
        self.assertAlmostEqual(vram_need / GIB, 12.5, places=6)

    def test_smallest_vram_fit_is_chosen(self):
        # 9.0 GiB cannot hold ncmoe=2 (9.5), so one more CPU layer is required
        ncmoe, vram_need, _, _, _ = self.pick(9.0 * GIB)
        self.assertEqual(ncmoe, 3)
        self.assertAlmostEqual(vram_need / GIB, 8.0, places=6)

    def test_host_pressure_is_reported_not_hidden(self):
        # ncmoe=2 keeps 3.0 GiB of experts on the CPU (+0.5 margin) and only
        # 1.0 GiB is available. Lowering ncmoe would exceed VRAM by
        # construction, so the planner must refuse to claim a fix.
        ncmoe, _, host_need, host_ok, notes = self.pick(10 * GIB, avail_host=1 * GIB)
        self.assertEqual(ncmoe, 2)
        self.assertAlmostEqual(host_need / GIB, 3.5, places=6)
        self.assertFalse(host_ok)
        self.assertEqual(len(notes), 1)
        self.assertIn("host RAM too small", notes[0])
        self.assertIn("cannot go lower", notes[0])

    def test_infeasible_configuration_raises(self):
        with self.assertRaises(SystemExit):
            self.pick(4 * GIB)

    def test_ncmoe_never_exceeds_layer_count(self):
        # every expert on the CPU needs 5 + 1 + 0.5 = 6.5 GiB
        ncmoe, _, _, _, _ = self.pick(7 * GIB)
        self.assertEqual(ncmoe, self.n_layer)


class GptOssGeometry(unittest.TestCase):
    """Guard the numbers quoted in docs/PROBE-REPORT.md (needs the real GGUF)."""

    MODEL = os.path.join(HERE, "..", "models", "gpt-oss-120b-MXFP4.gguf")

    @unittest.skipUnless(os.path.exists(MODEL), "gpt-oss-120b GGUF not downloaded")
    def test_expert_bytes_per_token(self):
        meta, tensors = moe_fit.read_gguf(self.MODEL)
        expert = {}
        for name, dims, ttype in tensors:
            if name.startswith("blk.") and "_exps." in name:
                li = int(name.split(".")[1])
                expert[li] = expert.get(li, 0) + moe_fit.tensor_bytes(dims, ttype)
        per_layer = expert[0] / moe_fit.MIB
        self.assertAlmostEqual(per_layer, 1617.9, delta=1.0)
        # top-4 of 128 experts, 8 CPU layers: the figure in the report
        per_token = 4 / 128 * 8 * expert[0] / moe_fit.MIB
        self.assertAlmostEqual(per_token, 404.5, delta=1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
