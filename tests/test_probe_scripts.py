#!/usr/bin/env python3
"""Unit tests for the Python probe scripts (no model, no GPU, no server).

Run: python3 -m unittest tests.test_probe_scripts
     (or: python3 -I -m unittest discover -s tests -p "test_probe_scripts.py")

Note: on Python >= 3.11 `-I` implies `-P`, which removes the current directory
from sys.path; the dotted name `tests.test_probe_scripts` then cannot be
imported under `-I`, so use the `discover -s tests` form there. The test logic
itself is stdlib-only and runs identically under isolation.

Covers:
  - scripts/compare_mmid.py: parsing, positional comparison, rtol/atol and
    the header/length mismatch paths (both process exit codes).
  - scripts/quality_gate.py: needle prompt construction (depth split, filler
    count, needle placement) and the argparse defaults.
  - scripts/routing_skew.py: period detection plus topk_coverage/lru_hit on
    synthetic traces with hand-computed answers.
"""
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(HERE, "..", "scripts")
sys.path.insert(0, SCRIPTS)

import compare_mmid  # noqa: E402
import quality_gate  # noqa: E402
import routing_skew  # noqa: E402


def write_temp(text, suffix=".txt"):
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "w") as f:
        f.write(text)
    return path


class MmidFiles(unittest.TestCase):
    """Temp-file helpers shared by the compare_mmid tests."""

    def make(self, text):
        path = write_temp(text)
        self.addCleanup(os.unlink, path)
        return path

    def run_cli(self, *args):
        return subprocess.run(
            [sys.executable, "-I", os.path.join(SCRIPTS, "compare_mmid.py"), *args],
            capture_output=True, text=True)


class LoadTest(MmidFiles):
    def test_header_and_floats(self):
        path = self.make("n_tokens 3\n1.0\n2.5\n\n3.25\n")
        header, vals = compare_mmid.load(path)
        self.assertEqual(header, "n_tokens 3")
        self.assertEqual(vals, [1.0, 2.5, 3.25])

    def test_blank_lines_are_skipped(self):
        path = self.make("h\n\n1\n   \n-2.5e-3\n")
        header, vals = compare_mmid.load(path)
        self.assertEqual(header, "h")
        self.assertEqual(vals, [1.0, -0.0025])


class CompareValuesTest(unittest.TestCase):
    def test_identical(self):
        self.assertEqual(compare_mmid.compare_values([1.0, -2.0], [1.0, -2.0])[3], [])

    def test_positional_order_matters(self):
        max_abs, _, _, bad = compare_mmid.compare_values([1.0, 2.0], [2.0, 1.0])
        self.assertAlmostEqual(max_abs, 1.0)
        self.assertEqual(len(bad), 2)
        self.assertEqual([i for i, _, _ in bad], [0, 1])

    def test_atol_accepts_tiny_absolute_difference(self):
        # |x-y| = 5e-6 < atol (1e-5) even though the relative error is 1.
        self.assertEqual(compare_mmid.compare_values([0.0], [5e-6])[3], [])

    def test_atol_rejects_larger_absolute_difference(self):
        _, _, _, bad = compare_mmid.compare_values([0.0], [1e-4])
        self.assertEqual([i for i, _, _ in bad], [0])

    def test_rtol_accepts_relative_difference(self):
        # |100 - 100.001| = 1e-3 < 1e-5 + 5e-5 * 100.001 = 5.01e-3.
        self.assertEqual(
            compare_mmid.compare_values([100.0], [100.001], rtol=5e-5, atol=1e-5)[3], [])

    def test_rtol_rejects_larger_relative_difference(self):
        # |100 - 100.01| = 1e-2 > 5.01e-3.
        _, _, _, bad = compare_mmid.compare_values([100.0], [100.01], rtol=5e-5, atol=1e-5)
        self.assertEqual(len(bad), 1)

    def test_zero_tolerance_requires_exact_match(self):
        self.assertEqual(
            compare_mmid.compare_values([1.5], [1.5], rtol=0.0, atol=0.0)[3], [])
        _, _, _, bad = compare_mmid.compare_values([1.5], [1.5000001], rtol=0.0, atol=0.0)
        self.assertEqual(len(bad), 1)

    def test_reports_worst_index_and_errors(self):
        max_abs, max_rel, worst, _ = compare_mmid.compare_values(
            [1.0, 10.0, 2.0], [1.0, 10.5, 2.0])
        self.assertAlmostEqual(max_abs, 0.5)
        self.assertEqual(worst, 1)
        self.assertGreater(max_rel, 0.0)


class CompareFilesTest(MmidFiles):
    def test_equal_files_exit_zero(self):
        a = self.make("n_tokens 2\n1.0\n2.0\n")
        b = self.make("n_tokens 2\n1.0\n2.0\n")
        self.assertEqual(compare_mmid.compare_files(a, b), 0)

    def test_within_tolerance_exit_zero(self):
        a = self.make("n\n1.0\n2.0\n")
        b = self.make("n\n1.0\n2.000001\n")
        self.assertEqual(compare_mmid.compare_files(a, b), 0)

    def test_outside_tolerance_exit_one(self):
        a = self.make("n\n1.0\n2.0\n")
        b = self.make("n\n1.0\n2.1\n")
        self.assertEqual(compare_mmid.compare_files(a, b), 1)

    def test_header_mismatch_exit_one(self):
        a = self.make("n_tokens 2\n1.0\n2.0\n")
        b = self.make("n_tokens 3\n1.0\n2.0\n")
        self.assertEqual(compare_mmid.compare_files(a, b), 1)

    def test_length_mismatch_exit_one(self):
        a = self.make("n\n1.0\n2.0\n")
        b = self.make("n\n1.0\n")
        self.assertEqual(compare_mmid.compare_files(a, b), 1)

    def test_cli_exit_codes(self):
        good = self.make("n\n1.0\n2.0\n")
        same = self.make("n\n1.0\n2.0\n")
        bad = self.make("n\n1.0\n2.5\n")
        self.assertEqual(self.run_cli(good, same).returncode, 0)
        self.assertEqual(self.run_cli(good, bad).returncode, 1)

    def test_cli_header_mismatch_exit_code(self):
        a = self.make("n_tokens 2\n1.0\n2.0\n")
        b = self.make("n_tokens 3\n1.0\n2.0\n")
        p = self.run_cli(a, b)
        self.assertEqual(p.returncode, 1)
        self.assertIn("header mismatch", p.stdout)

    def test_cli_length_mismatch_exit_code(self):
        a = self.make("n\n1.0\n2.0\n")
        b = self.make("n\n1.0\n")
        p = self.run_cli(a, b)
        self.assertEqual(p.returncode, 1)
        self.assertIn("length mismatch", p.stdout)


class NeedlePromptTest(unittest.TestCase):
    def test_middle_depth_splits_filler_evenly(self):
        prompt = quality_gate.build_needle_prompt(10, 0.5)
        half = 5
        expected = (quality_gate.NEEDLE_PREFIX + quality_gate.FILLER * half
                    + quality_gate.NEEDLE + quality_gate.FILLER * (10 - half)
                    + quality_gate.NEEDLE_QUESTION)
        self.assertEqual(prompt, expected)
        self.assertEqual(prompt.count(quality_gate.FILLER), 10)

    def test_zero_depth_puts_needle_first(self):
        prompt = quality_gate.build_needle_prompt(4, 0.0)
        self.assertTrue(prompt.startswith(
            quality_gate.NEEDLE_PREFIX + quality_gate.NEEDLE))
        self.assertEqual(prompt.count(quality_gate.FILLER), 4)

    def test_depth_one_puts_needle_last(self):
        prompt = quality_gate.build_needle_prompt(4, 1.0)
        self.assertTrue(prompt.startswith(
            quality_gate.NEEDLE_PREFIX + quality_gate.FILLER * 4
            + quality_gate.NEEDLE))

    def test_depth_split_truncates_like_int(self):
        # int(7 * 0.5) == 3 paragraphs before the needle, 4 after.
        prompt = quality_gate.build_needle_prompt(7, 0.5)
        half = int(7 * 0.5)
        self.assertEqual(half, 3)
        self.assertTrue(prompt.startswith(
            quality_gate.NEEDLE_PREFIX + quality_gate.FILLER * half
            + quality_gate.NEEDLE))
        self.assertEqual(prompt.count(quality_gate.FILLER), 7)

    def test_needle_content_and_question(self):
        prompt = quality_gate.build_needle_prompt(3, 0.35)
        for text in ("Okafor", "4471", "14 March 1997", "QUESTION:"):
            self.assertIn(text, prompt)

    def test_argument_defaults(self):
        args = quality_gate.parse_args([])
        self.assertEqual(args.filler_reps, 340)
        self.assertEqual(args.needle_depth, 0.0)
        self.assertEqual(args.prompts, "route,code,math,needle")
        self.assertEqual(args.ctx, 32768)
        self.assertEqual(args.ngl, 99)
        self.assertEqual(args.ncmoe, 9)
        self.assertEqual(args.kv, "f16")
        self.assertEqual(args.n_predict, 96)
        self.assertEqual(args.env, [])
        self.assertEqual(args.name, "run")
        self.assertEqual(args.out, "results/quality")
        self.assertEqual(args.extra, [])

    def test_defaults_build_full_filler(self):
        args = quality_gate.parse_args([])
        prompt = quality_gate.build_needle_prompt(args.filler_reps, args.needle_depth)
        self.assertEqual(prompt.count(quality_gate.FILLER), 340)
        self.assertTrue(prompt.startswith(
            quality_gate.NEEDLE_PREFIX + quality_gate.NEEDLE))


def synthetic_trace(period, tokens, start=0):
    """One op per layer per token; layers use disjoint sets of expert ids."""
    ops = []
    for _ in range(tokens):
        for layer in range(period):
            ops.append([start + layer * 10 + j for j in range(4)])
    return ops


class PeriodicTraceTest(unittest.TestCase):
    def test_detects_period_eight(self):
        ops = synthetic_trace(8, 20)
        self.assertEqual(len(ops), 160)
        self.assertEqual(routing_skew.detect_period(ops), 8)

    def test_detects_period_six(self):
        ops = synthetic_trace(6, 20)
        self.assertEqual(len(ops), 120)
        self.assertEqual(routing_skew.detect_period(ops), 6)

    def test_candidate_list_is_honoured(self):
        ops = synthetic_trace(8, 20)
        self.assertEqual(routing_skew.detect_period(ops, candidates=(4, 8)), 8)

    def test_short_trace_falls_back_to_eight(self):
        self.assertEqual(routing_skew.detect_period(synthetic_trace(4, 2)), 8)

    def test_split_layers_groups_by_position(self):
        ops = synthetic_trace(4, 2)  # 2 tokens, layers 0..3
        layers = routing_skew.split_layers(ops, 4)
        self.assertEqual([len(x) for x in layers], [8, 8, 8, 8])  # 2 ops * 4 ids
        for layer in range(4):
            expected = [layer * 10 + j for _ in range(2) for j in range(4)]
            self.assertEqual(layers[layer], expected)


class CoverageTest(unittest.TestCase):
    def test_hand_computed_topk(self):
        ids = [1, 1, 2, 3, 1, 2, 4]  # counts: 1->3, 2->2, 3->1, 4->1, total 7
        self.assertAlmostEqual(routing_skew.topk_coverage(ids, 1), 3 / 7)
        self.assertAlmostEqual(routing_skew.topk_coverage(ids, 2), 5 / 7)
        self.assertAlmostEqual(routing_skew.topk_coverage(ids, 3), 6 / 7)
        self.assertAlmostEqual(routing_skew.topk_coverage(ids, 4), 1.0)

    def test_ties_do_not_overshoot(self):
        ids = [1, 2, 3, 4]  # every expert used exactly once
        self.assertAlmostEqual(routing_skew.topk_coverage(ids, 2), 0.5)

    def test_empty_input(self):
        self.assertEqual(routing_skew.topk_coverage([], 4), 0.0)


class LruHitTest(unittest.TestCase):
    def test_hand_computed_two_slot_cache(self):
        # 1 miss, 2 miss, 1 hit, 3 miss (evict 2), 1 hit, 2 miss (evict 3)
        self.assertAlmostEqual(
            routing_skew.lru_hit([1, 2, 1, 3, 1, 2], 2), 2 / 6)

    def test_cache_bigger_than_working_set(self):
        self.assertAlmostEqual(
            routing_skew.lru_hit([1, 2, 1, 2, 1], 4), 3 / 5)

    def test_single_expert_repeat(self):
        self.assertAlmostEqual(routing_skew.lru_hit([5, 5, 5], 1), 2 / 3)

    def test_all_misses(self):
        self.assertEqual(routing_skew.lru_hit([1, 2, 3], 3), 0.0)

    def test_empty_input(self):
        self.assertEqual(routing_skew.lru_hit([], 2), 0.0)


if __name__ == "__main__":
    unittest.main()
