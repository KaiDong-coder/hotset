#!/usr/bin/env python3
"""Unit tests for the VRAM queue (no GPU, no subprocess, no waiting).

Run: python3 -I -m unittest discover -s tests -p "test_gpu_queue.py"

Covers the pure pieces the real script acts on:
  - scripts/gpu_queue.py: nvidia-smi row parsing, the is-free decision, the
    N-consecutive-samples state machine (including a flickering GPU), the
    timeout clock, and the JSON evidence record naming the resident process.
  - scripts/bench_queue.py: the declarative waiting list, the
    gpu_queue.py argument it builds, and the sequential runner (a fake runner
    records the order; nothing touches a GPU).
  - the policy invariant: neither helper source contains a process-control
    call (the tenant must never be touched).
"""
import datetime
import json
import os
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(HERE, "..", "scripts")
sys.path.insert(0, SCRIPTS)

import bench_queue  # noqa: E402
import gpu_queue  # noqa: E402

SYN_MEMORY = "0, 49140, 41985, 6549\n"
SYN_APPS = (
    "2534964, /opt/strata/engine/strata, 41938\n"
    "2860067, third_party/llama.cpp/build/bin/llama-server, 262\n"
)


class FakeClock:
    """Monotonic clock that only advances when ``sleep`` is called."""

    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds


def fixed_samples(*free_mib_values):
    values = iter(free_mib_values)
    last = [None]

    def sample():
        try:
            last[0] = next(values)
        except StopIteration:
            pass
        return {"total_mib": 49140, "used_mib": 41985, "free_mib": last[0]}

    return sample


class ParseGpuMemory(unittest.TestCase):
    def test_three_and_four_column_rows(self):
        self.assertEqual(
            gpu_queue.parse_gpu_memory("49140, 41985, 6549\n"),
            {"total_mib": 49140, "used_mib": 41985, "free_mib": 6549})
        self.assertEqual(
            gpu_queue.parse_gpu_memory(SYN_MEMORY),
            {"total_mib": 49140, "used_mib": 41985, "free_mib": 6549})

    def test_first_gpu_wins(self):
        two = "0, 49140, 41985, 6549\n1, 24576, 100, 24476\n"
        self.assertEqual(gpu_queue.parse_gpu_memory(two)["total_mib"], 49140)

    def test_blank_and_nodata(self):
        self.assertIsNone(gpu_queue.parse_gpu_memory(""))
        self.assertIsNone(gpu_queue.parse_gpu_memory("\n\n"))
        # A skipped column must not shift the triple.
        self.assertIsNone(gpu_queue.parse_gpu_memory("0, [N/A], 41985, 6549\n"))


class ParseComputeApps(unittest.TestCase):
    def test_pid_name_mib(self):
        apps = gpu_queue.parse_compute_apps(SYN_APPS)
        self.assertEqual(len(apps), 2)
        self.assertEqual(apps[0], {
            "pid": 2534964,
            "process_name": "/opt/strata/engine/strata",
            "used_mib": 41938,
        })
        self.assertEqual(apps[1]["pid"], 2860067)

    def test_unsupported_memory_is_none(self):
        apps = gpu_queue.parse_compute_apps("4242, thing, [N/A]\n")
        self.assertEqual(apps[0]["used_mib"], None)

    def test_non_pid_rows_are_skipped(self):
        self.assertEqual(gpu_queue.parse_compute_apps(
            "No running processes found\n"), [])
        self.assertEqual(gpu_queue.parse_compute_apps(""), [])


class FreeDecision(unittest.TestCase):
    def test_boundary_is_inclusive(self):
        self.assertTrue(gpu_queue.is_free(47000, 47000))
        self.assertFalse(gpu_queue.is_free(46999, 47000))

    def test_missing_reading_is_not_free(self):
        self.assertFalse(gpu_queue.is_free(None, 47000))

    def test_budget_status(self):
        self.assertEqual(gpu_queue.budget_status(47000, 49140), "possible")
        self.assertEqual(gpu_queue.budget_status(50000, 49140),
                         "unsatisfiable")
        self.assertEqual(gpu_queue.budget_status(47000, None), "unknown")


class SettleTrackerTest(unittest.TestCase):
    def test_needs_n_consecutive_samples(self):
        tracker = gpu_queue.SettleTracker(47000, 3)
        self.assertFalse(tracker.feed(47000))
        self.assertFalse(tracker.feed(48000))
        self.assertTrue(tracker.feed(47000))
        self.assertEqual(tracker.samples, 3)

    def test_a_low_sample_resets_the_streak(self):
        tracker = gpu_queue.SettleTracker(47000, 3)
        self.assertFalse(tracker.feed(50000))
        self.assertFalse(tracker.feed(50000))
        self.assertFalse(tracker.feed(46999))
        self.assertEqual(tracker.streak, 0)
        self.assertFalse(tracker.feed(50000))
        self.assertFalse(tracker.feed(50000))
        self.assertTrue(tracker.feed(50000))
        self.assertEqual(tracker.best_streak, 3)

    def test_flickering_gpu_does_not_settle(self):
        # free, free, busy, free, free -- never three in a row
        tracker = gpu_queue.SettleTracker(47000, 3)
        for free in (48000, 48000, 6888, 48000, 48000):
            self.assertFalse(tracker.feed(free))
        self.assertEqual(tracker.streak, 2)
        self.assertFalse(tracker.settled)

    def test_single_sample_settles_immediately(self):
        tracker = gpu_queue.SettleTracker(1, 1)
        self.assertTrue(tracker.feed(1))

    def test_zero_samples_is_rejected(self):
        with self.assertRaises(ValueError):
            gpu_queue.SettleTracker(47000, 0)


class WaitForBudget(unittest.TestCase):
    def test_flicker_restarts_the_count(self):
        clock = FakeClock()
        sample = fixed_samples(48000, 48000, 6888, 48000, 48000, 48000)
        result = gpu_queue.wait_for_budget(
            sample, need_mib=47000, settle_samples=3, interval_s=30,
            timeout_s=1000, sleep_fn=clock.sleep, clock=clock.now)
        self.assertEqual(result["status"], "free")
        self.assertEqual(result["samples_taken"], 6)
        self.assertGreater(result["waited_s"], 0)

    def test_timeout_bounds_the_wait(self):
        clock = FakeClock()
        result = gpu_queue.wait_for_budget(
            fixed_samples(6888), need_mib=47000, settle_samples=3,
            interval_s=30, timeout_s=60, sleep_fn=clock.sleep,
            clock=clock.now)
        self.assertEqual(result["status"], "timeout")
        self.assertEqual(result["samples_taken"], 3)
        self.assertEqual(result["waited_s"], 60)

    def test_zero_timeout_waits_forever(self):
        clock = FakeClock()
        sample = fixed_samples(6888, 6888, 47000, 47000, 47000)
        result = gpu_queue.wait_for_budget(
            sample, need_mib=47000, settle_samples=3, interval_s=30,
            timeout_s=0, sleep_fn=clock.sleep, clock=clock.now)
        self.assertEqual(result["status"], "free")
        self.assertEqual(result["samples_taken"], 5)


class EvidenceRecord(unittest.TestCase):
    def test_record_path_names_utc_start_and_pid(self):
        when = datetime.datetime(2026, 10, 22, 1, 2, 3,
                                 tzinfo=datetime.timezone.utc)
        path = gpu_queue.record_path("/tmp/recs", when, 4242)
        self.assertEqual(path, "/tmp/recs/gpu-queue-20261022T010203Z-4242.json")

    def test_blocked_wait_record_names_resident_and_mib(self):
        sample = gpu_queue.parse_gpu_memory(SYN_MEMORY)
        residents = gpu_queue.parse_compute_apps(SYN_APPS)
        residents[0]["user"] = "fjd"
        clock = FakeClock()
        result = gpu_queue.wait_for_budget(
            fixed_samples(6549), need_mib=47000, settle_samples=3,
            interval_s=30, timeout_s=60, sleep_fn=clock.sleep,
            clock=clock.now)
        self.assertEqual(result["status"], "timeout")

        record = gpu_queue.new_record(
            need_mib=47000, settle_samples=3, interval_s=30, timeout_s=60,
            started_at="2026-10-22T01:00:00+00:00",
            finished_at="2026-10-22T01:01:00+00:00",
            status="timeout", sample=sample, residents=residents,
            waited_s=result["waited_s"],
            samples_taken=result["samples_taken"],
            consecutive_free=result["tracker"].streak,
            reason="budget never settled")
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "blocked.json")
            gpu_queue.write_record(path, record)
            with open(path) as f:
                saved = json.load(f)
        self.assertEqual(saved["status"], "timeout")
        self.assertEqual(saved["residents"][0]["pid"], 2534964)
        self.assertEqual(saved["residents"][0]["used_mib"], 41938)
        self.assertEqual(saved["residents"][0]["process_name"],
                         "/opt/strata/engine/strata")
        self.assertEqual(saved["gpu"]["free_mib"], 6549)

    def test_record_is_valid_json_with_wait_duration(self):
        record = gpu_queue.new_record(
            need_mib=47000, settle_samples=3, interval_s=30, timeout_s=60,
            started_at="2026-10-22T01:00:00+00:00", status="waiting",
            waited_s=12.3456)
        self.assertEqual(record["schema"], "gpu-queue/1")
        self.assertEqual(record["waited_s"], 12.346)
        json.dumps(record)


class NoProcessControl(unittest.TestCase):
    """The fixed policy: the helper never touches a process it did not start."""

    FORBIDDEN = ("kill", "pkill", "renice", "send_signal", "signal.",
                 "nice(")

    def test_helper_sources_have_no_process_control(self):
        for name in ("gpu_queue.py", "bench_queue.py"):
            with open(os.path.join(SCRIPTS, name)) as f:
                source = f.read().lower()
            for token in self.FORBIDDEN:
                self.assertNotIn(token, source,
                                 f"{name} contains forbidden {token!r}")


class BenchQueueList(unittest.TestCase):
    def test_waiting_list_has_the_probe_023_table_and_the_split_ab(self):
        entries = bench_queue.build_queue(model="/tmp/model.gguf",
                                          out_dir="/tmp/out")
        names = [e["name"] for e in entries]
        self.assertEqual(names[:3], ["fixed-fast", "fixed-generic", "warm"])
        self.assertEqual(names[3:], ["split-moe-cache-0",
                                     "split-moe-cache-14",
                                     "split-moe-cache-28"])
        by_name = {e["name"]: e for e in entries}
        self.assertTrue(by_name["fixed-fast"]["enabled"])
        self.assertTrue(by_name["fixed-generic"]["enabled"])
        self.assertTrue(by_name["warm"]["enabled"])
        for name in ("split-moe-cache-0", "split-moe-cache-14",
                     "split-moe-cache-28"):
            self.assertFalse(by_name[name]["enabled"])
            self.assertIn("PROBE-025", by_name[name]["blocked_by"])

    def test_bench_serve_commands_put_out_before_extra(self):
        entries = bench_queue.build_queue(model="/tmp/model.gguf",
                                          out_dir="/tmp/out")
        fast = next(e for e in entries if e["name"] == "fixed-fast")
        cmd = fast["cmd"]
        self.assertIn("--ncmoe", cmd)
        self.assertIn("8", cmd)
        self.assertIn("--extra", cmd)
        self.assertLess(cmd.index("--out"), cmd.index("--extra"))

    def test_generic_disables_both_fast_paths(self):
        entries = bench_queue.build_queue(model="/tmp/model.gguf",
                                          out_dir="/tmp/out")
        generic = next(e for e in entries if e["name"] == "fixed-generic")
        self.assertIn("GGML_MMID_IL4=0", generic["cmd"])
        self.assertIn("GGML_MMID_PREFILL=0", generic["cmd"])

    def test_format_queue_marks_blocked_entries(self):
        entries = bench_queue.build_queue(model="/tmp/model.gguf",
                                          out_dir="/tmp/out")
        text = bench_queue.format_queue(entries)
        for entry in entries:
            self.assertIn(entry["name"], text)
        self.assertIn("ready", text)
        self.assertIn("blocked", text)

    def test_select_entries_preserves_requested_order(self):
        entries = bench_queue.build_queue(model="/tmp/model.gguf",
                                          out_dir="/tmp/out")
        picked = bench_queue.select_entries(entries, "warm,fixed-fast")
        self.assertEqual([e["name"] for e in picked], ["warm", "fixed-fast"])
        with self.assertRaises(KeyError):
            bench_queue.select_entries(entries, "nope")

    def test_gpu_queue_argv_wraps_the_command(self):
        entry = {
            "name": "fixed-fast", "need_mib": 47700,
            "cmd": ["python3", "-I", "bench_serve.py", "--out", "/tmp/x"],
        }
        argv = bench_queue.gpu_queue_argv(
            entry, python="/usr/bin/python3", gpu_queue="/s/gpu_queue.py",
            log_dir="/logs")
        self.assertEqual(argv[:4], ["/usr/bin/python3", "-I",
                                    "/s/gpu_queue.py", "--need-mib"])
        self.assertEqual(argv[4], "47700")
        self.assertEqual(argv[argv.index("--record") + 1],
                         "/logs/fixed-fast.json")
        self.assertEqual(argv[argv.index("--") + 1:], entry["cmd"])


class BenchQueueRunner(unittest.TestCase):
    """Sequential execution, with a fake runner instead of gpu_queue.py."""

    def _entries(self, *names):
        return [{
            "name": name, "task": "T", "why": "w", "need_mib": 1,
            "enabled": True, "blocked_by": None, "cmd": ["true"],
        } for name in names]

    def _runner(self, log_dir, calls, status, rc):
        def run(argv):
            name = argv[argv.index("--record") + 1]
            calls.append(os.path.basename(name))
            with open(name, "w") as f:
                json.dump({"status": status}, f)
            return rc
        return run

    def test_runs_in_declared_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls = []
            entries = self._entries("first", "second")
            rc = bench_queue.run_queue(
                entries, log_dir=tmp,
                runner=self._runner(tmp, calls, "ran", 0))
            self.assertEqual(rc, 0)
            self.assertEqual(calls, ["first.json", "second.json"])

    def test_dry_run_executes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls = []
            rc = bench_queue.run_queue(
                self._entries("first"), log_dir=tmp, dry_run=True,
                runner=self._runner(tmp, calls, "ran", 0))
            self.assertEqual(rc, 0)
            self.assertEqual(calls, [])

    def test_wait_failure_stops_the_queue(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls = []
            entries = self._entries("first", "second")
            rc = bench_queue.run_queue(
                entries, log_dir=tmp,
                runner=self._runner(tmp, calls, "timeout", 1))
            self.assertEqual(rc, 1)
            self.assertEqual(calls, ["first.json"])

    def test_a_failed_benchmark_does_not_stop_the_queue(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls = []

            def runner(argv):
                name = argv[argv.index("--record") + 1]
                calls.append(os.path.basename(name))
                with open(name, "w") as f:
                    json.dump({"status": "ran"}, f)
                return 5

            rc = bench_queue.run_queue(self._entries("first", "second"),
                                       log_dir=tmp, runner=runner)
            self.assertEqual(rc, 1)
            self.assertEqual(calls, ["first.json", "second.json"])


if __name__ == "__main__":
    unittest.main()
