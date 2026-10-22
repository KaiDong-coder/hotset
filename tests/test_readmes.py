#!/usr/bin/env python3
"""The English and Chinese docs must make the same numeric claims.

A translated document drifts silently: a number gets corrected in one file and
not the other, and the translated page keeps publishing a figure nobody
re-measured. This test treats the throughput and bandwidth figures as the
contract between the two languages and fails when either side gains or loses
one.

Deliberately narrow: it compares the *numbers attached to tok/s, GB/s and
MiB/token*, not prose. Word order and phrasing are free to differ; a new tok/s
figure in one file and not the other is not.
"""
import os
import re
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

PAIRS = (("README.md", "README.zh-CN.md"),
         ("ROADMAP.md", "ROADMAP.zh-CN.md"))

# 30.6 tok/s, 16.5 GB/s, 246.6 MiB/token -> ("30.6", "tok/s") ...
CLAIM = re.compile(r"(\d+(?:\.\d+)?)\s*(tok/s|GB/s|MiB/token)")

# The labels that separate a replayed projection from a measurement.
PROJECTION_WORD = {"README.md": "project", "README.zh-CN.md": "投影",
                   "ROADMAP.md": "project", "ROADMAP.zh-CN.md": "投影"}


def claims(path):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    return {(value, unit) for value, unit in CLAIM.findall(text)}


def near(path, value, lines=5):
    """Every window of `lines` lines around a `value ... tok/s` occurrence."""
    with open(path, encoding="utf-8") as f:
        text = f.readlines()
    windows = []
    for i, line in enumerate(text):
        if value in line and "tok/s" in line:
            windows.append(" ".join(text[max(0, i - lines):i + lines + 1]))
    return windows


class TestDocsAgree(unittest.TestCase):
    def test_same_claims(self):
        for en_name, zh_name in PAIRS:
            if not (os.path.exists(os.path.join(ROOT, en_name))
                    and os.path.exists(os.path.join(ROOT, zh_name))):
                self.skipTest(f"missing one of {en_name}, {zh_name}")
            with self.subTest(pair=en_name):
                en, zh = claims(os.path.join(ROOT, en_name)), claims(
                    os.path.join(ROOT, zh_name))
                self.assertEqual(en - zh, set(),
                                 f"claims only in {en_name}: {sorted(en - zh)}")
                self.assertEqual(zh - en, set(),
                                 f"claims only in {zh_name}: {sorted(zh - en)}")

    def test_no_number_is_promoted(self):
        # A projected row is the easiest thing to lose in translation: saying
        # "50.3 tok/s" without the label reads as a measurement.
        for name, word in PROJECTION_WORD.items():
            path = os.path.join(ROOT, name)
            if not os.path.exists(path):
                continue
            for value in ("50.3", "67.6"):
                with self.subTest(file=name, value=value):
                    windows = near(path, value)
                    self.assertTrue(windows, f"{name}: no {value} tok/s row")
                    for window in windows:
                        self.assertIn(word, window,
                                      f"{name}: {value} tok/s is not labelled "
                                      f"'{word}': {window.strip()[:160]!r}")

    def test_links_resolve(self):
        # A README that promises "results/README.md" and has no such file is
        # worse than one that promises nothing: the first comment on any forum
        # will be the dead link.
        links = re.compile(r"\]\(([^)\s]+)\)")
        for name in ("README.md", "README.zh-CN.md", "ROADMAP.md",
                     "ROADMAP.zh-CN.md"):
            path = os.path.join(ROOT, name)
            if not os.path.exists(path):
                continue
            with open(path, encoding="utf-8") as f:
                text = f.read()
            for target in links.findall(text):
                if target.startswith(("http://", "https://", "#", "mailto:")):
                    continue
                target = target.split("#")[0]
                with self.subTest(file=name, link=target):
                    self.assertTrue(os.path.exists(os.path.join(ROOT, target)),
                                    f"{name}: dead link {target}")

    def test_language_switch(self):
        for en_name, zh_name in PAIRS:
            for name, other in ((en_name, zh_name), (zh_name, en_name)):
                path = os.path.join(ROOT, name)
                if not os.path.exists(path):
                    continue
                with self.subTest(file=name):
                    with open(path, encoding="utf-8") as f:
                        head = "".join(f.readlines()[:15])
                    self.assertIn(other, head,
                                  f"{name} does not link {other} near the top")


if __name__ == "__main__":
    unittest.main()
