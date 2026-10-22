#!/usr/bin/env python3
"""The two READMEs must make the same numeric claims.

A translated README drifts silently: a number gets corrected in README.md and
not in README.zh-CN.md, and the Chinese page keeps publishing a number nobody
re-measured. This test treats the throughput and bandwidth figures as the
contract between the two files and fails when either side gains or loses one.

Deliberately narrow: it compares the *set of numbers attached to tok/s, GB/s
and MiB/token*, not prose. Word order and phrasing are free to differ; a new
tok/s figure in one file and not the other is not.
"""
import os
import re
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FILES = ("README.md", "README.zh-CN.md")

# 30.6 tok/s, 16.5 GB/s, 246.6 MiB/token -> ("30.6", "tok/s") ...
CLAIM = re.compile(r"(\d+(?:\.\d+)?)\s*(tok/s|GB/s|MiB/token)")


def claims(path):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    return {(value, unit) for value, unit in CLAIM.findall(text)}


class TestReadmesAgree(unittest.TestCase):
    def setUp(self):
        self.claims = {
            name: claims(os.path.join(ROOT, name))
            for name in FILES
            if os.path.exists(os.path.join(ROOT, name))
        }
        if len(self.claims) != len(FILES):
            self.skipTest("both READMEs are not present")

    def test_same_claims(self):
        en, zh = self.claims["README.md"], self.claims["README.zh-CN.md"]
        self.assertEqual(
            en - zh, set(), "claims only in README.md: %s" % sorted(en - zh))
        self.assertEqual(
            zh - en, set(), "claims only in README.zh-CN.md: %s" % sorted(zh - en))

    def test_no_number_is_promoted(self):
        # A projected row is the easiest thing to lose in translation: the
        # Chinese page saying "50.3 tok/s" without "projected" would read as a
        # measurement. Require the word next to the value in both files.
        for name, word in (("README.md", "project"), ("README.zh-CN.md", "投影")):
            path = os.path.join(ROOT, name)
            with open(path, encoding="utf-8") as f:
                lines = f.readlines()
            # The label sits in the sentence around the row, not necessarily on
            # the same line, so widen the window instead of demanding one line.
            for value in ("50.3", "67.6"):
                idx = [i for i, ln in enumerate(lines)
                       if value in ln and "tok/s" in ln]
                self.assertTrue(idx, f"{name}: no {value} tok/s row found")
                for i in idx:
                    window = " ".join(lines[max(0, i - 1):i + 2])
                    self.assertIn(word, window,
                                  f"{name}: {value} tok/s is not labelled "
                                  f"'{word}': {window.strip()[:160]!r}")

    def test_language_switch(self):
        for name, other in (("README.md", "README.zh-CN.md"),
                            ("README.zh-CN.md", "README.md")):
            path = os.path.join(ROOT, name)
            with open(path, encoding="utf-8") as f:
                head = "".join(f.readlines()[:15])
            self.assertIn(other, head, f"{name} does not link {other} near the top")


if __name__ == "__main__":
    unittest.main()
