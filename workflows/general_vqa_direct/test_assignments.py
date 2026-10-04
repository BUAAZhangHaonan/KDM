"""Ownership and coverage checks for moving a sealed tail between workers."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from workflows.general_vqa_direct import prepare_assignment as assign


class AssignmentTests(unittest.TestCase):
    def test_partition_union_and_balance(self):
        for size, count in [(1, 1), (1936, 3), (887, 4), (8, 8)]:
            keys = [str(i) for i in range(size)]
            shards = assign.partition_keys(keys, count)
            flat = sum(shards, [])
            self.assertEqual(set(flat), set(keys))
            self.assertEqual(len(flat), len(set(flat)))
            self.assertLessEqual(max(map(len, shards)) - min(map(len, shards)), 1)

    def test_invalid_partitions(self):
        for keys, count in [([], 1), (["a"], 0), (["a"], 2), (["a", "a"], 1)]:
            with self.assertRaises(ValueError):
                assign.partition_keys(keys, count)

    def state(self, *, live=False, identity="registered", release=True, tamper=False):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        run = root / "source/claims/source_run"
        run.mkdir(parents=True)
        pid = os.getpid() if live else 99999999
        tick = int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]) if live else 1
        owner = {"pid": pid, "start_tick": tick, "identity": identity, "keys": ["done", "next", "held"]}
        (run / "owner.json").write_text(json.dumps(owner))
        if release:
            (run / "released.json").write_text(json.dumps({
                "claim_identity": "wrong" if tamper else assign.generate.stable_hash(owner),
                "reason": "stop_after_completed_chunk", "keys": ["next"],
                "released_utc": "2026-10-04T16:00:00+00:00"}))
        with patch.object(assign, "ROOT", root):
            return assign.claim_state({"output": root / "source", "identity": "registered"}, {"done"})

    def test_only_exited_released_keys_are_free(self):
        occupied, bindings, released = self.state()
        self.assertEqual(occupied, {"held"})
        self.assertEqual(set(released), {"next"})
        self.assertEqual(len(bindings), 2)

    def test_unreleased_keys_stay_occupied(self):
        self.assertEqual(self.state(release=False)[0], {"next", "held"})

    def test_live_source_is_rejected(self):
        with self.assertRaisesRegex(AssertionError, "still running"):
            self.state(live=True)

    def test_foreign_identity_is_rejected(self):
        with self.assertRaisesRegex(AssertionError, "identity"):
            self.state(identity="foreign")

    def test_unbound_release_is_rejected(self):
        with self.assertRaises(AssertionError):
            self.state(tamper=True)


if __name__ == "__main__":
    unittest.main()
