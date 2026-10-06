"""Kernel descriptor ownership and shard ordering checks for shared admission."""
import unittest
from workflows.hallusion_blind.generate_shared128 import validate_lock_records, prioritize

class SharedAdmissionTests(unittest.TestCase):
    def test_descriptor_owned_flock_preserves_external_creator_pid(self):
        main = ["pos: 0", "lock: 1: FLOCK ADVISORY READ 91 08:01:100 0 EOF"]
        slot = ["lock: 1: FLOCK ADVISORY WRITE 93 08:01:101 0 EOF"]
        proof = validate_lock_records(main, slot, "08:01:100", "08:01:101")
        self.assertTrue(proof["exclusive_slot"])
        self.assertEqual(proof["main_lock_creator_pid"], 91)
        self.assertEqual(proof["slot_lock_creator_pid"], 93)
        with self.assertRaises(ValueError):
            validate_lock_records(main, slot, "08:01:999", "08:01:101")

    def test_exclusive_main_is_not_disguised_as_shared(self):
        with self.assertRaises(ValueError):
            validate_lock_records(["lock: 1: FLOCK ADVISORY WRITE 91 08:01:100 0 EOF"],
                                  ["lock: 1: FLOCK ADVISORY WRITE 93 08:01:101 0 EOF"],
                                  "08:01:100", "08:01:101")

    def test_slot_cannot_be_shared_or_absent(self):
        for slot in ([], ["lock: 1: FLOCK ADVISORY READ 93 08:01:101 0 EOF"]):
            with self.assertRaises(ValueError):
                validate_lock_records(["lock: 1: FLOCK ADVISORY READ 91 08:01:100 0 EOF"],
                                      slot, "08:01:100", "08:01:101")

    def test_priority_preserves_exact_assigned_keys(self):
        self.assertEqual(prioritize(["a", "b", "c"], ["c"]), ["c", "a", "b"])
        for bad in (["c", "c"], ["foreign"]):
            with self.assertRaises(ValueError):
                prioritize(["a", "b", "c"], bad)

if __name__ == "__main__":
    unittest.main()
