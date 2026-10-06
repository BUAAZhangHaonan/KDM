"""Real scheduling invariant checks for the explicit shared admission."""
import unittest
from workflows.hallusion_blind.generate_shared128 import validate_lock_records, prioritize

class SharedAdmissionTests(unittest.TestCase):
    def test_shared_and_slot_must_belong_to_process(self):
        proof = ["1: FLOCK ADVISORY READ 91 08:01:100 0 EOF",
                 "2: FLOCK ADVISORY WRITE 91 08:01:101 0 EOF"]
        self.assertTrue(validate_lock_records(proof, 91, "08:01:100", "08:01:101")["exclusive_slot"])
        with self.assertRaises(ValueError):
            validate_lock_records(proof, 92, "08:01:100", "08:01:101")

    def test_exclusive_main_is_not_disguised_as_shared(self):
        with self.assertRaises(ValueError):
            validate_lock_records(["1: FLOCK ADVISORY WRITE 91 08:01:100 0 EOF",
                                   "2: FLOCK ADVISORY WRITE 91 08:01:101 0 EOF"],
                                  91, "08:01:100", "08:01:101")

    def test_slot_cannot_be_shared_or_absent(self):
        for slot in ([], ["2: FLOCK ADVISORY READ 91 08:01:101 0 EOF"]):
            with self.assertRaises(ValueError):
                validate_lock_records(["1: FLOCK ADVISORY READ 91 08:01:100 0 EOF"] + slot,
                                      91, "08:01:100", "08:01:101")

    def test_priority_preserves_exact_assigned_keys(self):
        self.assertEqual(prioritize(["a", "b", "c"], ["c"]), ["c", "a", "b"])
        for bad in (["c", "c"], ["foreign"]):
            with self.assertRaises(ValueError):
                prioritize(["a", "b", "c"], bad)

if __name__ == "__main__":
    unittest.main()
