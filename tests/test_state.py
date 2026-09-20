import unittest

from factorio_lm.emulator import KVStorage, VectorRegister, VectorShiftRegister
from factorio_lm.fixtures import K_INPUT, REGISTER_INPUT, V_INPUT
from factorio_lm.int32 import INT32_MAX, INT32_MIN


class RegisterEmulatorTests(unittest.TestCase):
    def test_register_commit_hold_reset_and_priority(self):
        register = VectorRegister(4)
        self.assertEqual(register.read(), [0, 0, 0, 0])
        self.assertEqual(register.step([1, -2, INT32_MAX + 1, 4], commit=True), [1, -2, INT32_MIN, 4])
        self.assertEqual(register.step([9, 9, 9, 9]), [1, -2, INT32_MIN, 4])
        self.assertEqual(register.step([7, 7, 7, 7], commit=True, reset=True), [0, 0, 0, 0])

    def test_shift_register_age_order(self):
        shift = VectorShiftRegister(depth=3, width=2)
        shift.step([1, 10], commit=True)
        self.assertEqual(shift.read(), [[1, 10], [0, 0], [0, 0]])
        shift.step([2, 20], commit=True)
        shift.step([3, 30], commit=True)
        self.assertEqual(shift.read(), [[3, 30], [2, 20], [1, 10]])
        shift.step([4, 40], commit=True)
        self.assertEqual(shift.read(), [[4, 40], [3, 30], [2, 20]])
        shift.step(reset=True)
        self.assertEqual(shift.read(), [[0, 0], [0, 0], [0, 0]])

    def test_kv_storage_commits_atomically(self):
        storage = KVStorage(depth=2, width=32)
        state = storage.step(K_INPUT, V_INPUT, commit=True)
        self.assertEqual(state["keys"][0], K_INPUT)
        self.assertEqual(state["values"][0], V_INPUT)
        second_keys = list(reversed(K_INPUT))
        second_values = list(reversed(V_INPUT))
        state = storage.step(second_keys, second_values, commit=True)
        self.assertEqual(state["keys"], [second_keys, K_INPUT])
        self.assertEqual(state["values"], [second_values, V_INPUT])

    def test_invalid_state_dimensions(self):
        with self.assertRaises(ValueError):
            VectorRegister(0)
        with self.assertRaises(ValueError):
            VectorShiftRegister(0, 32)
        with self.assertRaises(ValueError):
            VectorRegister(2).step([1], commit=True)
        with self.assertRaises(ValueError):
            KVStorage(depth=2, width=2).step([1, 2], None, commit=True)


if __name__ == "__main__":
    unittest.main()

