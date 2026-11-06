"""Check all three residue rings, including mapped and parallel transfers."""
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine64"))
import tree_mod_arith as M


def main():
    rng = random.Random(20261105)
    p = (1 << 64) - 1
    for _ in range(400):
        a = rng.randrange(p)
        b = rng.randrange(p)
        w = rng.randrange(1 << 32)
        assert int(M._plus_ones(np.uint64(a), np.uint64(b))) == (a + b) % p
        assert int(M._times_ones(np.uint64(a), np.uint64(w))) == a * w % p
    for modulus in ((1 << 64), p, 7):
        M.configure(modulus)
        original = np.array([1, (1 << 64) - 2, (1 << 63) + 17,
                             (1 << 64) - 5, 5, (1 << 63) + 29], dtype=np.uint64)
        source_values = np.array([int(v) % modulus for v in original], dtype=np.uint64)
        source = source_values.view(np.uint8)
        source_shape = np.array([[2, 3]], np.int64)
        target_shape = np.array([[3, 4]], np.int64)
        source_offsets = np.array([0, 48], np.int64)
        target_offsets = np.array([0, 96], np.int64)
        widths = np.array([8], np.uint8)
        ops = np.array([[0, 0, 0, 2, 0, 3, 0, 0, 0, 0, 0, 0, 7, 1],
                        [0, 0, 0, 2, 0, 3, 1, 0, 0, 0, 2, 0, 9, 1]], np.int64)
        indices = np.array([1, 2, 1, 3, 1], np.int64)
        maps = np.zeros(0, np.int64)
        for threshold in (1, 1_000_000):
            target = np.zeros(12, dtype=np.uint64).view(np.uint8)
            M.execute(source, target, source_offsets, target_offsets, widths, widths,
                      source_shape, target_shape, ops, indices, maps, threshold)
            expected = np.zeros((3, 4), dtype=object)
            for i in range(2):
                for j in range(3):
                    expected[i, j] += 7 * int(source_values[i * 3 + j])
                    expected[int(indices[i]), int(indices[2 + j])] += 9 * int(source_values[i * 3 + j])
            actual = target.view(np.uint64).reshape(3, 4)
            for i in range(3):
                for j in range(4):
                    assert int(actual[i, j]) == int(expected[i, j]) % modulus
        lookup = np.array([[0, 1], [2, 3]], np.int64)
        weights = np.array([3, 5, 7, 11], np.uint32)
        block = target.view(np.uint64).reshape(3, 4)[:2, :2].copy().reshape(-1).view(np.uint8)
        residue, hi = M.dot128(block, 0, 8, np.array([2, 2], np.int64), lookup,
                               weights, np.uint64(0), np.uint64(0))
        expected_dot = sum(int(block.view(np.uint64)[i * 2 + j]) * int(weights[lookup[i, j]])
                           for i in range(2) for j in range(2)) % modulus
        assert int(residue) == expected_dot and int(hi) == 0
    print("M37_MODULAR_RINGS_PASS: 400 randomized arithmetic pairs, 3 rings, serial/parallel mapped transfers")


if __name__ == "__main__":
    main()
