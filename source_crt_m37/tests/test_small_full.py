"""Whole small-order family comparisons against the frozen exact rank-64 engine."""
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from common import configure

cache = Path(os.environ.get('MEANDER_CACHE_DIR', str(HERE.parents[1] / 'm38cache')))
cache.mkdir(exist_ok=True)
configure('64', 2, cache)
from linear_support import calculate

KNOWN = json.loads((HERE / 'KNOWN_RESULTS.json').read_text())
FLAGS = dict(workers=2, tail=8, memory_mib=2048, timeout=900, dense_cut=-125,
             minimize_grammar=True, split_maps=True, lifetime=True,
             geometry_mode='target', parallel_geometry=2, index_prealloc_bytes=0)


def main():
    count = 0
    for n, groups in ((6, 2), (10, 3), (14, 4)):
        sizes = list(range(1, (n + 1) // 2 + 1))
        partition = [sizes[i::groups] for i in range(groups)]
        exact = []
        for subset in partition:
            value = calculate(n, lower_sizes=subset, numeric_mode='parallel', **FLAGS)
            exact.append(int(value['Mn']))
        assert sum(exact) == int(KNOWN[str(n)])
        for modulus in (1 << 64, (1 << 64) - 1, 7):
            residues = []
            for subset, value in zip(partition, exact):
                result = calculate(n, lower_sizes=subset, numeric_mode='modular', modulus=modulus, **FLAGS)
                residue = int(result['Mn'])
                assert residue == value % modulus, (n, subset, modulus, residue, value)
                assert result['modulus_decimal'] == str(modulus)
                residues.append(residue)
                count += 1
            assert sum(residues) % modulus == int(KNOWN[str(n)]) % modulus
    print(f'M37_MODULAR_WHOLE_SMALL_PASS: {count} group-residue comparisons')


if __name__ == '__main__':
    main()
