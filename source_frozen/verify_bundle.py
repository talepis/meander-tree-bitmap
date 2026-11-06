"""Standard-library file integrity check, independent of NumPy and Numba."""
import hashlib
import json
from pathlib import Path

def main():
    root=Path(__file__).resolve().parent
    manifest=json.loads((root/'SHA256_MANIFEST.json').read_text(encoding='utf-8'))
    for relative,digest in manifest['files'].items():
        path=(root/relative).resolve()
        if root not in path.parents or not path.is_file():
            raise ValueError('Missing or invalid member: '+relative)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError('SHA256 mismatch: '+relative)
    print('PACKAGE_SOURCE_HASHES_PASS: %d files'%len(manifest['files']))

if __name__=='__main__':
    main()
