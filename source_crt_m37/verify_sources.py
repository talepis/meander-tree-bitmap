"""Verify the immutable file manifest for the M37 modular family package."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    manifest = json.loads((HERE / 'CRT_SOURCE_SHA256.json').read_text())
    files = manifest['files']
    actual = {p.relative_to(HERE).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in HERE.rglob('*') if p.is_file() and p.suffix in ('.py', '.json', '.txt')
              and p.name != 'CRT_SOURCE_SHA256.json'
              and not any(part.startswith('.') or part == '__pycache__' for part in p.relative_to(HERE).parts)}
    if actual != files:
        missing = set(files) - set(actual)
        extra = set(actual) - set(files)
        changed = [k for k in set(files) & set(actual) if files[k] != actual[k]]
        raise RuntimeError(f'Frozen source mismatch: missing={missing}, extra={extra}, changed={changed}')
    print(f'M37_CRT_SOURCE_HASHES_PASS: {len(files)} files')


if __name__ == '__main__':
    main()
