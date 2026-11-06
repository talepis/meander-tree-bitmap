"""Check every published member against the root SHA-256 manifest."""
import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parent
manifest=json.loads((root/'MANIFEST_SHA256.json').read_text(encoding='utf-8'))
expected=manifest['files']
actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p.name!='MANIFEST_SHA256.json' and '.git' not in p.parts and '.venv' not in p.parts}
assert actual==set(expected),(sorted(actual-set(expected)),sorted(set(expected)-actual))
for rel,digest in expected.items():
    p=(root/rel).resolve()
    assert root in p.parents and hashlib.sha256(p.read_bytes()).hexdigest()==digest,rel
print('PUBLIC_PACKAGE_SHA256_PASS:',len(expected),'files')
