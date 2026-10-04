"""Resume durable evaluation rows without changing the frozen experiment."""
import json
from pathlib import Path
from benchmarks.public_eval.datasets import digest, smoke_cases


def select_cases(cases, frozen, selection, per_dataset):
    if selection == 'smoke':
        if per_dataset < 1:
            raise ValueError('per_dataset must be positive')
        return smoke_cases(cases, frozen, per_dataset)
    if selection not in ('development', 'evaluation'):
        raise ValueError('Unknown selection')
    return sorted((c for c in cases if frozen['split'][c['id']] == selection), key=lambda c: c['id'])


def open_run(directory, manifest, frozen, resume=False):
    root = Path(directory)
    if not resume:
        root.mkdir(parents=True, exist_ok=False)
        (root/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        (root/'split.json').write_text(json.dumps(frozen, indent=2), encoding='utf-8')
        return manifest, []
    saved = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    unhashed = {k:v for k,v in saved.items() if k != 'manifest_sha256'}
    if digest(unhashed) != saved['manifest_sha256']:
        raise ValueError('Saved manifest checksum mismatch')
    # Hardware availability changes between processes; experimental inputs cannot.
    for key in manifest:
        if key not in ('hardware', 'manifest_sha256') and saved.get(key) != manifest[key]:
            raise ValueError('Cannot resume changed experiment: '+key)
    if json.loads((root/'split.json').read_text(encoding='utf-8')) != frozen:
        raise ValueError('Frozen split changed')
    path = root/'rows.jsonl'
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []
    pairs = set()
    for row in rows:
        pair = (row['case_id'], row['profile'])
        if row['case_id'] not in saved['case_ids'] or row['profile'] not in saved['profiles'] or pair in pairs:
            raise ValueError('Invalid or duplicate durable row')
        pairs.add(pair)
    return saved, rows
