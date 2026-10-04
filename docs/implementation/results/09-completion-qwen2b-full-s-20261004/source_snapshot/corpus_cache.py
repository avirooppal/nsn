"""Immutable native-corpus builds; each question receives a private DB copy."""
import hashlib
import json
from pathlib import Path
import sqlite3
import time
from benchmarks.public_eval.datasets import digest


def checksum(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


class CorpusCache:
    def __init__(self, root, identity):
        self.root = Path(root)
        self.identity = identity

    def prepare(self, runtime, events, encoder, render):
        # No question, answer, gold evidence, generated memory or grade is input.
        native = [{k: e[k] for k in ('id', 'session', 'date', 'role', 'speaker', 'content')}
                  | {'caption': e.get('caption')} for e in events]
        key = digest([self.identity, native, encoder.fingerprint if encoder else None])
        entry = self.root / key
        metadata_path = entry / 'build.json'
        started = time.perf_counter()
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
            if metadata['key'] != key or any(checksum(entry / name) != value for name, value in metadata['file_sha256'].items()):
                raise ValueError('Cached native corpus checksum mismatch')
            with sqlite3.connect('file:' + (entry / 'events.sqlite').as_posix() + '?mode=ro', uri=True) as source:
                with runtime.storage._runtime_connection() as target:
                    source.backup(target)
            embeddings = None
            if encoder:
                import numpy as np
                embeddings = np.load(entry / 'vectors.npy', allow_pickle=False)
                if embeddings.shape != (len(events), encoder.dimension):
                    raise ValueError('Cached embedding shape mismatch')
            return embeddings, {**metadata, 'hit': True, 'restore_ms': (time.perf_counter() - started) * 1000}

        entry.mkdir(parents=True, exist_ok=True)
        for event in events:
            runtime.append_event(namespace='development', id=event['id'], role=event['role'],
                                 content=render(event), session_id=event['session'],
                                 metadata={'native_date': event['date'], 'speaker': event['speaker']})
        ingestion_ms = (time.perf_counter() - started) * 1000
        embeddings = None
        embedding_ms = 0
        if encoder:
            import numpy as np
            started = time.perf_counter()
            embeddings = np.asarray(encoder.encode_batch([render(event) for event in events]), dtype=np.float32)
            embedding_ms = (time.perf_counter() - started) * 1000
            np.save(entry / 'vectors.npy', embeddings, allow_pickle=False)
        with sqlite3.connect(entry / 'events.sqlite') as destination:
            with runtime.storage._runtime_connection() as source:
                source.backup(destination)
        files = ['events.sqlite'] + (['vectors.npy'] if encoder else [])
        metadata = {'key': key, 'native_events': len(events), 'ingestion_build_ms': ingestion_ms,
                    'embedding_build_ms': embedding_ms,
                    'file_sha256': {name: checksum(entry / name) for name in files}}
        temporary = entry / 'build.tmp'
        temporary.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        temporary.replace(metadata_path)  # Incomplete builds are never cache hits.
        return embeddings, {**metadata, 'hit': False, 'restore_ms': 0}
