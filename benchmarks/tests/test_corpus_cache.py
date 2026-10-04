import sqlite3
import pytest
from nsn import Runtime
from benchmarks.public_eval.corpus_cache import CorpusCache
from benchmarks.public_eval.runner import text, lexical, packed


def events(content='Gateway production port 8080.'):
    return [{'id': 'native-1', 'session': 'session-1', 'date': '2025-01-01',
             'role': 'user', 'speaker': 'speaker', 'content': content}]


def test_cache_copies_are_isolated_and_rank_like_fresh_build(tmp_path):
    cache = CorpusCache(tmp_path / 'cache', {'source': 'pinned', 'sdk': 'pinned'})
    with Runtime(tmp_path / 'first') as first:
        _, build = cache.prepare(first, events(), None, text)
        ranked = lexical(first, 'Gateway production port')
        first.append_event(namespace='development', content='Never cache this prediction.')
        first.storage.delete_event('native-1', 'development')
    with Runtime(tmp_path / 'second') as second:
        _, restored = cache.prepare(second, events(), None, text)
        assert lexical(second, 'Gateway production port') == ranked
        with sqlite3.connect(second.db_path) as conn:
            assert conn.execute('SELECT id FROM events').fetchall() == [('native-1',)]
        assert not second.retrieve_pack('Gateway', namespace='other').items
    assert not build['hit'] and restored['hit']


def test_fresh_and_cached_nsn_candidates_produce_identical_answer_prompt(tmp_path):
    from benchmarks.public_eval.runner import SYSTEM
    source = events() + [{**events()[0], 'id': 'native-2', 'role': 'assistant',
                          'content': 'Payment gateway staging port 9090.'},
                         {**events()[0], 'id': 'native-3',
                          'content': 'Production database host db.internal.'}]
    cache = CorpusCache(tmp_path / 'cache', {'source': 'pinned', 'sdk': 'pinned'})
    with Runtime(tmp_path / 'build') as rt:
        cache.prepare(rt, source, None, text)
    contexts = []
    for name in ('fresh', 'cached'):
        with Runtime(tmp_path / name) as rt:
            if name == 'cached':
                cache.prepare(rt, source, None, text)
            else:
                for event in source:
                    rt.append_event(namespace='development', id=event['id'], role=event['role'],
                                    content=text(event), session_id=event['session'])
            lookup = {e['id']: e for e in source}
            pack = rt.retrieve_pack('gateway production port', namespace='development',
                                    max_tokens=512, use_graph=False, use_summaries=False)
            ranked = [lookup[item.id] for item in pack.items]
            context, ids, omitted = packed(ranked, 512)
            contexts.append((SYSTEM + '\nReference conversation:\n' + context +
                             '\nQuestion: gateway production port\nAnswer:', ids, omitted))
    assert contexts[0] == contexts[1]
    assert '8080' in contexts[0][0]


def test_cache_content_identity_and_corruption_are_visible(tmp_path):
    cache = CorpusCache(tmp_path / 'cache', {'source': 'pinned'})
    with Runtime(tmp_path / 'a') as rt:
        _, old = cache.prepare(rt, events(), None, text)
    with Runtime(tmp_path / 'b') as rt:
        _, new = cache.prepare(rt, events('Gateway port 9090.'), None, text)
    assert old['key'] != new['key']
    (tmp_path / 'cache' / old['key'] / 'events.sqlite').write_bytes(b'corrupt')
    with Runtime(tmp_path / 'c') as rt:
        with pytest.raises(ValueError, match='checksum mismatch'):
            cache.prepare(rt, events(), None, text)


def test_embedding_cache_reuse_and_encoder_revision_invalidation(tmp_path):
    import numpy as np
    from neurosleepnet.embeddings.pooled import FastHashEncoder
    class Encoder(FastHashEncoder):
        def __init__(self, revision):
            super().__init__(dimension=16, revision=revision)
            self.calls = 0
        def encode_batch(self, texts):
            self.calls += 1
            return super().encode_batch(texts)
    cache = CorpusCache(tmp_path / 'cache', {'source': 'pinned'})
    encoder = Encoder('v1')
    source = events()
    source[0]['native'] = {'has_answer': True}  # Scoring-only annotation is never cached.
    with Runtime(tmp_path / 'a') as rt:
        first, info = cache.prepare(rt, source, encoder, text)
    with Runtime(tmp_path / 'b') as rt:
        second, hit = cache.prepare(rt, events(), encoder, text)
        assert encoder.calls == 1 and hit['hit']
        assert np.array_equal(first, second)
        assert 'has_answer' not in rt.storage.get_event('native-1', 'development')['metadata']
    with Runtime(tmp_path / 'c') as rt:
        _, changed = cache.prepare(rt, source, Encoder('v2'), text)
        assert changed['key'] != info['key'] and not changed['hit']
