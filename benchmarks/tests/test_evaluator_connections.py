import sqlite3
from pathlib import Path
import pytest
from nsn import Runtime
from benchmarks.public_eval.runner import lexical, text
from benchmarks.public_eval.corpus_cache import CorpusCache


def track_connections(monkeypatch):
    original = sqlite3.connect
    tracked = []
    def connect(*args, **kwargs):
        conn = original(*args, **kwargs)
        tracked.append(conn)
        return conn
    monkeypatch.setattr(sqlite3, 'connect', connect)
    return tracked


def assert_closed(connections):
    assert connections
    for connection in connections:
        with pytest.raises(sqlite3.ProgrammingError, match='closed'):
            connection.execute('SELECT 1')


def test_lexical_baseline_releases_private_database_handle(tmp_path, monkeypatch):
    runtime = Runtime(tmp_path / 'memory')
    runtime.append_event(namespace='development', id='native', content='gateway port 8080')
    tracked = track_connections(monkeypatch)
    try:
        assert lexical(runtime, 'gateway port') == ['native']
        assert_closed(tracked)
    finally:
        for connection in tracked: connection.close()
        runtime.close()
    # Required on Windows: cleanup must not wait for a connection's GC cycle.
    Path(runtime.db_path).unlink()


def test_corpus_backup_and_restore_release_non_owned_handles(tmp_path, monkeypatch):
    first, second = Runtime(tmp_path / 'first'), Runtime(tmp_path / 'second')
    tracked = track_connections(monkeypatch)
    event = {'id':'native','session':'session','date':'literal date','role':'user',
             'speaker':'speaker','content':'gateway port 8080'}
    cache = CorpusCache(tmp_path / 'cache', {'source':'pinned'})
    try:
        cache.prepare(first, [event], None, text)
        cache.prepare(second, [event], None, text)
        assert_closed(tracked)
    finally:
        for connection in tracked: connection.close()
        first.close(); second.close()
