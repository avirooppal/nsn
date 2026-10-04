import sqlite3
import pytest
from nsn import Runtime
from neurosleepnet.embeddings.pooled import FastHashEncoder


@pytest.mark.parametrize('semantic', [False, True])
@pytest.mark.parametrize('session', [None, 's'])
def test_cancelled_and_deleted_events_cannot_displace_live_evidence(tmp_path, semantic, session):
    rt=Runtime(str(tmp_path),encoder=FastHashEncoder(dimension=32) if semantic else None)
    try:
        live=rt.append_event(namespace='agent',session_id='s',content='gateway port live')
        cancelled=rt.append_event(namespace='agent',session_id='s',content='gateway port cancelled',turn_status='cancelled')
        deleted=rt.append_event(namespace='agent',session_id='s',content='gateway port deleted')
        # Mimic status changes by another process, without relying on cache invalidation.
        with sqlite3.connect(rt.db_path) as conn:
            conn.execute("UPDATE events SET turn_status='deleted' WHERE id=?",(deleted,))
        pack=rt.retrieve_pack('gateway port',namespace='agent',session_id=session,limit=1)
        assert {item.id for item in pack.items}=={live}
        assert not ({cancelled,deleted}&{item.id for item in pack.items})
    finally:rt.close()
