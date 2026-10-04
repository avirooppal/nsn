import sqlite3
from nsn import Runtime


def test_repeated_retrieval_is_scoped_and_returned_packs_are_independent(tmp_path):
    with Runtime(str(tmp_path)) as rt:
        a=rt.append_event(namespace='a',session_id='s1',content='gateway port 8080')
        b=rt.append_event(namespace='b',session_id='s1',content='gateway port 9090')
        first=rt.retrieve_pack('gateway port',namespace='a',session_id='s1')
        first.items[0].content='caller mutation'
        again=rt.retrieve_pack('gateway port',namespace='a',session_id='s1')
        assert {i.id for i in again.items}=={a} and '8080' in again.items[0].content
        assert {i.id for i in rt.retrieve_pack('gateway port',namespace='b').items}=={b}
        assert not rt.retrieve_pack('gateway port',namespace='a',session_id='s2').items


def test_cached_candidates_observe_own_appends_and_external_status_updates(tmp_path):
    with Runtime(str(tmp_path)) as rt:
        old=rt.append_event(namespace='a',content='gateway port 8080')
        assert rt.retrieve_pack('gateway port',namespace='a').items
        new=rt.append_event(namespace='a',content='gateway port 9090')
        assert {old,new}<={i.id for i in rt.retrieve_pack('gateway port',namespace='a').items}
        with sqlite3.connect(rt.db_path) as conn:
            conn.execute("UPDATE events SET turn_status='cancelled' WHERE id=?",(old,))
        assert {i.id for i in rt.retrieve_pack('gateway port',namespace='a').items}=={new}
        rt.storage.delete_event(new,'a')
        assert not rt.retrieve_pack('gateway port',namespace='a').items


def test_cache_observes_another_runtime_owner_commit(tmp_path):
    with Runtime(str(tmp_path)) as first,Runtime(str(tmp_path)) as second:
        assert not first.retrieve_pack('gateway port').items
        event=second.append_event(content='gateway port 8080')
        assert {i.id for i in first.retrieve_pack('gateway port').items}=={event}
