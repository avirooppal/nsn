"""Optional, explicit administration and procedure memory. Standard library only."""
import json
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from neurosleepnet.storage.migrations import CURRENT_SCHEMA_VERSION

TABLES = ('events', 'facts', 'fact_support', 'relationship_entities',
          'relationship_aliases', 'relationships', 'relationship_support',
          'unresolved_relationships', 'jobs', 'consolidation_sources',
          'derived_support', 'derived_cleanup', 'enrichment_proposals',
          'source_importance', 'procedures', 'procedure_traces')

class AdvancedMemory:
    def __init__(self, runtime):
        self.runtime = runtime

    def connect(self):
        if self.runtime._closed:
            raise RuntimeError('Runtime is closed')
        conn = sqlite3.connect(self.runtime.db_path, timeout=5)
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA foreign_keys=ON')
        conn.execute('PRAGMA recursive_triggers=ON')
        return conn

    def scope(self, namespace):
        return namespace or self.runtime.default_namespace

    def timeline(self, namespace=None, limit=100, session_id=None):
        if not 1 <= limit <= 10000:
            raise ValueError('limit must be 1..10000')
        conn = self.connect()
        try:
            return [dict(r) for r in conn.execute('SELECT * FROM events WHERE namespace=? AND (? IS NULL OR session_id=?) ORDER BY observed_at,id LIMIT ?', (self.scope(namespace), session_id, session_id, limit))]
        finally:
            conn.close()

    def stats(self, namespace=None):
        ns = self.scope(namespace)
        conn = self.connect()
        try:
            conn.execute('BEGIN')
            counts = {}
            for t in TABLES:
                where = 'fact_id IN (SELECT id FROM facts WHERE namespace=?)' if t == 'fact_support' else 'namespace=?'
                counts[t] = conn.execute(f'SELECT COUNT(*) FROM {t} WHERE {where}', (ns,)).fetchone()[0]
            return {'namespace':ns, 'schema_version':CURRENT_SCHEMA_VERSION, 'counts':counts}
        finally:
            conn.close()

    def export_memory(self, namespace=None):
        ns = self.scope(namespace)
        conn = self.connect()
        try:
            conn.execute('BEGIN')
            tables = {}
            for table in TABLES:
                where = 'fact_id IN (SELECT id FROM facts WHERE namespace=?)' if table == 'fact_support' else 'namespace=?'
                tables[table] = [dict(r) for r in conn.execute(f'SELECT * FROM {table} WHERE {where}', (ns,))]
            return {'format': 'nsn-scoped-backup', 'schema_version': CURRENT_SCHEMA_VERSION, 'namespace': ns, 'tables': tables}
        finally:
            conn.close()

    def import_memory(self, data, namespace=None):
        """Restore a trusted backup into an empty namespace, atomically. No overwrite."""
        ns = self.scope(namespace)
        if not isinstance(data, dict) or set(data) != {'format','schema_version','namespace','tables'} or data['format'] != 'nsn-scoped-backup' or data['schema_version'] != CURRENT_SCHEMA_VERSION or data['namespace'] != ns or not isinstance(data['tables'],dict) or set(data['tables']) != set(TABLES):
            raise ValueError('Unsupported backup schema or namespace')
        if len(json.dumps(data).encode('utf-8')) > 32 * 1024 * 1024:
            raise ValueError('Backup exceeds 32 MiB')
        conn = self.connect()
        try:
            with conn:
                conn.execute('BEGIN IMMEDIATE')
                if conn.execute('SELECT 1 FROM compact_vectors WHERE namespace=? LIMIT 1',(ns,)).fetchone():
                    raise ValueError('Destination namespace has existing vector records')
                for t in TABLES:
                    rows = data['tables'][t]
                    cols = {r['name'] for r in conn.execute(f'PRAGMA table_info({t})')}
                    if not isinstance(rows,list) or any(not isinstance(r,dict) or set(r)!=cols for r in rows):
                        raise ValueError('Invalid backup row schema')
                for t in TABLES:
                    where = 'fact_id IN (SELECT id FROM facts WHERE namespace=?)' if t == 'fact_support' else 'namespace=?'
                    if conn.execute(f'SELECT 1 FROM {t} WHERE {where} LIMIT 1', (ns,)).fetchone():
                        raise ValueError('Destination namespace must be empty')
                event_ids = {r['id'] for r in data['tables']['events']}
                fact_ids = {r['id'] for r in data['tables']['facts']}
                procedure_ids = {r['id'] for r in data['tables']['procedures']}
                for t in TABLES:
                    cols = [r['name'] for r in conn.execute(f'PRAGMA table_info({t})')]
                    rows = data['tables'][t]
                    if not isinstance(rows, list):
                        raise ValueError('Invalid backup rows')
                    # Insert triggers construct these ledgers; restore their original revisions.
                    if t in ('consolidation_sources', 'derived_cleanup'):
                        conn.execute(f'DELETE FROM {t} WHERE namespace=?', (ns,))
                    for row in rows:
                        if not isinstance(row, dict) or set(row) != set(cols) or ('namespace' in row and row['namespace'] != ns):
                            raise ValueError('Invalid row schema or scope')
                        if t == 'fact_support' and (row['fact_id'] not in fact_ids or row['event_id'] not in event_ids):
                            raise ValueError('Invalid fact lineage')
                        if t in ('relationship_support','unresolved_relationships','procedure_traces') and row['event_id'] not in event_ids:
                            raise ValueError('Invalid event lineage')
                        if t == 'procedure_traces' and row['procedure_id'] not in procedure_ids:
                            raise ValueError('Invalid procedure lineage')
                        if t == 'derived_support' and (row['summary_id'] not in event_ids or row['source_id'] not in (fact_ids if row['source_type'] == 'fact' else event_ids)):
                            raise ValueError('Invalid summary lineage')
                        values = dict(row)
                        if t == 'jobs' and values['status'] == 'processing':
                            values.update(status='pending', leased_by=None, lease_expires_at=None)
                        conn.execute(f'INSERT INTO {t} ({",".join(cols)}) VALUES ({",".join("?" for _ in cols)})', [values[c] for c in cols])
                conn.execute('INSERT INTO events_fts(id,content,namespace) SELECT id,content,namespace FROM events WHERE namespace=?', (ns,))
                if conn.execute('PRAGMA foreign_key_check').fetchone():
                    raise ValueError('Invalid backup references')
        except sqlite3.IntegrityError as exc:
            raise ValueError('Backup conflicts with existing IDs or constraints') from exc
        finally:
            conn.close()
        return self.stats(ns)

    def delete_session(self, session_id, namespace=None):
        if not isinstance(session_id,str) or not session_id:
            raise ValueError('A nonempty session ID is required')
        ns = self.scope(namespace)
        conn = self.connect()
        try:
            with conn:
                conn.execute('BEGIN IMMEDIATE')
                conn.execute('DELETE FROM events_fts WHERE namespace=? AND id IN (SELECT id FROM events WHERE namespace=? AND session_id=?)',(ns,ns,session_id))
                cursor = conn.execute('DELETE FROM events WHERE namespace=? AND session_id=?',(ns,session_id))
                return cursor.rowcount
        finally:
            conn.close()

    def delete_namespace(self, namespace=None):
        ns = self.scope(namespace)
        conn = self.connect()
        try:
            with conn:
                conn.execute('BEGIN IMMEDIATE')
                conn.execute('DELETE FROM events_fts WHERE namespace=?', (ns,))
                # Child-first deletes, then trigger-created cleanup rows.
                for t in reversed(TABLES):
                    if t in ('relationship_entities','events','facts'):
                        continue
                    where = 'fact_id IN (SELECT id FROM facts WHERE namespace=?)' if t == 'fact_support' else 'namespace=?'
                    conn.execute(f'DELETE FROM {t} WHERE {where}', (ns,))
                for t in ('facts','events','relationship_entities','compact_vectors','consolidation_sources','derived_cleanup'):
                    conn.execute(f'DELETE FROM {t} WHERE namespace=?', (ns,))
                for t in ('graph_edges','graph_nodes','memories'):
                    conn.execute(f'DELETE FROM {t} WHERE namespace=?', (ns,))
        finally:
            conn.close()

    def record_procedure(self, title, steps, namespace=None, prerequisites=None, environment=None, tools=None):
        if not isinstance(title, str) or not title.strip() or not isinstance(steps, list) or not steps or not all(isinstance(s,str) and s.strip() for s in steps):
            raise ValueError('Procedure requires title and nonempty text steps')
        prerequisites = [] if prerequisites is None else prerequisites
        environment = {} if environment is None else environment
        tools = [] if tools is None else tools
        if not isinstance(prerequisites,list) or not isinstance(tools,list) or not isinstance(environment, dict) or not all(isinstance(k,str) and isinstance(v,str) for k,v in environment.items()) or not all(isinstance(x,str) for x in prerequisites + tools):
            raise ValueError('Invalid procedure constraints')
        pid = str(uuid.uuid4())
        conn = self.connect()
        try:
            with conn:
                conn.execute('INSERT INTO procedures VALUES (?,?,?,?,?,?,?,?,?)', (pid,self.scope(namespace),title,json.dumps(steps),json.dumps(prerequisites),json.dumps(environment),json.dumps(tools),'unverified',datetime.now(timezone.utc).isoformat()))
        finally:
            conn.close()
        return pid

    def record_outcome(self, procedure_id, event_id, outcome, namespace=None, evidence='assistant_claim'):
        """Trusted host calls may mark observed or explicitly confirmed results.

        Never expose evidence selection as a model-controlled tool argument.
        """
        if outcome not in ('success','failure') or evidence not in ('assistant_claim','observed','explicit_confirmation'):
            raise ValueError('Invalid outcome or evidence channel')
        ns = self.scope(namespace)
        conn = self.connect()
        try:
            with conn:
                conn.execute('BEGIN IMMEDIATE')
                event = conn.execute("SELECT * FROM events WHERE id=? AND namespace=? AND role!='summary' AND turn_status NOT IN ('deleted','cancelled')", (event_id,ns)).fetchone()
                if event is None or not conn.execute('SELECT 1 FROM procedures WHERE id=? AND namespace=?', (procedure_id,ns)).fetchone():
                    raise ValueError('Missing or out-of-scope trace/procedure')
                if evidence == 'observed' and event['role'] not in ('tool','user'):
                    raise ValueError('Observed outcomes require a host/tool trace')
                if evidence == 'observed' and outcome == 'success' and event['turn_status'] != 'completed':
                    raise ValueError('Observed success requires a completed source trace')
                conn.execute('INSERT INTO procedure_traces VALUES (?,?,?,?,?,?,?)', (str(uuid.uuid4()),ns,procedure_id,event_id,outcome,evidence,datetime.now(timezone.utc).isoformat()))
                if evidence != 'assistant_claim':
                    conn.execute('UPDATE procedures SET status=? WHERE id=? AND namespace=?', ('confirmed' if outcome=='success' else 'failed',procedure_id,ns))
        finally:
            conn.close()

    def retrieve_procedures(self, query, namespace=None, environment=None, tools=None, limit=5):
        if not 1 <= limit <= 100:
            raise ValueError('limit must be 1..100')
        ns = self.scope(namespace)
        conn = self.connect()
        try:
            terms = set(re.findall(r'\w+',query.casefold()))
            results = []
            for r in conn.execute('SELECT * FROM procedures WHERE namespace=?', (ns,)):
                p = dict(r)
                for key in ('steps','prerequisites','environment','tools'):
                    p[key] = json.loads(p[key])
                traces = [dict(t) for t in conn.execute("SELECT t.* FROM procedure_traces t JOIN events e ON e.id=t.event_id AND e.namespace=t.namespace WHERE t.namespace=? AND t.procedure_id=? AND e.turn_status NOT IN ('deleted','cancelled') ORDER BY t.created_at,t.id", (ns,p['id']))]
                verified = [t for t in traces if t['evidence'] != 'assistant_claim']
                p['status'] = ('confirmed' if verified[-1]['outcome']=='success' else 'failed') if verified else 'unverified'
                if p['status']=='unverified' or any((environment or {}).get(k)!=v for k,v in p['environment'].items()) or not set(p['tools']).issubset(tools or []):
                    continue
                score = len(terms & set(re.findall(r'\w+', p['title'].casefold()+' '+' '.join(p['steps']).casefold())))
                if score:
                    p.update(traces=traces, warning=p['status']=='failed', score=score)
                    results.append(p)
            return sorted(results,key=lambda p:(-p['score'],p['id']))[:limit]
        finally:
            conn.close()
