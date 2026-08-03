import sqlite3
import json
from datetime import datetime, timezone
from .base import StorageAdapter

class SQLiteAdapter(StorageAdapter):
    """
    SQLite implementation of StorageAdapter.
    """
    def __init__(self, db_path: str = "neurosleepnet.db"):
        self.db_path = db_path
        self._initialize_db()

    def _initialize_db(self):
        # Create database file and tables
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                metadata TEXT,
                importance REAL DEFAULT 0.0,
                trust_score REAL DEFAULT 0.5,
                embedding TEXT,
                namespace TEXT DEFAULT 'default',
                memory_type TEXT DEFAULT 'semantic',
                access_count INTEGER DEFAULT 0,
                last_accessed_at TEXT
            )
        ''')
        try:
            cursor.execute('ALTER TABLE memories ADD COLUMN metadata TEXT')
        except sqlite3.OperationalError:
            pass
        try:
            cursor.execute('ALTER TABLE memories ADD COLUMN importance REAL DEFAULT 0.0')
        except sqlite3.OperationalError:
            pass
        try:
            cursor.execute('ALTER TABLE memories ADD COLUMN trust_score REAL DEFAULT 0.5')
        except sqlite3.OperationalError:
            pass
        try:
            cursor.execute('ALTER TABLE memories ADD COLUMN embedding TEXT')
        except sqlite3.OperationalError:
            pass
        try:
            cursor.execute("ALTER TABLE memories ADD COLUMN namespace TEXT DEFAULT 'default'")
        except sqlite3.OperationalError:
            pass
        try:
            cursor.execute("ALTER TABLE memories ADD COLUMN memory_type TEXT DEFAULT 'semantic'")
        except sqlite3.OperationalError:
            pass
        try:
            cursor.execute("ALTER TABLE memories ADD COLUMN access_count INTEGER DEFAULT 0")
        except sqlite3.OperationalError:
            pass
        try:
            cursor.execute("ALTER TABLE memories ADD COLUMN last_accessed_at TEXT")
        except sqlite3.OperationalError:
            pass

        cursor.execute('''
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                id UNINDEXED,
                content,
                namespace
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS graph_nodes (
                id TEXT PRIMARY KEY,
                label TEXT NOT NULL,
                name TEXT NOT NULL,
                properties TEXT,
                created_at TEXT NOT NULL
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS graph_edges (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                relation TEXT NOT NULL,
                properties TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(source_id) REFERENCES graph_nodes(id),
                FOREIGN KEY(target_id) REFERENCES graph_nodes(id)
            )
        ''')
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_namespace ON memories(namespace)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type)")
        
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        
        conn.commit()
        conn.close()

    def store_graph_node(self, node_id: str, label: str, name: str, properties: str, created_at: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO graph_nodes (id, label, name, properties, created_at)
            VALUES (?, ?, ?, ?, ?)
        ''', (node_id, label, name, properties, created_at))
        conn.commit()
        conn.close()

    def store_graph_edge(self, edge_id: str, source_id: str, target_id: str, relation: str, properties: str, created_at: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO graph_edges (id, source_id, target_id, relation, properties, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (edge_id, source_id, target_id, relation, properties, created_at))
        conn.commit()
        conn.close()

    def get_graph_node(self, node_id: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT id, label, name, properties, created_at FROM graph_nodes WHERE id = ?', (node_id,))
        row = cursor.fetchone()
        conn.close()
        if row:
            return {"id": row[0], "label": row[1], "name": row[2], "properties": json.loads(row[3]) if row[3] else {}, "created_at": row[4]}
        return None

    def query_graph(self, node_name: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT id, label, properties FROM graph_nodes WHERE name = ?', (node_name,))
        node_row = cursor.fetchone()
        
        if not node_row:
            conn.close()
            return None
            
        node_id = node_row[0]
        result = {
            "node": {"id": node_id, "name": node_name, "label": node_row[1], "properties": json.loads(node_row[2]) if node_row[2] else {}},
            "edges": []
        }
        
        cursor.execute('''
            SELECT e.relation, e.properties, n.id, n.name, n.label, n.properties 
            FROM graph_edges e
            JOIN graph_nodes n ON e.target_id = n.id
            WHERE e.source_id = ?
        ''', (node_id,))
        
        edge_rows = cursor.fetchall()
        for row in edge_rows:
            result["edges"].append({
                "relation": row[0],
                "edge_properties": json.loads(row[1]) if row[1] else {},
                "target": {
                    "id": row[2],
                    "name": row[3],
                    "label": row[4],
                    "properties": json.loads(row[5]) if row[5] else {}
                }
            })
            
        conn.close()
        return result

    def store(self, memory_id: str, content: str, created_at: str, metadata: str = "{}", importance: float = 0.0, trust_score: float = 0.5, embedding: str = "[]", namespace: str = "default", memory_type: str = "semantic"):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO memories (id, content, created_at, metadata, importance, trust_score, embedding, namespace, memory_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (memory_id, content, created_at, metadata, importance, trust_score, embedding, namespace, memory_type))
        
        cursor.execute('''
            INSERT OR REPLACE INTO memories_fts (id, content, namespace)
            VALUES (?, ?, ?)
        ''', (memory_id, content, namespace))
        
        conn.commit()
        conn.close()

    def get(self, memory_id: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, content, created_at, metadata, importance, trust_score, embedding, namespace, memory_type, access_count, last_accessed_at FROM memories WHERE id = ?
        ''', (memory_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                "id": row[0],
                "content": row[1],
                "created_at": row[2],
                "metadata": json.loads(row[3]) if row[3] else {},
                "importance": row[4],
                "trust_score": row[5],
                "embedding": json.loads(row[6]) if row[6] else [],
                "namespace": row[7],
                "memory_type": row[8],
                "access_count": row[9],
                "last_accessed_at": row[10]
            }
        return None

    def delete(self, memory_id: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            DELETE FROM memories WHERE id = ?
        ''', (memory_id,))
        cursor.execute('''
            DELETE FROM memories_fts WHERE id = ?
        ''', (memory_id,))
        conn.commit()
        conn.close()

    def list(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, content, created_at, metadata, importance, trust_score, embedding, namespace, memory_type, access_count, last_accessed_at FROM memories
        ''')
        rows = cursor.fetchall()
        conn.close()
        
        records = []
        for row in rows:
            records.append({
                "id": row[0],
                "content": row[1],
                "created_at": row[2],
                "metadata": json.loads(row[3]) if row[3] else {},
                "importance": row[4],
                "trust_score": row[5],
                "embedding": json.loads(row[6]) if row[6] else [],
                "namespace": row[7],
                "memory_type": row[8],
                "access_count": row[9],
                "last_accessed_at": row[10]
            })
        return records

    def search_keyword(self, query: str, limit: int = 5, namespace: str = None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        # Escape query for FTS5 by wrapping in quotes to treat it as a phrase
        safe_query = '"' + query.replace('"', '') + '"'
        
        if namespace:
            cursor.execute('''
                SELECT m.id, m.content, m.created_at, m.metadata, m.importance, m.trust_score, m.embedding, m.namespace, m.memory_type, m.access_count, m.last_accessed_at
                FROM memories_fts fts
                JOIN memories m ON fts.id = m.id
                WHERE memories_fts MATCH ? AND fts.namespace = ?
                LIMIT ?
            ''', (safe_query, namespace, limit))
        else:
            cursor.execute('''
                SELECT m.id, m.content, m.created_at, m.metadata, m.importance, m.trust_score, m.embedding, m.namespace, m.memory_type, m.access_count, m.last_accessed_at
                FROM memories_fts fts
                JOIN memories m ON fts.id = m.id
                WHERE memories_fts MATCH ? 
                LIMIT ?
            ''', (safe_query, limit))
            
        rows = cursor.fetchall()
        conn.close()
        
        records = []
        for row in rows:
            records.append({
                "id": row[0],
                "content": row[1],
                "created_at": row[2],
                "metadata": json.loads(row[3]) if row[3] else {},
                "importance": row[4],
                "trust_score": row[5],
                "embedding": json.loads(row[6]) if row[6] else [],
                "namespace": row[7],
                "memory_type": row[8],
                "access_count": row[9],
                "last_accessed_at": row[10],
                "score": 1.0
            })
        return records

    def increment_access(self, memory_id: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        now = datetime.now(timezone.utc).isoformat()
        cursor.execute('''
            UPDATE memories SET access_count = access_count + 1, last_accessed_at = ? WHERE id = ?
        ''', (now, memory_id))
        conn.commit()
        conn.close()

    def list_namespace(self, namespace: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, content, created_at, metadata, importance, trust_score, embedding, namespace, memory_type, access_count, last_accessed_at 
            FROM memories
            WHERE namespace = ?
        ''', (namespace,))
        rows = cursor.fetchall()
        conn.close()
        
        records = []
        for row in rows:
            records.append({
                "id": row[0],
                "content": row[1],
                "created_at": row[2],
                "metadata": json.loads(row[3]) if row[3] else {},
                "importance": row[4],
                "trust_score": row[5],
                "embedding": json.loads(row[6]) if row[6] else [],
                "namespace": row[7],
                "memory_type": row[8],
                "access_count": row[9],
                "last_accessed_at": row[10]
            })
        return records

    def timeline(self, namespace: str, memory_type: str = None, limit: int = 20, ascending: bool = True) -> list:
        """Returns chronologically ordered memories, optionally filtered by type."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = (
            "SELECT id, content, created_at, metadata, importance, trust_score, "
            "namespace, memory_type, access_count, last_accessed_at "
            "FROM memories WHERE namespace = ?"
        )
        params = [namespace]

        if memory_type:
            query += " AND memory_type = ?"
            params.append(memory_type)

        order = "ASC" if ascending else "DESC"
        query += f" ORDER BY created_at {order} LIMIT ?"
        params.append(limit)

        cursor.execute(query, tuple(params))
        rows = cursor.fetchall()
        conn.close()

        records = []
        for row in rows:
            records.append({
                "id": row[0],
                "content": row[1],
                "created_at": row[2],
                "metadata": json.loads(row[3]) if row[3] else {},
                "importance": float(row[4]),
                "trust_score": float(row[5]),
                "namespace": row[6],
                "memory_type": row[7],
                "access_count": int(row[8]),
                "last_accessed_at": row[9],
            })
        return records
