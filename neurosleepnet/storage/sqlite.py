import sqlite3
import json
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
                trust_score REAL DEFAULT 0.5
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
        conn.commit()
        conn.close()

    def store(self, memory_id: str, content: str, created_at: str, metadata: str = "{}", importance: float = 0.0, trust_score: float = 0.5):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO memories (id, content, created_at, metadata, importance, trust_score)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (memory_id, content, created_at, metadata, importance, trust_score))
        conn.commit()
        conn.close()

    def get(self, memory_id: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, content, created_at, metadata, importance, trust_score FROM memories WHERE id = ?
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
                "trust_score": row[5]
            }
        return None

    def delete(self, memory_id: str):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            DELETE FROM memories WHERE id = ?
        ''', (memory_id,))
        conn.commit()
        conn.close()

    def list(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, content, created_at, metadata, importance, trust_score FROM memories
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
                "trust_score": row[5]
            })
        return records
