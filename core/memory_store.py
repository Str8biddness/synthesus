# core/memory_store.py
# Synthesus 2.0 - Memory Store
# Persistent episodic and semantic memory for synthetic characters

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class Memory:
    id: str
    character_id: str
    memory_type: str       # "episodic" | "semantic" | "working"
    content: str
    importance: float = 0.5   # [0.0, 1.0]
    access_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_accessed: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["tags"] = json.dumps(self.tags)
        d["metadata"] = json.dumps(self.metadata)
        return d

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Memory":
        d = dict(row)
        d["tags"] = json.loads(d.get("tags") or "[]")
        d["metadata"] = json.loads(d.get("metadata") or "{}")
        return cls(**d)


# ---------------------------------------------------------------------------
# MemoryStore
# ---------------------------------------------------------------------------

class MemoryStore:
    """SQLite-backed episodic and semantic memory store for Synthesus characters."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS memories (
        id TEXT PRIMARY KEY,
        character_id TEXT NOT NULL,
        memory_type TEXT NOT NULL,
        content TEXT NOT NULL,
        importance REAL DEFAULT 0.5,
        access_count INTEGER DEFAULT 0,
        created_at TEXT NOT NULL,
        last_accessed TEXT NOT NULL,
        tags TEXT DEFAULT '[]',
        metadata TEXT DEFAULT '{}'
    );
    CREATE INDEX IF NOT EXISTS idx_mem_character ON memories(character_id);
    CREATE INDEX IF NOT EXISTS idx_mem_type ON memories(memory_type);
    CREATE INDEX IF NOT EXISTS idx_mem_importance ON memories(importance DESC);
    """

    def __init__(self, db_path: str = "data/memory.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        with self._get_conn() as conn:
            conn.executescript(self.SCHEMA)

    @contextmanager
    def _get_conn(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def store(
        self,
        character_id: str,
        content: str,
        memory_type: str = "episodic",
        importance: float = 0.5,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Memory:
        now = datetime.now(timezone.utc).isoformat()
        mem = Memory(
            id=str(uuid.uuid4()),
            character_id=character_id,
            memory_type=memory_type,
            content=content,
            importance=max(0.0, min(1.0, importance)),
            created_at=now,
            last_accessed=now,
            tags=tags or [],
            metadata=metadata or {},
        )
        d = mem.to_dict()
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO memories
                (id, character_id, memory_type, content, importance,
                 access_count, created_at, last_accessed, tags, metadata)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    d["id"], d["character_id"], d["memory_type"], d["content"],
                    d["importance"], d["access_count"], d["created_at"],
                    d["last_accessed"], d["tags"], d["metadata"],
                ),
            )
        return mem

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def recall(
        self,
        character_id: str,
        query: str,
        memory_type: Optional[str] = None,
        top_k: int = 10,
        min_importance: float = 0.0,
    ) -> List[Memory]:
        """Simple keyword-based recall. Returns most important matching memories."""
        all_mems = self._fetch_all(character_id, memory_type, min_importance)
        if not query.strip():
            return all_mems[:top_k]

        tokens = set(query.lower().split())
        scored = [
            (m, self._score(m, tokens))
            for m in all_mems
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        results = [m for m, _ in scored if _ > 0][:top_k]

        # Touch access stats
        now = datetime.now(timezone.utc).isoformat()
        ids = [m.id for m in results]
        if ids:
            with self._get_conn() as conn:
                for mid in ids:
                    conn.execute(
                        "UPDATE memories SET access_count = access_count + 1, last_accessed=? WHERE id=?",
                        (now, mid),
                    )
        return results

    def get(self, memory_id: str) -> Optional[Memory]:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT * FROM memories WHERE id=?", (memory_id,)
            ).fetchone()
        return Memory.from_row(row) if row else None

    def list(
        self,
        character_id: str,
        memory_type: Optional[str] = None,
        limit: int = 50,
    ) -> List[Memory]:
        return self._fetch_all(character_id, memory_type, limit=limit)

    # ------------------------------------------------------------------
    # Delete / Consolidate
    # ------------------------------------------------------------------

    def forget(self, memory_id: str) -> bool:
        with self._get_conn() as conn:
            cur = conn.execute("DELETE FROM memories WHERE id=?", (memory_id,))
        return cur.rowcount > 0

    def prune(
        self,
        character_id: str,
        keep_top: int = 200,
        min_importance: float = 0.1,
    ) -> int:
        """Remove low-importance memories beyond keep_top threshold."""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT id FROM memories WHERE character_id=? AND importance>=? "
                "ORDER BY importance DESC, access_count DESC",
                (character_id, min_importance),
            ).fetchall()
            keep_ids = {r["id"] for r in rows[:keep_top]}
            all_ids = {r["id"] for r in conn.execute(
                "SELECT id FROM memories WHERE character_id=?", (character_id,)
            ).fetchall()}
            remove_ids = all_ids - keep_ids
            if remove_ids:
                conn.executemany(
                    "DELETE FROM memories WHERE id=?",
                    [(i,) for i in remove_ids],
                )
        return len(remove_ids)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self, character_id: str) -> Dict[str, Any]:
        with self._get_conn() as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM memories WHERE character_id=?",
                (character_id,),
            ).fetchone()[0]
            by_type = conn.execute(
                "SELECT memory_type, COUNT(*) as cnt FROM memories "
                "WHERE character_id=? GROUP BY memory_type",
                (character_id,),
            ).fetchall()
        return {
            "total": total,
            "by_type": {r["memory_type"]: r["cnt"] for r in by_type},
        }

    # ------------------------------------------------------------------
    # Private utilities
    # ------------------------------------------------------------------

    def _fetch_all(
        self,
        character_id: str,
        memory_type: Optional[str] = None,
        min_importance: float = 0.0,
        limit: int = 1000,
    ) -> List[Memory]:
        with self._get_conn() as conn:
            if memory_type:
                rows = conn.execute(
                    "SELECT * FROM memories WHERE character_id=? AND memory_type=? "
                    "AND importance>=? ORDER BY importance DESC LIMIT ?",
                    (character_id, memory_type, min_importance, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM memories WHERE character_id=? AND importance>=? "
                    "ORDER BY importance DESC LIMIT ?",
                    (character_id, min_importance, limit),
                ).fetchall()
        return [Memory.from_row(r) for r in rows]

    @staticmethod
    def _score(mem: Memory, tokens: set) -> float:
        content_tokens = set(mem.content.lower().split())
        overlap = len(tokens & content_tokens)
        tag_overlap = sum(1 for t in mem.tags if any(tok in t.lower() for tok in tokens))
        return overlap * 1.0 + tag_overlap * 0.5 + mem.importance * 0.2
