# core/els_bridge.py
# Synthesus 2.0 - ELS Bridge (Experience-Learning-Synthesis)
# Captures interactions, extracts candidate patterns, and integrates approved ones

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
class Interaction:
    id: str
    character_id: str
    user_input: str
    character_response: str
    outcome_success: bool
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidatePattern:
    id: str
    character_id: str
    trigger: str
    response_template: str
    score: float = 0.5
    status: str = "pending"   # "pending" | "approved" | "rejected" | "integrated"
    source_interaction_id: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ---------------------------------------------------------------------------
# ELSBridge
# ---------------------------------------------------------------------------

class ELSBridge:
    """
    Experience-Learning-Synthesis Bridge.

    Responsibilities:
    - Capture raw interactions to SQLite
    - Score and surface candidate patterns from successful interactions
    - Persist candidates to JSON for human/automated review
    - Integrate approved candidates back into the pattern store
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS interactions (
        id TEXT PRIMARY KEY,
        character_id TEXT NOT NULL,
        user_input TEXT NOT NULL,
        character_response TEXT NOT NULL,
        outcome_success INTEGER NOT NULL,
        timestamp TEXT NOT NULL,
        metadata TEXT DEFAULT '{}'
    );
    CREATE TABLE IF NOT EXISTS candidate_patterns (
        id TEXT PRIMARY KEY,
        character_id TEXT NOT NULL,
        trigger TEXT NOT NULL,
        response_template TEXT NOT NULL,
        score REAL DEFAULT 0.5,
        status TEXT DEFAULT 'pending',
        source_interaction_id TEXT,
        created_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_int_character ON interactions(character_id);
    CREATE INDEX IF NOT EXISTS idx_cand_character ON candidate_patterns(character_id);
    CREATE INDEX IF NOT EXISTS idx_cand_status ON candidate_patterns(status);
    """

    def __init__(
        self,
        db_path: str = "data/interactions.db",
        patterns_path: str = "data/candidate_patterns.json",
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.patterns_path = Path(patterns_path)
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
    # Capture
    # ------------------------------------------------------------------

    def capture(
        self,
        character_id: str,
        user_input: str,
        character_response: str,
        outcome_success: bool,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Interaction:
        interaction = Interaction(
            id=str(uuid.uuid4()),
            character_id=character_id,
            user_input=user_input,
            character_response=character_response,
            outcome_success=outcome_success,
            metadata=metadata or {},
        )
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO interactions
                (id, character_id, user_input, character_response,
                 outcome_success, timestamp, metadata)
                VALUES (?,?,?,?,?,?,?)
                """,
                (
                    interaction.id, interaction.character_id,
                    interaction.user_input, interaction.character_response,
                    int(interaction.outcome_success), interaction.timestamp,
                    json.dumps(interaction.metadata),
                ),
            )
        if outcome_success:
            self._extract_candidate(interaction)
        return interaction

    # ------------------------------------------------------------------
    # Candidate extraction
    # ------------------------------------------------------------------

    def _extract_candidate(self, interaction: Interaction) -> None:
        words = interaction.user_input.lower().split()
        stop = {"the", "a", "an", "is", "it", "of", "in", "to", "and", "or"}
        sig = [w for w in words if w not in stop][:5]
        if len(sig) < 2:
            return
        trigger = " ".join(sig)
        template = interaction.character_response[:250].strip()
        score = 0.5 + (0.1 if interaction.outcome_success else 0.0)

        cid = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT INTO candidate_patterns
                (id, character_id, trigger, response_template,
                 score, status, source_interaction_id, created_at)
                VALUES (?,?,?,?,?,?,?,?)
                """,
                (cid, interaction.character_id, trigger, template,
                 score, "pending", interaction.id, now),
            )

        # Also append to JSON file for easy review
        existing: List[Dict] = []
        if self.patterns_path.exists():
            try:
                existing = json.loads(self.patterns_path.read_text())
            except Exception:
                existing = []
        existing.append({
            "id": cid,
            "character_id": interaction.character_id,
            "trigger": trigger,
            "response_template": template,
            "score": score,
            "status": "pending",
            "source_interaction_id": interaction.id,
            "created_at": now,
        })
        self.patterns_path.write_text(json.dumps(existing, indent=2))

    # ------------------------------------------------------------------
    # Review & Integration
    # ------------------------------------------------------------------

    def get_candidates(
        self,
        character_id: str,
        status: str = "pending",
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM candidate_patterns "
                "WHERE character_id=? AND status=? LIMIT ?",
                (character_id, status, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def integrate_patterns(
        self,
        character_id: str,
        approved: List[Dict[str, Any]],
    ) -> int:
        """Mark approved candidates as integrated and write to patterns JSON."""
        if not approved:
            return 0

        patterns_file = Path("data/patterns.json")
        existing: List[Dict] = []
        if patterns_file.exists():
            try:
                existing = json.loads(patterns_file.read_text())
            except Exception:
                existing = []

        added = 0
        for a in approved:
            existing.append({
                "character_id": character_id,
                "pattern_type": "reasoning",
                "trigger": a["trigger"],
                "response_template": a["response_template"],
                "weight": a.get("score", 0.5),
            })
            added += 1

        patterns_file.write_text(json.dumps(existing, indent=2))

        # Mark as integrated in DB
        ids = [a["id"] for a in approved]
        with self._get_conn() as conn:
            conn.executemany(
                "UPDATE candidate_patterns SET status='integrated' WHERE id=?",
                [(i,) for i in ids],
            )
        return added

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self, character_id: str) -> Dict[str, Any]:
        with self._get_conn() as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM interactions WHERE character_id=?",
                (character_id,),
            ).fetchone()[0]
            success = conn.execute(
                "SELECT COUNT(*) FROM interactions "
                "WHERE character_id=? AND outcome_success=1",
                (character_id,),
            ).fetchone()[0]
            pending = conn.execute(
                "SELECT COUNT(*) FROM candidate_patterns "
                "WHERE character_id=? AND status='pending'",
                (character_id,),
            ).fetchone()[0]
        return {
            "total_interactions": total,
            "success_rate": round(success / total, 3) if total else 0,
            "pending_candidates": pending,
        }
