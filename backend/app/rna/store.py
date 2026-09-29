"""SQLite persistence for RNA sequences and 3D structures (stdlib only)."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path

DEFAULT_DB = Path(os.environ.get("GLUCOLAB_DATA_DIR", Path(__file__).resolve().parents[2] / ".data")) / "glucolab.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS rna_sequences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    sequence TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    length INTEGER NOT NULL,
    source TEXT NOT NULL DEFAULT 'upload',
    tags TEXT NOT NULL DEFAULT '[]',
    created_at REAL NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_rna_seq_sha ON rna_sequences(sha256);
CREATE TABLE IF NOT EXISTS rna_structures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sequence_id INTEGER REFERENCES rna_sequences(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    kind TEXT NOT NULL,            -- 'predicted' | 'uploaded' | 'pdb'
    method TEXT NOT NULL,
    pdb TEXT NOT NULL,
    confidence TEXT NOT NULL DEFAULT '[]',
    mean_confidence REAL,
    meta TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_struct_seq ON rna_structures(sequence_id);
"""


class Store:
    def __init__(self, path: Path | str = DEFAULT_DB):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        if not self.path.exists():  # database removed while running: recreate schema
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(self.path) as c0:
                c0.executescript(SCHEMA)
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------ sequences
    def add_sequence(self, name: str, sequence: str, description: str = "", source: str = "upload",
                     tags: list[str] | None = None) -> tuple[dict, bool]:
        """Insert; returns (row, created). Identical sequences are de-duplicated by SHA-256."""
        sha = hashlib.sha256(sequence.encode()).hexdigest()
        with self._lock, self._conn() as c:
            row = c.execute("SELECT * FROM rna_sequences WHERE sha256 = ?", (sha,)).fetchone()
            if row:
                return self._seq(row), False
            cur = c.execute(
                "INSERT INTO rna_sequences (name, description, sequence, sha256, length, source, tags, created_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (name, description, sequence, sha, len(sequence), source, json.dumps(tags or []), time.time()),
            )
            row = c.execute("SELECT * FROM rna_sequences WHERE id = ?", (cur.lastrowid,)).fetchone()
            return self._seq(row), True

    def list_sequences(self, q: str = "", limit: int = 200, offset: int = 0) -> tuple[list[dict], int]:
        like = f"%{q}%"
        with self._conn() as c:
            total = c.execute(
                "SELECT COUNT(*) FROM rna_sequences WHERE name LIKE ? OR description LIKE ?", (like, like)
            ).fetchone()[0]
            rows = c.execute(
                "SELECT * FROM rna_sequences WHERE name LIKE ? OR description LIKE ? "
                "ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (like, like, limit, offset),
            ).fetchall()
        return [self._seq(r) for r in rows], total

    def get_sequence(self, sid: int) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM rna_sequences WHERE id = ?", (sid,)).fetchone()
        return self._seq(row) if row else None

    def update_sequence(self, sid: int, name: str | None = None, description: str | None = None,
                        tags: list[str] | None = None) -> dict | None:
        cur = self.get_sequence(sid)
        if not cur:
            return None
        with self._lock, self._conn() as c:
            c.execute(
                "UPDATE rna_sequences SET name = ?, description = ?, tags = ? WHERE id = ?",
                (name if name is not None else cur["name"],
                 description if description is not None else cur["description"],
                 json.dumps(tags if tags is not None else cur["tags"]), sid),
            )
        return self.get_sequence(sid)

    def delete_sequence(self, sid: int) -> bool:
        with self._lock, self._conn() as c:
            return c.execute("DELETE FROM rna_sequences WHERE id = ?", (sid,)).rowcount > 0

    # ------------------------------------------------------------ structures
    def add_structure(self, sequence_id: int | None, name: str, kind: str, method: str, pdb: str,
                      confidence: list[float] | None = None, meta: dict | None = None) -> dict:
        conf = confidence or []
        mean = sum(conf) / len(conf) if conf else None
        with self._lock, self._conn() as c:
            cur = c.execute(
                "INSERT INTO rna_structures (sequence_id, name, kind, method, pdb, confidence, mean_confidence, meta, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (sequence_id, name, kind, method, pdb, json.dumps(conf), mean, json.dumps(meta or {}), time.time()),
            )
            row = c.execute("SELECT * FROM rna_structures WHERE id = ?", (cur.lastrowid,)).fetchone()
        return self._struct(row, include_pdb=True)

    def list_structures(self, sequence_id: int | None = None) -> list[dict]:
        with self._conn() as c:
            if sequence_id is None:
                rows = c.execute("SELECT * FROM rna_structures ORDER BY created_at DESC LIMIT 500").fetchall()
            else:
                rows = c.execute("SELECT * FROM rna_structures WHERE sequence_id = ? ORDER BY created_at DESC",
                                 (sequence_id,)).fetchall()
        return [self._struct(r) for r in rows]

    def get_structure(self, stid: int) -> dict | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM rna_structures WHERE id = ?", (stid,)).fetchone()
        return self._struct(row, include_pdb=True) if row else None

    def delete_structure(self, stid: int) -> bool:
        with self._lock, self._conn() as c:
            return c.execute("DELETE FROM rna_structures WHERE id = ?", (stid,)).rowcount > 0

    # ------------------------------------------------------------ rows
    @staticmethod
    def _seq(r: sqlite3.Row) -> dict:
        return {
            "id": r["id"], "name": r["name"], "description": r["description"], "sequence": r["sequence"],
            "length": r["length"], "source": r["source"], "tags": json.loads(r["tags"]),
            "sha256": r["sha256"], "created_at": r["created_at"],
        }

    @staticmethod
    def _struct(r: sqlite3.Row, include_pdb: bool = False) -> dict:
        d = {
            "id": r["id"], "sequence_id": r["sequence_id"], "name": r["name"], "kind": r["kind"],
            "method": r["method"], "mean_confidence": r["mean_confidence"], "meta": json.loads(r["meta"]),
            "created_at": r["created_at"],
        }
        if include_pdb:
            d["pdb"] = r["pdb"]
            d["confidence"] = json.loads(r["confidence"])
        return d


_store: Store | None = None
_store_lock = threading.Lock()


def get_store() -> Store:
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = Store()
    return _store


def set_store(store: Store) -> None:
    """Used by tests to point at a temporary database."""
    global _store
    _store = store
