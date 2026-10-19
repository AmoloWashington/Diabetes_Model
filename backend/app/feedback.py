"""Reviewer feedback: stored in SQLite and, when configured, filed as a GitHub issue.

On hosts with an ephemeral filesystem (such as Render's free tier) the SQLite copy is lost on
restart, so setting FEEDBACK_GITHUB_TOKEN and FEEDBACK_GITHUB_REPO is the durable option: each
submission becomes an issue in that repository. No names or contact details are collected.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

log = logging.getLogger("glucolab.feedback")

ROLE_LABELS = {
    "clinician": "Doctor or clinician", "diabetes_educator": "Diabetes educator or nurse",
    "researcher": "Researcher", "lecturer": "Lecturer or teacher", "student": "Student",
    "person_with_diabetes": "Person living with diabetes", "family_carer": "Family member or carer",
    "other": "Other",
}

_lock = threading.Lock()


def _db_path() -> Path:
    base = Path(os.environ.get("GLUCOLAB_DATA_DIR", Path(__file__).resolve().parents[1] / ".data"))
    base.mkdir(parents=True, exist_ok=True)
    return base / "feedback.db"


def _connect() -> sqlite3.Connection:
    con = sqlite3.connect(_db_path())
    con.execute(
        "CREATE TABLE IF NOT EXISTS feedback (id INTEGER PRIMARY KEY AUTOINCREMENT, created_at REAL NOT NULL, "
        "payload TEXT NOT NULL, github_issue TEXT)"
    )
    return con


def to_markdown(fb: dict) -> str:
    pages = ", ".join(fb.get("pages") or []) or "not specified"
    parts = [
        f"**Role:** {ROLE_LABELS.get(fb['role'], fb['role'])}",
        f"**Usefulness:** {fb['rating']}/5",
        f"**Would use or recommend it:** {fb['would_use']}",
        f"**Pages reviewed:** {pages}",
        f"**Interface language:** {fb.get('language', 'en')}",
    ]
    for key, title in (("accuracy_concerns", "Anything inaccurate, unclear or unsafe"),
                       ("suggestions", "What would make it more useful")):
        if fb.get(key):
            parts.append(f"\n### {title}\n\n{fb[key]}")
    parts.append("\n---\n_Submitted through the GlucoLab feedback page._")
    return "\n".join(parts)


def _file_github_issue(fb: dict) -> str | None:
    token = os.environ.get("FEEDBACK_GITHUB_TOKEN", "").strip()
    repo = os.environ.get("FEEDBACK_GITHUB_REPO", "").strip()
    if not token or not repo or repo.count("/") != 1:
        return None
    body = json.dumps({
        "title": f"[Feedback] {ROLE_LABELS.get(fb['role'], fb['role'])}: {fb['rating']}/5",
        "body": to_markdown(fb),
    }).encode()
    req = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/issues", data=body, method="POST",
        headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
                 "X-GitHub-Api-Version": "2022-11-28", "Content-Type": "application/json",
                 "User-Agent": "GlucoLab-feedback"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read()).get("html_url")
    except (urllib.error.URLError, TimeoutError, ValueError) as e:
        log.warning("Could not file feedback as a GitHub issue: %s", e)
        return None


def submit(fb: dict) -> dict:
    issue = _file_github_issue(fb)
    with _lock, _connect() as con:
        cur = con.execute("INSERT INTO feedback (created_at, payload, github_issue) VALUES (?, ?, ?)",
                          (time.time(), json.dumps(fb), issue))
        fid = cur.lastrowid
    return {"ok": True, "id": fid, "filed_on_github": issue is not None}


def list_all(limit: int = 500) -> list[dict]:
    with _lock, _connect() as con:
        rows = con.execute("SELECT id, created_at, payload, github_issue FROM feedback ORDER BY id DESC LIMIT ?",
                           (limit,)).fetchall()
    return [{"id": r[0], "created_at": r[1], **json.loads(r[2]), "github_issue": r[3]} for r in rows]
