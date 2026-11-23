import sqlite3
import secrets
import string
from datetime import datetime, timezone

DB_PATH = "class_notes.db"


def get_conn():
    conn = sqlite3.connect(DB_PATH, timeout=15)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with get_conn() as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS groups (
                group_code TEXT PRIMARY KEY,
                group_name TEXT NOT NULL,
                topic TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS submissions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                group_code TEXT NOT NULL,
                member_name TEXT NOT NULL,
                extracted_text TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (group_code) REFERENCES groups(group_code)
            );

            CREATE TABLE IF NOT EXISTS master_notes (
                group_code TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                source_count INTEGER NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (group_code) REFERENCES groups(group_code)
            );
        """)


def _generate_unique_code(conn) -> str:
    alphabet = string.ascii_uppercase + string.digits
    while True:
        code = "".join(secrets.choice(alphabet) for _ in range(6))
        exists = conn.execute(
            "SELECT 1 FROM groups WHERE group_code = ?", (code,)
        ).fetchone()
        if not exists:
            return code


def create_group(group_name: str, topic: str) -> str:
    now = datetime.now(timezone.utc).isoformat()
    with get_conn() as conn:
        code = _generate_unique_code(conn)
        conn.execute(
            "INSERT INTO groups (group_code, group_name, topic, created_at) VALUES (?, ?, ?, ?)",
            (code, group_name.strip(), topic.strip(), now),
        )
    return code


def create_multiple_groups(group_name: str, topics: list[str]) -> list[dict]:
    """Create multiple topic rooms under the same class/group name at once."""
    clean_group = group_name.strip()
    clean_topics = [t.strip() for t in topics if t and t.strip()]
    if not clean_group or not clean_topics:
        return []

    created = []
    now = datetime.now(timezone.utc).isoformat()
    with get_conn() as conn:
        for topic in clean_topics:
            code = _generate_unique_code(conn)
            conn.execute(
                "INSERT INTO groups (group_code, group_name, topic, created_at) VALUES (?, ?, ?, ?)",
                (code, clean_group, topic, now),
            )
            created.append(
                {"group_code": code, "group_name": clean_group, "topic": topic}
            )
    return created


def list_all_groups():
    """Return all rooms along with submission counts and master note status."""
    with get_conn() as conn:
        return conn.execute("""
            SELECT
                g.group_code,
                g.group_name,
                g.topic,
                g.created_at,
                COUNT(s.id) AS submission_count,
                CASE WHEN m.group_code IS NOT NULL THEN 1 ELSE 0 END AS has_master
            FROM groups g
            LEFT JOIN submissions s ON g.group_code = s.group_code
            LEFT JOIN master_notes m ON g.group_code = m.group_code
            GROUP BY g.group_code, g.group_name, g.topic, g.created_at, m.group_code
            ORDER BY g.created_at DESC
        """).fetchall()


def delete_group(group_code: str):
    """Delete a room along with all of its submissions and master note."""
    code = group_code.upper()
    with get_conn() as conn:
        conn.execute("DELETE FROM master_notes WHERE group_code = ?", (code,))
        conn.execute("DELETE FROM submissions WHERE group_code = ?", (code,))
        conn.execute("DELETE FROM groups WHERE group_code = ?", (code,))


def add_submission(group_code: str, member_name: str, extracted_text: str):
    now = datetime.now(timezone.utc).isoformat()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO submissions (group_code, member_name, extracted_text, created_at) VALUES (?, ?, ?, ?)",
            (group_code.upper(), member_name.strip(), extracted_text.strip(), now),
        )


def get_group_details(group_code: str):
    code = group_code.upper()
    with get_conn() as conn:
        group = conn.execute(
            "SELECT * FROM groups WHERE group_code = ?", (code,)
        ).fetchone()
        submissions = conn.execute(
            "SELECT id, member_name, extracted_text, created_at FROM submissions WHERE group_code = ? ORDER BY id ASC",
            (code,),
        ).fetchall()
    return group, submissions


def save_master_note(group_code: str, content: str, source_count: int):
    now = datetime.now(timezone.utc).isoformat()
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO master_notes (group_code, content, source_count, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(group_code) DO UPDATE SET
                content = excluded.content,
                source_count = excluded.source_count,
                updated_at = excluded.updated_at
            """,
            (group_code.upper(), content, source_count, now),
        )


def get_master_note(group_code: str):
    with get_conn() as conn:
        return conn.execute(
            "SELECT * FROM master_notes WHERE group_code = ?", (group_code.upper(),)
        ).fetchone()