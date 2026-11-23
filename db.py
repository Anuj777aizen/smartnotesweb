import hashlib
import hmac
import secrets
import sqlite3
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
            CREATE TABLE IF NOT EXISTS users (
                login_id TEXT PRIMARY KEY,
                full_name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'student',
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS groups (
                group_code TEXT PRIMARY KEY,
                group_name TEXT NOT NULL,
                topic TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS room_members (
                group_code TEXT NOT NULL,
                login_id TEXT NOT NULL,
                full_name TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'student',
                joined_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                PRIMARY KEY (group_code, login_id),
                FOREIGN KEY (group_code) REFERENCES groups(group_code)
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


def _hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
    return f"{salt.hex()}:{digest.hex()}"


def _verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt_hex, digest_hex = stored_hash.split(":", 1)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(digest_hex)
        candidate = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)
        return hmac.compare_digest(candidate, expected)
    except Exception:
        return False


def register_user(login_id: str, full_name: str, password: str) -> tuple[bool, str]:
    """Create a new user account. The very first registered user becomes admin."""
    clean_id = login_id.strip().lower()
    clean_name = full_name.strip()

    if len(clean_id) < 3 or " " in clean_id:
        return False, "Login ID must be at least 3 characters with no spaces."
    if not clean_name:
        return False, "Please enter your full name."
    if len(password) < 6:
        return False, "Password must be at least 6 characters long."

    now = datetime.now(timezone.utc).isoformat()
    pw_hash = _hash_password(password)

    with get_conn() as conn:
        user_count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        role = "admin" if user_count == 0 else "student"
        try:
            conn.execute(
                "INSERT INTO users (login_id, full_name, password_hash, role, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (clean_id, clean_name, pw_hash, role, now),
            )
            return True, f"Account created ({role}). You can log in now."
        except sqlite3.IntegrityError:
            return False, "This Login ID is already taken."


def authenticate_user(login_id: str, password: str) -> dict | None:
    """Verify credentials and return user info dict, or None if invalid."""
    clean_id = login_id.strip().lower()
    with get_conn() as conn:
        row = conn.execute(
            "SELECT login_id, full_name, password_hash, role FROM users WHERE login_id = ?",
            (clean_id,),
        ).fetchone()

    if row and _verify_password(password, row["password_hash"]):
        return {
            "login_id": row["login_id"],
            "full_name": row["full_name"],
            "role": row["role"],
        }
    return None


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


def record_room_member(group_code: str, user: dict | None):
    """Record a user joining a room, or update their last active timestamp."""
    if not group_code or not user or not user.get("login_id"):
        return

    now = datetime.now(timezone.utc).isoformat()
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO room_members (group_code, login_id, full_name, role, joined_at, last_seen_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(group_code, login_id) DO UPDATE SET
                full_name = excluded.full_name,
                role = excluded.role,
                last_seen_at = excluded.last_seen_at
            """,
            (
                group_code.upper(),
                user["login_id"].strip().lower(),
                user.get("full_name", "").strip() or user["login_id"],
                user.get("role", "student"),
                now,
                now,
            ),
        )


def get_room_members(group_code: str):
    """Return all users who have joined this room along with their submission counts."""
    code = group_code.upper()
    with get_conn() as conn:
        return conn.execute(
            """
            SELECT
                m.login_id,
                m.full_name,
                m.role,
                m.joined_at,
                m.last_seen_at,
                (
                    SELECT COUNT(*)
                    FROM submissions s
                    WHERE s.group_code = m.group_code
                      AND LOWER(s.member_name) LIKE '%(' || LOWER(m.login_id) || ')'
                ) AS notes_submitted
            FROM room_members m
            WHERE m.group_code = ?
            ORDER BY m.joined_at ASC
            """,
            (code,),
        ).fetchall()


def list_all_groups():
    """Return all rooms with member counts, submission counts, and master note status."""
    with get_conn() as conn:
        return conn.execute("""
            SELECT
                g.group_code,
                g.group_name,
                g.topic,
                g.created_at,
                (SELECT COUNT(*) FROM submissions s WHERE s.group_code = g.group_code) AS submission_count,
                (SELECT COUNT(*) FROM room_members rm WHERE rm.group_code = g.group_code) AS member_count,
                CASE WHEN EXISTS (
                    SELECT 1 FROM master_notes m WHERE m.group_code = g.group_code
                ) THEN 1 ELSE 0 END AS has_master
            FROM groups g
            ORDER BY g.created_at DESC
        """).fetchall()


def delete_group(group_code: str):
    """Delete a room along with its members, submissions, and master note."""
    code = group_code.upper()
    with get_conn() as conn:
        conn.execute("DELETE FROM room_members WHERE group_code = ?", (code,))
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