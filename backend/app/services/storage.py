import sqlite3
import json
import hashlib
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional, List

from ..models import (
    LogEntry, ExtractionResult, ImageAnalysisResult, FollowupExchange,
    MedicationEntry, MedicationLogEntry,
    AmbientSession, AmbientEvent, AmbientChunk,
    ScheduledCheckin, CheckinType, UserProfile,
)


def _utcnow() -> datetime:
    """Return current UTC time as naive datetime (compatible with existing DB data)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


DB_PATH = Path(__file__).parent.parent.parent / "data" / "symptom_logs.db"
_db_initialized = False
_db_lock = threading.Lock()


@contextmanager
def _get_connection(timeout: float = 10.0):
    """Context manager that provides a SQLite connection with automatic commit/rollback."""
    _ensure_db()
    conn = sqlite3.connect(str(DB_PATH), timeout=timeout)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

LOG_COLUMNS = (
    "id, user_id, recorded_at, transcript, description, photo_b64, "
    "extracted_json, followup_question, followup_answer, followup_answered_at, deleted, "
    "image_analysis_json, contact_clinician_note, contact_clinician_reason, followup_response, "
    "followup_exchanges_json"
)
MED_COLUMNS = (
    "id, user_id, name, dose, frequency, reason, start_date, end_date, "
    "notes, is_active, created_at, updated_at, reminder_enabled, reminder_times"
)
MED_LOG_COLUMNS = (
    "id, user_id, medication_id, medication_name, dose_taken, "
    "taken_at, notes, symptom_log_id"
)
AMBIENT_SESSION_COLUMNS = (
    "id, user_id, session_type, status, label, started_at, ended_at, "
    "chunk_count, total_duration_seconds, created_at, updated_at"
)
AMBIENT_EVENT_COLUMNS = (
    "id, session_id, user_id, event_type, timestamp, duration_seconds, "
    "confidence, metadata_json, chunk_index"
)
CHECKIN_COLUMNS = (
    "id, user_id, checkin_type, scheduled_for, message, context_json, "
    "created_at, triggered, dismissed, response, responded_at"
)
PROFILE_COLUMNS = (
    "user_id, name, age, gender, "
    "conditions_json, allergies_json, regular_medications_json, "
    "surgeries_json, family_history_json, social_history_json, patterns_json, "
    "health_summary, created_at, updated_at, intake_completed, intake_questions_asked, "
    "intake_answered_json, intake_last_question_id, intake_started_at, intake_completed_at, "
    "intake_pending_raw_json"
)


# ---------------------------------------------------------------------------
# Schema migration system
# ---------------------------------------------------------------------------
_SCHEMA_VERSION = 8  # Bump when adding new migrations

# Each migration version maps to a list of (table, column_definition) tuples.
# These run only for databases created before the column was added.
_MIGRATIONS: dict[int, list[tuple[str, str]]] = {
    1: [
        ("logs", "description TEXT"),
        ("logs", "photo_b64 TEXT"),
        ("logs", "deleted INTEGER DEFAULT 0"),
        ("logs", "image_analysis_json TEXT"),
        ("logs", "contact_clinician_note TEXT"),
        ("logs", "contact_clinician_reason TEXT"),
    ],
    2: [
        ("ambient_sessions", "result_json TEXT"),
        ("scheduled_checkins", "dedupe_key TEXT"),
    ],
    3: [
        ("user_profiles", "surgeries_json TEXT"),
        ("user_profiles", "family_history_json TEXT"),
        ("user_profiles", "social_history_json TEXT"),
        ("user_profiles", "intake_completed INTEGER DEFAULT 0"),
        ("user_profiles", "intake_questions_asked INTEGER DEFAULT 0"),
        ("user_profiles", "intake_answered_json TEXT"),
        ("user_profiles", "intake_last_question_id TEXT"),
        ("user_profiles", "intake_started_at TEXT"),
        ("user_profiles", "intake_completed_at TEXT"),
    ],
    4: [
        ("medications", "reminder_enabled INTEGER DEFAULT 0"),
        ("medications", "reminder_times TEXT"),
    ],
    5: [
        ("user_profiles", "intake_pending_raw_json TEXT"),
    ],
    6: [
        ("logs", "followup_response TEXT"),
    ],
    7: [
        ("logs", "followup_exchanges_json TEXT DEFAULT '[]'"),
    ],
    8: [
        ("user_profiles", "name TEXT"),
        ("user_profiles", "age INTEGER"),
        ("user_profiles", "gender TEXT"),
    ],
}


def _add_column(cursor, table: str, column_def: str) -> None:
    """Add a column to a table, silently skipping if it already exists."""
    try:
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column_def}")
    except sqlite3.OperationalError:
        pass


def _get_schema_version(cursor) -> int:
    """Return the current schema version, or 0 for a pre-versioning database."""
    try:
        row = cursor.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
        return row[0] if row else 0
    except sqlite3.OperationalError:
        return 0


def _set_schema_version(cursor, version: int) -> None:
    """Upsert the schema version tracker."""
    cursor.execute(
        "INSERT INTO schema_version (id, version) VALUES (1, ?) "
        "ON CONFLICT(id) DO UPDATE SET version = excluded.version",
        (version,),
    )


def _run_migrations(cursor) -> None:
    """Apply any pending schema migrations."""
    cursor.execute(
        "CREATE TABLE IF NOT EXISTS schema_version "
        "(id INTEGER PRIMARY KEY, version INTEGER NOT NULL)"
    )
    current = _get_schema_version(cursor)
    for ver in sorted(_MIGRATIONS):
        if current < ver:
            for table, col_def in _MIGRATIONS[ver]:
                _add_column(cursor, table, col_def)
    _set_schema_version(cursor, _SCHEMA_VERSION)


def _ensure_db():
    """Create database and tables if they don't exist. Safe to call multiple times."""
    global _db_initialized
    if _db_initialized:
        return
    with _db_lock:
        if _db_initialized:  # Double-check after acquiring lock
            return
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()

        # --- Core tables (full schema for fresh databases) ---

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS logs (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                recorded_at TEXT NOT NULL,
                transcript TEXT NOT NULL,
                description TEXT,
                photo_b64 TEXT,
                extracted_json TEXT NOT NULL,
                followup_question TEXT,
                followup_answer TEXT,
                followup_answered_at TEXT,
                deleted INTEGER DEFAULT 0,
                image_analysis_json TEXT,
                contact_clinician_note TEXT,
                contact_clinician_reason TEXT,
                followup_response TEXT,
                followup_exchanges_json TEXT DEFAULT '[]'
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON logs(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_recorded_at ON logs(recorded_at)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_deleted ON logs(user_id, deleted)")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS medications (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                dose TEXT,
                frequency TEXT,
                reason TEXT,
                start_date TEXT,
                end_date TEXT,
                notes TEXT,
                is_active INTEGER DEFAULT 1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                reminder_enabled INTEGER DEFAULT 0,
                reminder_times TEXT
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_meds_user_id ON medications(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_meds_user_active ON medications(user_id, is_active)")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS medication_logs (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                medication_id TEXT,
                medication_name TEXT NOT NULL,
                dose_taken TEXT,
                taken_at TEXT NOT NULL,
                notes TEXT,
                symptom_log_id TEXT
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_medlogs_user_id ON medication_logs(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_medlogs_taken_at ON medication_logs(taken_at)")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ambient_sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                session_type TEXT NOT NULL,
                status TEXT NOT NULL,
                label TEXT,
                started_at TEXT NOT NULL,
                ended_at TEXT,
                chunk_count INTEGER DEFAULT 0,
                total_duration_seconds REAL DEFAULT 0,
                result_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ambient_user_id ON ambient_sessions(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_ambient_status ON ambient_sessions(status)")
        cursor.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_ambient_user_active
            ON ambient_sessions(user_id) WHERE status = 'active'
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ambient_events (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                duration_seconds REAL,
                confidence REAL NOT NULL,
                metadata_json TEXT,
                chunk_index INTEGER NOT NULL
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_events_session ON ambient_events(session_id)")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ambient_chunks (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                duration_seconds REAL NOT NULL,
                uploaded_at TEXT NOT NULL,
                processed INTEGER DEFAULT 0,
                events_detected INTEGER DEFAULT 0
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_chunks_session ON ambient_chunks(session_id)")
        cursor.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS idx_chunks_session_index "
            "ON ambient_chunks(session_id, chunk_index)"
        )

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scheduled_checkins (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                checkin_type TEXT NOT NULL,
                scheduled_for TEXT NOT NULL,
                message TEXT NOT NULL,
                context_json TEXT,
                dedupe_key TEXT,
                created_at TEXT NOT NULL,
                triggered INTEGER DEFAULT 0,
                dismissed INTEGER DEFAULT 0,
                response TEXT,
                responded_at TEXT
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_checkins_user ON scheduled_checkins(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_checkins_scheduled ON scheduled_checkins(scheduled_for)")
        cursor.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_checkins_dedupe
            ON scheduled_checkins(user_id, dedupe_key)
            WHERE dedupe_key IS NOT NULL
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS watchdog_runs (
                user_id TEXT PRIMARY KEY,
                ran_at TEXT NOT NULL,
                force_ran_at TEXT
            )
        """)
        try:
            cursor.execute("ALTER TABLE watchdog_runs ADD COLUMN force_ran_at TEXT")
        except Exception:
            pass  # Column already exists

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS patient_baselines (
                user_id TEXT PRIMARY KEY,
                baseline_text TEXT NOT NULL,
                last_compressed_at TEXT NOT NULL
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS watchdog_observations (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                observation TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id TEXT PRIMARY KEY,
                name TEXT,
                age INTEGER,
                gender TEXT,
                conditions_json TEXT,
                allergies_json TEXT,
                regular_medications_json TEXT,
                surgeries_json TEXT,
                family_history_json TEXT,
                social_history_json TEXT,
                patterns_json TEXT,
                health_summary TEXT,
                intake_completed INTEGER DEFAULT 0,
                intake_questions_asked INTEGER DEFAULT 0,
                intake_answered_json TEXT,
                intake_last_question_id TEXT,
                intake_started_at TEXT,
                intake_completed_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS medication_reminder_actions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                medication_id TEXT NOT NULL,
                due_at TEXT NOT NULL,
                action TEXT NOT NULL,
                snoozed_until TEXT,
                created_at TEXT NOT NULL
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_reminder_actions_user ON medication_reminder_actions(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_reminder_actions_med ON medication_reminder_actions(medication_id)")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cycle_day_logs (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                date TEXT NOT NULL,
                flow_level TEXT NOT NULL,
                is_period_day INTEGER DEFAULT 1,
                notes TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_cycle_user ON cycle_day_logs(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_cycle_date ON cycle_day_logs(user_id, date)")
        cursor.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS idx_cycle_user_date
            ON cycle_day_logs(user_id, date)
        """)

        # --- Versioned migrations (for databases created before new columns were added) ---
        _run_migrations(cursor)

        conn.commit()
        conn.close()
        _db_initialized = True


def _row_to_log(row: tuple) -> LogEntry:
    """Convert a database row to a LogEntry."""
    try:
        extracted_data = json.loads(row[6]) if row[6] else {}
        extracted = ExtractionResult(**extracted_data)
    except (json.JSONDecodeError, TypeError, ValueError):
        # Fallback to empty extraction if JSON is corrupted
        extracted = ExtractionResult(transcript=row[3] or "")

    # Parse image analysis if present (column index 11, after deleted at 10)
    image_analysis = None
    if len(row) > 11 and row[11]:
        try:
            image_analysis = ImageAnalysisResult(**json.loads(row[11]))
        except (json.JSONDecodeError, TypeError, ValueError):
            image_analysis = None

    contact_clinician_note = row[12] if len(row) > 12 else None
    contact_clinician_reason = row[13] if len(row) > 13 else None

    # Deserialize followup exchanges — prefer new JSON column, fall back to old scalars
    exchanges: list[FollowupExchange] = []
    exchanges_json = row[15] if len(row) > 15 else None
    if exchanges_json and exchanges_json != '[]':
        try:
            exchanges = [FollowupExchange(**e) for e in json.loads(exchanges_json)]
        except (json.JSONDecodeError, TypeError, ValueError):
            exchanges = []
    elif row[7]:  # Old scalar followup_question column
        followup_response = row[14] if len(row) > 14 else None
        exchanges = [FollowupExchange(
            question=row[7],
            answer=row[8],
            agent_response=followup_response,
            answered_at=datetime.fromisoformat(row[9]) if row[9] else None,
        )]

    return LogEntry(
        id=row[0],
        user_id=row[1],
        recorded_at=datetime.fromisoformat(row[2]),
        transcript=row[3],
        description=row[4],
        photo_b64=row[5],
        extracted=extracted,
        image_analysis=image_analysis,
        contact_clinician_note=contact_clinician_note,
        contact_clinician_reason=contact_clinician_reason,
        followup_exchanges=exchanges,
        # Note: deleted column (row[10]) is not exposed in LogEntry model
    )


def _serialize_exchanges(exchanges: list) -> str:
    """Serialize followup exchanges to JSON string."""
    return json.dumps([e.model_dump(mode='json') for e in exchanges]) if exchanges else '[]'


def create_log(entry: LogEntry) -> None:
    """Persist a new log entry."""
    with _get_connection() as conn:
        conn.execute(
            """
            INSERT INTO logs (id, user_id, recorded_at, transcript, description, photo_b64,
                              extracted_json, followup_question, followup_answer, followup_answered_at,
                              deleted, image_analysis_json, contact_clinician_note, contact_clinician_reason,
                              followup_response, followup_exchanges_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.id,
                entry.user_id,
                entry.recorded_at.isoformat(),
                entry.transcript,
                entry.description,
                entry.photo_b64,
                entry.extracted.model_dump_json(),
                None,  # legacy followup_question (use exchanges_json)
                None,  # legacy followup_answer
                None,  # legacy followup_answered_at
                0,  # deleted = False by default
                entry.image_analysis.model_dump_json() if entry.image_analysis else None,
                entry.contact_clinician_note,
                entry.contact_clinician_reason,
                None,  # legacy followup_response
                _serialize_exchanges(entry.followup_exchanges),
            ),
        )


def get_log(log_id: str) -> Optional[LogEntry]:
    """Retrieve a log by ID."""
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(f"SELECT {LOG_COLUMNS} FROM logs WHERE id = ?", (log_id,))
        row = cursor.fetchone()
        if row is None:
            return None
        return _row_to_log(row)


def list_logs(user_id: str, limit: int = 50, include_deleted: bool = False) -> List[LogEntry]:
    """List recent logs for a user.

    Args:
        user_id: User ID to filter by
        limit: Maximum number of logs to return
        include_deleted: If True, include soft-deleted logs in results
    """
    with _get_connection() as conn:
        cursor = conn.cursor()

        if include_deleted:
            cursor.execute(
                f"SELECT {LOG_COLUMNS} FROM logs WHERE user_id = ? ORDER BY recorded_at DESC LIMIT ?",
                (user_id, limit),
            )
        else:
            cursor.execute(
                f"SELECT {LOG_COLUMNS} FROM logs WHERE user_id = ? AND deleted = 0 ORDER BY recorded_at DESC LIMIT ?",
                (user_id, limit),
            )

        rows = cursor.fetchall()
        return [_row_to_log(row) for row in rows]


def list_logs_in_days(
    user_id: str,
    days: int,
    limit: int = 200,
    include_deleted: bool = False,
    reference_date: Optional[datetime] = None,
) -> List[LogEntry]:
    """List logs for a user within the last N days.

    Args:
        reference_date: If provided, use this as the anchor instead of wall-clock
            time. Essential for simulation data where recorded_at may be weeks
            in the past.
    """
    ref = reference_date or _utcnow()
    cutoff = (ref - timedelta(days=days)).isoformat()
    with _get_connection() as conn:
        cursor = conn.cursor()

        if include_deleted:
            cursor.execute(
                f"""
                SELECT {LOG_COLUMNS}
                FROM logs
                WHERE user_id = ? AND recorded_at >= ?
                ORDER BY recorded_at DESC
                LIMIT ?
                """,
                (user_id, cutoff, limit),
            )
        else:
            cursor.execute(
                f"""
                SELECT {LOG_COLUMNS}
                FROM logs
                WHERE user_id = ? AND deleted = 0 AND recorded_at >= ?
                ORDER BY recorded_at DESC
                LIMIT ?
                """,
                (user_id, cutoff, limit),
            )

        rows = cursor.fetchall()
        return [_row_to_log(row) for row in rows]


def list_all_logs(user_id: str, include_deleted: bool = False) -> List[LogEntry]:
    """Fetch ALL logs for a user, ordered oldest-first (chronological)."""
    with _get_connection() as conn:
        cursor = conn.cursor()
        if include_deleted:
            cursor.execute(
                f"SELECT {LOG_COLUMNS} FROM logs WHERE user_id = ? ORDER BY recorded_at ASC",
                (user_id,),
            )
        else:
            cursor.execute(
                f"SELECT {LOG_COLUMNS} FROM logs WHERE user_id = ? AND deleted = 0 ORDER BY recorded_at ASC",
                (user_id,),
            )
        rows = cursor.fetchall()
        return [_row_to_log(row) for row in rows]


def list_logs_in_date_range(
    user_id: str,
    start_date: str,
    end_date: str,
    limit: int = 200,
    include_deleted: bool = False,
) -> List[LogEntry]:
    """List logs for a user within a specific date range.

    Args:
        user_id: User ID to filter by
        start_date: ISO 8601 datetime string (inclusive lower bound)
        end_date: ISO 8601 datetime string (exclusive upper bound)
        limit: Maximum number of logs to return
        include_deleted: If True, include soft-deleted logs
    """
    with _get_connection() as conn:
        cursor = conn.cursor()

        if include_deleted:
            cursor.execute(
                f"""
                SELECT {LOG_COLUMNS}
                FROM logs
                WHERE user_id = ? AND recorded_at >= ? AND recorded_at < ?
                ORDER BY recorded_at DESC
                LIMIT ?
                """,
                (user_id, start_date, end_date, limit),
            )
        else:
            cursor.execute(
                f"""
                SELECT {LOG_COLUMNS}
                FROM logs
                WHERE user_id = ? AND deleted = 0 AND recorded_at >= ? AND recorded_at < ?
                ORDER BY recorded_at DESC
                LIMIT ?
                """,
                (user_id, start_date, end_date, limit),
            )

        rows = cursor.fetchall()
        return [_row_to_log(row) for row in rows]


def list_logs_with_image_analysis(user_id: str, limit: int = 20) -> List[LogEntry]:
    """Retrieve recent logs for a user that have image analysis results."""
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            f"""
            SELECT {LOG_COLUMNS}
            FROM logs
            WHERE user_id = ? AND deleted = 0 AND image_analysis_json IS NOT NULL
            ORDER BY recorded_at DESC
            LIMIT ?
            """,
            (user_id, limit),
        )
        rows = cursor.fetchall()
        return [_row_to_log(row) for row in rows]


def update_followup_exchanges(log_id: str, exchanges: list) -> None:
    """Overwrite the full followup exchanges list for a log."""
    with _get_connection() as conn:
        conn.execute(
            "UPDATE logs SET followup_exchanges_json = ? WHERE id = ?",
            (_serialize_exchanges(exchanges), log_id),
        )


def update_extraction(log_id: str, extraction: ExtractionResult) -> None:
    """Update the extracted_json column for a log entry."""
    with _get_connection() as conn:
        conn.execute(
            "UPDATE logs SET extracted_json = ? WHERE id = ?",
            (extraction.model_dump_json(), log_id),
        )


def delete_log(log_id: str) -> bool:
    """Soft delete a log entry (marks as deleted, but keeps in database).

    Args:
        log_id: The log ID to soft delete

    Returns:
        True if log was found and deleted, False otherwise
    """
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE logs SET deleted = 1 WHERE id = ?",
            (log_id,),
        )
        affected = cursor.rowcount
    return affected > 0


def permanent_delete_log(log_id: str) -> bool:
    """Permanently delete a log entry from the database.

    Args:
        log_id: The log ID to permanently delete

    Returns:
        True if log was found and deleted, False otherwise
    """
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "DELETE FROM logs WHERE id = ?",
            (log_id,),
        )
        affected = cursor.rowcount
    return affected > 0


# === Medication Storage ===

def _row_to_medication(row: tuple) -> MedicationEntry:
    """Convert a database row to a MedicationEntry."""
    # Parse reminder_times from JSON (row[13])
    reminder_times = []
    if len(row) > 13 and row[13]:
        try:
            reminder_times = json.loads(row[13])
        except (json.JSONDecodeError, TypeError):
            reminder_times = []

    # Get reminder_enabled (row[12])
    reminder_enabled = bool(row[12]) if len(row) > 12 and row[12] is not None else False

    return MedicationEntry(
        id=row[0],
        user_id=row[1],
        name=row[2],
        dose=row[3],
        frequency=row[4],
        reason=row[5],
        start_date=datetime.fromisoformat(row[6]) if row[6] else None,
        end_date=datetime.fromisoformat(row[7]) if row[7] else None,
        notes=row[8],
        is_active=bool(row[9]),
        created_at=datetime.fromisoformat(row[10]),
        updated_at=datetime.fromisoformat(row[11]),
        reminder_enabled=reminder_enabled,
        reminder_times=reminder_times,
    )


def _row_to_medication_log(row: tuple) -> MedicationLogEntry:
    """Convert a database row to a MedicationLogEntry."""
    return MedicationLogEntry(
        id=row[0],
        user_id=row[1],
        medication_id=row[2],
        medication_name=row[3],
        dose_taken=row[4],
        taken_at=datetime.fromisoformat(row[5]),
        notes=row[6],
        symptom_log_id=row[7],
    )


def create_medication(entry: MedicationEntry) -> None:
    """Create a new saved medication."""
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO medications (id, user_id, name, dose, frequency, reason,
                                     start_date, end_date, notes, is_active, created_at, updated_at,
                                     reminder_enabled, reminder_times)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.id,
                entry.user_id,
                entry.name,
                entry.dose,
                entry.frequency,
                entry.reason,
                entry.start_date.isoformat() if entry.start_date else None,
                entry.end_date.isoformat() if entry.end_date else None,
                entry.notes,
                1 if entry.is_active else 0,
                entry.created_at.isoformat(),
                entry.updated_at.isoformat(),
                1 if entry.reminder_enabled else 0,
                json.dumps(entry.reminder_times) if entry.reminder_times else None,
            ),
        )


def get_medication(med_id: str) -> Optional[MedicationEntry]:
    """Get a medication by ID."""
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(f"SELECT {MED_COLUMNS} FROM medications WHERE id = ?", (med_id,))
        row = cursor.fetchone()
        if row is None:
            return None
        return _row_to_medication(row)


def list_medications(user_id: str, active_only: bool = True) -> List[MedicationEntry]:
    """List medications for a user."""
    with _get_connection() as conn:
        cursor = conn.cursor()
        if active_only:
            cursor.execute(
                f"SELECT {MED_COLUMNS} FROM medications WHERE user_id = ? AND is_active = 1 ORDER BY name",
                (user_id,),
            )
        else:
            cursor.execute(
                f"SELECT {MED_COLUMNS} FROM medications WHERE user_id = ? ORDER BY is_active DESC, name",
                (user_id,),
            )
        rows = cursor.fetchall()
        return [_row_to_medication(row) for row in rows]


def update_medication(med_id: str, updates: dict) -> Optional[MedicationEntry]:
    """Update a medication."""
    # Build dynamic update query
    allowed_fields = ['name', 'dose', 'frequency', 'reason', 'start_date', 'end_date', 'notes', 'is_active',
                      'reminder_enabled', 'reminder_times']
    set_clauses = []
    values = []
    for field in allowed_fields:
        if field in updates:
            set_clauses.append(f"{field} = ?")
            val = updates[field]
            if field in ('start_date', 'end_date') and val is not None:
                val = val.isoformat() if hasattr(val, 'isoformat') else val
            elif field == 'is_active' or field == 'reminder_enabled':
                val = 1 if val else 0
            elif field == 'reminder_times':
                val = json.dumps(val) if val is not None else None
            values.append(val)

    if not set_clauses:
        return get_medication(med_id)

    set_clauses.append("updated_at = ?")
    values.append(_utcnow().isoformat())
    values.append(med_id)

    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            f"UPDATE medications SET {', '.join(set_clauses)} WHERE id = ?",
            values,
        )
    return get_medication(med_id)


def create_medication_log(entry: MedicationLogEntry) -> None:
    """Log a medication dose taken."""
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO medication_logs (id, user_id, medication_id, medication_name,
                                         dose_taken, taken_at, notes, symptom_log_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.id,
                entry.user_id,
                entry.medication_id,
                entry.medication_name,
                entry.dose_taken,
                entry.taken_at.isoformat(),
                entry.notes,
                entry.symptom_log_id,
            ),
        )


def list_medication_logs(
    user_id: str,
    limit: int = 50,
    days: Optional[int] = None,
    reference_date: Optional[datetime] = None,
) -> List[MedicationLogEntry]:
    """List recent medication log entries for a user."""
    with _get_connection() as conn:
        cursor = conn.cursor()

        if days is not None:
            ref = reference_date or _utcnow()
            cutoff = (ref - timedelta(days=days)).isoformat()
            cursor.execute(
                f"SELECT {MED_LOG_COLUMNS} FROM medication_logs WHERE user_id = ? AND taken_at >= ? ORDER BY taken_at DESC LIMIT ?",
                (user_id, cutoff, limit),
            )
        else:
            cursor.execute(
                f"SELECT {MED_LOG_COLUMNS} FROM medication_logs WHERE user_id = ? ORDER BY taken_at DESC LIMIT ?",
                (user_id, limit),
            )
        rows = cursor.fetchall()
        return [_row_to_medication_log(row) for row in rows]


def get_pending_medication_reminders(user_id: str, as_of: Optional[datetime] = None):
    """Get pending medication reminders for a user.

    Returns medications with reminders enabled that are due now.
    Excludes reminders that were dismissed/taken/snoozed for today.
    """
    import os
    from ..models import PendingMedicationReminder

    # Get configurable time windows from environment
    early_minutes = int(os.getenv('REMINDER_EARLY_MINUTES', '15'))
    late_minutes = int(os.getenv('REMINDER_LATE_MINUTES', '60'))

    if as_of is None:
        as_of = _utcnow()

    with _get_connection() as conn:
        cursor = conn.cursor()

        # Get all active medications with reminders enabled
        cursor.execute(
            f"""
            SELECT {MED_COLUMNS} FROM medications
            WHERE user_id = ? AND is_active = 1 AND reminder_enabled = 1
            """,
            (user_id,)
        )
        meds = cursor.fetchall()

        # Get recent reminder actions (dismissals, snoozes, takes).
        # Include the previous day to preserve snoozes that cross midnight.
        today_start = as_of.replace(hour=0, minute=0, second=0, microsecond=0)
        actions_start = today_start - timedelta(days=1)
        cursor.execute(
            """
            SELECT medication_id, due_at, action, snoozed_until
            FROM medication_reminder_actions
            WHERE user_id = ? AND due_at >= ?
            """,
            (user_id, actions_start.isoformat())
        )
        actions = {(row[0], row[1]): {'action': row[2], 'snoozed_until': row[3]}
                   for row in cursor.fetchall()}

    pending = []
    for med_row in meds:
        med = _row_to_medication(med_row)
        if not med.reminder_times:
            continue

        # Check each reminder time for this medication
        for time_str in med.reminder_times:
            try:
                # Parse time like "08:00"
                hour, minute = map(int, time_str.split(':'))
                due_datetime = as_of.replace(hour=hour, minute=minute, second=0, microsecond=0)

                action_key = (med.id, due_datetime.isoformat())
                action_data = actions.get(action_key)
                effective_due = due_datetime

                if action_data:
                    if action_data["action"] == "snoozed":
                        snoozed_until_raw = action_data.get("snoozed_until")
                        if not snoozed_until_raw:
                            continue
                        try:
                            snoozed_until = datetime.fromisoformat(snoozed_until_raw)
                        except ValueError:
                            continue
                        if as_of < snoozed_until:
                            continue  # Still snoozed.
                        # Once snooze expires, evaluate due-ness from snoozed_until,
                        # not the original due time (which may already be out-of-window).
                        effective_due = snoozed_until
                    else:
                        continue  # already taken or dismissed

                # Check if this reminder is due (within configured time window)
                time_diff = (as_of - effective_due).total_seconds() / 60  # minutes
                if -early_minutes <= time_diff <= late_minutes:
                    is_overdue = time_diff > 0
                    pending.append(PendingMedicationReminder(
                        medication_id=med.id,
                        medication_name=med.name,
                        dose=med.dose,
                        scheduled_time=time_str,
                        # Keep the original due instance key for follow-up actions.
                        due_at=due_datetime,
                        is_overdue=is_overdue
                    ))
            except (ValueError, IndexError):
                continue  # Skip invalid time format

    return pending


def record_reminder_action(user_id: str, medication_id: str, due_at: datetime,
                          action: str, snoozed_until: Optional[datetime] = None) -> None:
    """Record that a user took action on a medication reminder.

    Args:
        user_id: User ID
        medication_id: Medication ID
        due_at: When the reminder was due
        action: 'taken', 'dismissed', or 'snoozed'
        snoozed_until: If action='snoozed', when to remind again
    """
    import uuid
    with _get_connection() as conn:
        cursor = conn.cursor()

        # Delete any existing action for this reminder instance
        cursor.execute(
            "DELETE FROM medication_reminder_actions WHERE user_id = ? AND medication_id = ? AND due_at = ?",
            (user_id, medication_id, due_at.isoformat())
        )

        # Insert new action
        cursor.execute(
            """
            INSERT INTO medication_reminder_actions
            (id, user_id, medication_id, due_at, action, snoozed_until, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                f"reminder_action_{uuid.uuid4().hex[:16]}",
                user_id,
                medication_id,
                due_at.isoformat(),
                action,
                snoozed_until.isoformat() if snoozed_until else None,
                _utcnow().isoformat()
            )
        )


# === Ambient Session Storage ===

def _row_to_ambient_session(row: tuple) -> AmbientSession:
    """Convert a database row to an AmbientSession."""
    return AmbientSession(
        id=row[0],
        user_id=row[1],
        session_type=row[2],
        status=row[3],
        label=row[4],
        started_at=datetime.fromisoformat(row[5]),
        ended_at=datetime.fromisoformat(row[6]) if row[6] else None,
        chunk_count=row[7],
        total_duration_seconds=row[8],
        created_at=datetime.fromisoformat(row[9]),
        updated_at=datetime.fromisoformat(row[10]),
    )


def _row_to_ambient_event(row: tuple) -> AmbientEvent:
    """Convert a database row to an AmbientEvent."""
    try:
        metadata = json.loads(row[7]) if row[7] else None
    except (json.JSONDecodeError, TypeError):
        metadata = None

    return AmbientEvent(
        id=row[0],
        session_id=row[1],
        user_id=row[2],
        event_type=row[3],
        timestamp=datetime.fromisoformat(row[4]),
        duration_seconds=row[5],
        confidence=row[6],
        metadata=metadata,
        chunk_index=row[8],
    )


def _row_to_ambient_chunk(row: tuple) -> AmbientChunk:
    """Convert a database row to an AmbientChunk."""
    return AmbientChunk(
        id=row[0],
        session_id=row[1],
        user_id=row[2],
        chunk_index=row[3],
        duration_seconds=row[4],
        uploaded_at=datetime.fromisoformat(row[5]),
        processed=bool(row[6]),
        events_detected=row[7],
    )


class ActiveSessionExistsError(Exception):
    """Raised when trying to create a session but an active one already exists."""
    pass


def create_ambient_session(session: AmbientSession) -> None:
    """Create a new ambient session.

    Raises:
        ActiveSessionExistsError: If user already has an active session.
    """
    try:
        with _get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO ambient_sessions
                (id, user_id, session_type, status, label, started_at, ended_at,
                 chunk_count, total_duration_seconds, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session.id,
                    session.user_id,
                    session.session_type.value if hasattr(session.session_type, 'value') else session.session_type,
                    session.status.value if hasattr(session.status, 'value') else session.status,
                    session.label,
                    session.started_at.isoformat(),
                    session.ended_at.isoformat() if session.ended_at else None,
                    session.chunk_count,
                    session.total_duration_seconds,
                    session.created_at.isoformat(),
                    session.updated_at.isoformat(),
                ),
            )
    except sqlite3.IntegrityError as e:
        if "idx_ambient_user_active" in str(e) or "UNIQUE constraint" in str(e):
            raise ActiveSessionExistsError(f"User {session.user_id} already has an active session")
        raise


def get_ambient_session(session_id: str) -> Optional[AmbientSession]:
    """Get an ambient session by ID."""
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(f"SELECT {AMBIENT_SESSION_COLUMNS} FROM ambient_sessions WHERE id = ?", (session_id,))
        row = cursor.fetchone()
        return _row_to_ambient_session(row) if row else None


def update_ambient_session(session_id: str, updates: dict) -> Optional[AmbientSession]:
    """Update an ambient session."""
    allowed_fields = ['status', 'ended_at', 'chunk_count', 'total_duration_seconds', 'result_json']
    set_clauses = []
    values = []

    for field in allowed_fields:
        if field in updates:
            set_clauses.append(f"{field} = ?")
            val = updates[field]
            if field == 'ended_at' and val is not None:
                val = val.isoformat() if hasattr(val, 'isoformat') else val
            elif field == 'status' and hasattr(val, 'value'):
                val = val.value
            values.append(val)

    if not set_clauses:
        return get_ambient_session(session_id)

    set_clauses.append("updated_at = ?")
    values.append(_utcnow().isoformat())
    values.append(session_id)

    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            f"UPDATE ambient_sessions SET {', '.join(set_clauses)} WHERE id = ?",
            values,
        )
    return get_ambient_session(session_id)


def increment_ambient_session_stats(session_id: str, duration_seconds: float) -> None:
    """Atomically increment ambient session counters."""
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE ambient_sessions
            SET chunk_count = chunk_count + 1,
                total_duration_seconds = total_duration_seconds + ?,
                updated_at = ?
            WHERE id = ?
            """,
            (duration_seconds, _utcnow().isoformat(), session_id),
        )


def list_ambient_sessions(user_id: str, limit: int = 20) -> List[AmbientSession]:
    """List ambient sessions for a user."""
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT {AMBIENT_SESSION_COLUMNS} FROM ambient_sessions WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit),
        )
        rows = cursor.fetchall()
        return [_row_to_ambient_session(row) for row in rows]


def get_active_session(user_id: str) -> Optional[AmbientSession]:
    """Get the active session for a user (if any)."""
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT {AMBIENT_SESSION_COLUMNS} FROM ambient_sessions WHERE user_id = ? AND status = 'active' LIMIT 1",
            (user_id,),
        )
        row = cursor.fetchone()
        return _row_to_ambient_session(row) if row else None


def create_ambient_event(event: AmbientEvent) -> None:
    """Create a new ambient event."""
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO ambient_events
            (id, session_id, user_id, event_type, timestamp, duration_seconds,
             confidence, metadata_json, chunk_index)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event.id,
                event.session_id,
                event.user_id,
                event.event_type.value if hasattr(event.event_type, 'value') else event.event_type,
                event.timestamp.isoformat(),
                event.duration_seconds,
                event.confidence,
                json.dumps(event.metadata) if event.metadata else None,
                event.chunk_index,
            ),
        )


def create_ambient_events_batch(events: List[AmbientEvent]) -> None:
    """Batch insert multiple ambient events in a single transaction."""
    if not events:
        return
    with _get_connection() as conn:
        for event in events:
            conn.execute(
                """
                INSERT INTO ambient_events
                (id, session_id, user_id, event_type, timestamp, duration_seconds,
                 confidence, metadata_json, chunk_index)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.id,
                    event.session_id,
                    event.user_id,
                    event.event_type.value if hasattr(event.event_type, 'value') else event.event_type,
                    event.timestamp.isoformat(),
                    event.duration_seconds,
                    event.confidence,
                    json.dumps(event.metadata) if event.metadata else None,
                    event.chunk_index,
                ),
            )


def list_session_events(session_id: str) -> List[AmbientEvent]:
    """List all events for a session."""
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT {AMBIENT_EVENT_COLUMNS} FROM ambient_events WHERE session_id = ? ORDER BY timestamp",
            (session_id,),
        )
        rows = cursor.fetchall()
        return [_row_to_ambient_event(row) for row in rows]


def get_session_result_json(session_id: str) -> Optional[str]:
    """Get the stored result JSON for a completed session."""
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT result_json FROM ambient_sessions WHERE id = ?",
            (session_id,),
        )
        row = cursor.fetchone()
        return row[0] if row else None


def create_ambient_chunk(chunk: AmbientChunk) -> bool:
    """Record an uploaded audio chunk.

    Returns:
        True if inserted, False if duplicate (same session_id/chunk_index).
    """
    try:
        with _get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO ambient_chunks
                (id, session_id, user_id, chunk_index, duration_seconds,
                 uploaded_at, processed, events_detected)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk.id,
                    chunk.session_id,
                    chunk.user_id,
                    chunk.chunk_index,
                    chunk.duration_seconds,
                    chunk.uploaded_at.isoformat(),
                    1 if chunk.processed else 0,
                    chunk.events_detected,
                ),
            )
        return True
    except sqlite3.IntegrityError:
        return False


def get_ambient_chunk_by_index(session_id: str, chunk_index: int) -> Optional[AmbientChunk]:
    """Get an uploaded chunk by session/index for idempotent retries."""
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT id, session_id, user_id, chunk_index, duration_seconds, uploaded_at, processed, events_detected
            FROM ambient_chunks
            WHERE session_id = ? AND chunk_index = ?
            LIMIT 1
            """,
            (session_id, chunk_index),
        )
        row = cursor.fetchone()
        return _row_to_ambient_chunk(row) if row else None


def list_session_events_for_chunk(session_id: str, chunk_index: int) -> List[AmbientEvent]:
    """List events emitted for a specific chunk index."""
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            f"""
            SELECT {AMBIENT_EVENT_COLUMNS}
            FROM ambient_events
            WHERE session_id = ? AND chunk_index = ?
            ORDER BY timestamp
            """,
            (session_id, chunk_index),
        )
        rows = cursor.fetchall()
        return [_row_to_ambient_event(row) for row in rows]


# === Scheduled Check-in Storage ===

def _row_to_scheduled_checkin(row: tuple) -> ScheduledCheckin:
    """Convert a database row to a ScheduledCheckin."""
    try:
        context = json.loads(row[5]) if row[5] else {}
    except (json.JSONDecodeError, TypeError):
        context = {}

    return ScheduledCheckin(
        id=row[0],
        user_id=row[1],
        checkin_type=row[2],
        scheduled_for=datetime.fromisoformat(row[3]),
        message=row[4],
        context=context,
        created_at=datetime.fromisoformat(row[6]),
        triggered=bool(row[7]),
        dismissed=bool(row[8]),
        response=row[9],
        responded_at=datetime.fromisoformat(row[10]) if row[10] else None,
    )


def _checkin_dedupe_key(checkin: ScheduledCheckin) -> Optional[str]:
    """Generate a dedupe key to reduce concurrent duplicate inserts."""
    try:
        scheduled_for = checkin.scheduled_for
        if scheduled_for.tzinfo is None:
            scheduled_for = scheduled_for.replace(tzinfo=timezone.utc)
        epoch = int(scheduled_for.timestamp())
    except Exception:
        return None

    message = (checkin.message or "").strip().lower()
    med_name = ""
    if checkin.context and isinstance(checkin.context, dict):
        med_name = (checkin.context.get("medication_name") or "").strip().lower()

    if checkin.checkin_type == CheckinType.MEDICATION_FOLLOWUP and med_name:
        bucket = epoch // (6 * 3600)
        return f"med:{med_name}:{bucket}"

    if checkin.checkin_type == CheckinType.PROFILE_INTAKE:
        question_id = ""
        if checkin.context and isinstance(checkin.context, dict):
            question_id = (checkin.context.get("question_id") or "").strip().lower()
        if not question_id:
            question_id = "generic"
        bucket = epoch // (12 * 3600)
        return f"intake:{question_id}:{bucket}"

    if message:
        bucket = epoch // (2 * 3600)
        digest = hashlib.sha256(message.encode("utf-8")).hexdigest()[:12]
        return f"msg:{digest}:{bucket}"

    return None


# ---------------------------------------------------------------------------
# Watchdog run tracking (cooldown)
# ---------------------------------------------------------------------------

def record_watchdog_run(user_id: str, force: bool = False) -> None:
    """Record that the watchdog ran for this user (upsert)."""
    now = datetime.now(timezone.utc).isoformat()
    with _get_connection() as conn:
        if force:
            conn.execute(
                "INSERT INTO watchdog_runs (user_id, ran_at, force_ran_at) VALUES (?, ?, ?)"
                " ON CONFLICT(user_id) DO UPDATE SET ran_at = excluded.ran_at, force_ran_at = excluded.force_ran_at",
                (user_id, now, now),
            )
        else:
            conn.execute(
                "INSERT INTO watchdog_runs (user_id, ran_at) VALUES (?, ?)"
                " ON CONFLICT(user_id) DO UPDATE SET ran_at = excluded.ran_at",
                (user_id, now),
            )


def get_last_watchdog_run(user_id: str) -> Optional[datetime]:
    """Return the last time the watchdog ran for this user, or None."""
    with _get_connection() as conn:
        row = conn.execute(
            "SELECT ran_at FROM watchdog_runs WHERE user_id = ?", (user_id,)
        ).fetchone()
        if row:
            return datetime.fromisoformat(row[0])
        return None


def get_last_force_watchdog_run(user_id: str) -> Optional[datetime]:
    """Return the last time a force-triggered watchdog ran, or None."""
    with _get_connection() as conn:
        row = conn.execute(
            "SELECT force_ran_at FROM watchdog_runs WHERE user_id = ?", (user_id,)
        ).fetchone()
        if row and row[0]:
            return datetime.fromisoformat(row[0])
        return None


def get_baseline_info(user_id: str) -> dict:
    """Return the patient's compressed historical baseline, or defaults."""
    with _get_connection() as conn:
        row = conn.execute(
            "SELECT baseline_text, last_compressed_at FROM patient_baselines WHERE user_id = ?",
            (user_id,),
        ).fetchone()
        if row:
            return {"text": row[0], "last_compressed_at": row[1]}
        return {"text": "", "last_compressed_at": "1970-01-01T00:00:00+00:00"}


def update_baseline(user_id: str, baseline_text: str, compressed_at: str) -> None:
    """Upsert the patient's compressed baseline and time cursor."""
    with _get_connection() as conn:
        conn.execute(
            "INSERT INTO patient_baselines (user_id, baseline_text, last_compressed_at) VALUES (?, ?, ?)"
            " ON CONFLICT(user_id) DO UPDATE SET baseline_text = excluded.baseline_text,"
            " last_compressed_at = excluded.last_compressed_at",
            (user_id, baseline_text, compressed_at),
        )


def list_logs_in_range(
    user_id: str, after: str, before: str, limit: int = 200
) -> List[LogEntry]:
    """List non-deleted logs between two ISO timestamps (after < recorded_at <= before)."""
    with _get_connection() as conn:
        rows = conn.execute(
            f"SELECT {LOG_COLUMNS} FROM logs"
            " WHERE user_id = ? AND deleted = 0 AND recorded_at > ? AND recorded_at <= ?"
            " ORDER BY recorded_at ASC LIMIT ?",
            (user_id, after, before, limit),
        ).fetchall()
        return [_row_to_log(row) for row in rows]


def store_watchdog_observation(user_id: str, observation: str) -> None:
    """Persist a clinician-facing observation from the Watchdog."""
    obs_id = f"wdobs_{uuid.uuid4().hex[:12]}"
    with _get_connection() as conn:
        conn.execute(
            "INSERT INTO watchdog_observations (id, user_id, observation, created_at)"
            " VALUES (?, ?, ?, ?)",
            (obs_id, user_id, observation, datetime.now(timezone.utc).isoformat()),
        )


def get_watchdog_observations(user_id: str, limit: int = 5) -> List[str]:
    """Fetch recent clinician-facing observations for Doctor Packet injection."""
    with _get_connection() as conn:
        rows = conn.execute(
            "SELECT observation FROM watchdog_observations"
            " WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
        return [row[0] for row in rows]


def create_scheduled_checkin(checkin: ScheduledCheckin) -> bool:
    """Create a new scheduled check-in. Returns False if deduped."""
    dedupe_key = _checkin_dedupe_key(checkin)
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT OR IGNORE INTO scheduled_checkins
            (id, user_id, checkin_type, scheduled_for, message, context_json, dedupe_key,
             created_at, triggered, dismissed, response, responded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                checkin.id,
                checkin.user_id,
                checkin.checkin_type.value if hasattr(checkin.checkin_type, 'value') else checkin.checkin_type,
                checkin.scheduled_for.isoformat(),
                checkin.message,
                json.dumps(checkin.context) if checkin.context else None,
                dedupe_key,
                checkin.created_at.isoformat(),
                1 if checkin.triggered else 0,
                1 if checkin.dismissed else 0,
                checkin.response,
                checkin.responded_at.isoformat() if checkin.responded_at else None,
            ),
        )
        inserted = cursor.rowcount == 1
    return inserted


def get_scheduled_checkin(checkin_id: str) -> Optional[ScheduledCheckin]:
    """Get a scheduled check-in by ID."""
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(f"SELECT {CHECKIN_COLUMNS} FROM scheduled_checkins WHERE id = ?", (checkin_id,))
        row = cursor.fetchone()
        return _row_to_scheduled_checkin(row) if row else None


def get_pending_checkins(user_id: str, as_of: Optional[datetime] = None) -> List[ScheduledCheckin]:
    """Get check-ins that are due (scheduled_for <= as_of, not dismissed, not responded)."""
    if as_of is None:
        as_of = _utcnow()

    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            f"""
            SELECT {CHECKIN_COLUMNS} FROM scheduled_checkins
            WHERE user_id = ?
              AND scheduled_for <= ?
              AND dismissed = 0
              AND response IS NULL
            ORDER BY scheduled_for ASC
            """,
            (user_id, as_of.isoformat()),
        )
        rows = cursor.fetchall()
        return [_row_to_scheduled_checkin(row) for row in rows]


def list_open_checkins(
    user_id: str,
    checkin_type: Optional[CheckinType] = None,
    limit: int = 100,
) -> List[ScheduledCheckin]:
    """List unresolved check-ins for dedupe/rate-limit logic."""
    with _get_connection() as conn:
        cursor = conn.cursor()

        if checkin_type is None:
            cursor.execute(
                f"""
                SELECT {CHECKIN_COLUMNS} FROM scheduled_checkins
                WHERE user_id = ?
                  AND dismissed = 0
                  AND response IS NULL
                ORDER BY scheduled_for DESC
                LIMIT ?
                """,
                (user_id, limit),
            )
        else:
            checkin_type_value = checkin_type.value if hasattr(checkin_type, "value") else str(checkin_type)
            cursor.execute(
                f"""
                SELECT {CHECKIN_COLUMNS} FROM scheduled_checkins
                WHERE user_id = ?
                  AND checkin_type = ?
                  AND dismissed = 0
                  AND response IS NULL
                ORDER BY scheduled_for DESC
                LIMIT ?
                """,
                (user_id, checkin_type_value, limit),
            )

        rows = cursor.fetchall()
        return [_row_to_scheduled_checkin(row) for row in rows]


def mark_checkin_triggered(checkin_id: str) -> None:
    """Mark a check-in as shown to the user."""
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE scheduled_checkins SET triggered = 1 WHERE id = ?",
            (checkin_id,),
        )


def respond_to_checkin(checkin_id: str, response: str) -> Optional[ScheduledCheckin]:
    """Record user's response to a check-in."""
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            UPDATE scheduled_checkins
            SET response = ?, responded_at = ?, triggered = 1
            WHERE id = ?
            """,
            (response, _utcnow().isoformat(), checkin_id),
        )
    return get_scheduled_checkin(checkin_id)


def dismiss_checkin(checkin_id: str) -> None:
    """Dismiss a check-in without responding."""
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE scheduled_checkins SET dismissed = 1, triggered = 1 WHERE id = ?",
            (checkin_id,),
        )


# === User Profile Storage (Long-Term Memory) ===

def _row_to_user_profile(row: tuple) -> UserProfile:
    """Convert a database row to a UserProfile."""
    def parse_json_list(json_str: Optional[str]) -> List[str]:
        if not json_str:
            return []
        try:
            return json.loads(json_str)
        except (json.JSONDecodeError, TypeError):
            return []

    return UserProfile(
        user_id=row[0],
        name=row[1],
        age=int(row[2]) if row[2] is not None else None,
        gender=row[3],
        conditions=parse_json_list(row[4]),
        allergies=parse_json_list(row[5]),
        regular_medications=parse_json_list(row[6]),
        surgeries=parse_json_list(row[7]),
        family_history=parse_json_list(row[8]),
        social_history=parse_json_list(row[9]),
        patterns=parse_json_list(row[10]),
        health_summary=row[11],
        created_at=datetime.fromisoformat(row[12]),
        updated_at=datetime.fromisoformat(row[13]),
        intake_completed=bool(row[14]) if len(row) > 14 else False,
        intake_questions_asked=int(row[15] or 0) if len(row) > 15 else 0,
        intake_answered_question_ids=parse_json_list(row[16]) if len(row) > 16 else [],
        intake_last_question_id=row[17] if len(row) > 17 else None,
        intake_started_at=datetime.fromisoformat(row[18]) if len(row) > 18 and row[18] else None,
        intake_completed_at=datetime.fromisoformat(row[19]) if len(row) > 19 and row[19] else None,
        intake_pending_raw=json.loads(row[20]) if len(row) > 20 and row[20] else {},
    )


def get_user_profile(user_id: str) -> Optional[UserProfile]:
    """Get a user's profile (long-term health context)."""
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(f"SELECT {PROFILE_COLUMNS} FROM user_profiles WHERE user_id = ?", (user_id,))
        row = cursor.fetchone()
        return _row_to_user_profile(row) if row else None


def get_or_create_user_profile(user_id: str) -> UserProfile:
    """Get a user's profile, creating an empty one if it doesn't exist."""
    profile = get_user_profile(user_id)
    if profile:
        return profile

    # Create empty profile
    now = _utcnow()
    profile = UserProfile(
        user_id=user_id,
        conditions=[],
        allergies=[],
        regular_medications=[],
        surgeries=[],
        family_history=[],
        social_history=[],
        patterns=[],
        health_summary=None,
        created_at=now,
        updated_at=now,
        intake_completed=False,
        intake_questions_asked=0,
        intake_answered_question_ids=[],
        intake_last_question_id=None,
        intake_started_at=None,
        intake_completed_at=None,
    )
    create_user_profile(profile)
    return profile


def create_user_profile(profile: UserProfile) -> None:
    """Create a new user profile."""
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO user_profiles
            (user_id, name, age, gender,
             conditions_json, allergies_json, regular_medications_json,
             surgeries_json, family_history_json, social_history_json, patterns_json, health_summary,
             intake_completed, intake_questions_asked, intake_answered_json, intake_last_question_id,
             intake_started_at, intake_completed_at, created_at, updated_at, intake_pending_raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                profile.user_id,
                profile.name,
                profile.age,
                profile.gender,
                json.dumps(profile.conditions) if profile.conditions else None,
                json.dumps(profile.allergies) if profile.allergies else None,
                json.dumps(profile.regular_medications) if profile.regular_medications else None,
                json.dumps(profile.surgeries) if profile.surgeries else None,
                json.dumps(profile.family_history) if profile.family_history else None,
                json.dumps(profile.social_history) if profile.social_history else None,
                json.dumps(profile.patterns) if profile.patterns else None,
                profile.health_summary,
                1 if profile.intake_completed else 0,
                profile.intake_questions_asked,
                json.dumps(profile.intake_answered_question_ids) if profile.intake_answered_question_ids else None,
                profile.intake_last_question_id,
                profile.intake_started_at.isoformat() if profile.intake_started_at else None,
                profile.intake_completed_at.isoformat() if profile.intake_completed_at else None,
                profile.created_at.isoformat(),
                profile.updated_at.isoformat(),
                json.dumps(profile.intake_pending_raw) if profile.intake_pending_raw else None,
            ),
        )


def update_user_profile(
    user_id: str,
    name: Optional[str] = None,
    age: Optional[int] = None,
    gender: Optional[str] = None,
    add_conditions: Optional[List[str]] = None,
    remove_conditions: Optional[List[str]] = None,
    add_allergies: Optional[List[str]] = None,
    remove_allergies: Optional[List[str]] = None,
    add_regular_medications: Optional[List[str]] = None,
    remove_regular_medications: Optional[List[str]] = None,
    add_surgeries: Optional[List[str]] = None,
    remove_surgeries: Optional[List[str]] = None,
    add_family_history: Optional[List[str]] = None,
    remove_family_history: Optional[List[str]] = None,
    add_social_history: Optional[List[str]] = None,
    remove_social_history: Optional[List[str]] = None,
    add_patterns: Optional[List[str]] = None,
    remove_patterns: Optional[List[str]] = None,
    health_summary: Optional[str] = None,
    intake_completed: Optional[bool] = None,
    intake_questions_asked: Optional[int] = None,
    intake_answered_question_ids: Optional[List[str]] = None,
    intake_last_question_id: Optional[str] = None,
    intake_started_at: Optional[datetime] = None,
    intake_completed_at: Optional[datetime] = None,
    intake_pending_raw: Optional[Dict[str, str]] = None,
) -> UserProfile:
    """
    Update a user's profile with incremental changes.

    Items are added/removed from lists, duplicates are prevented.
    Health summary is replaced if provided.
    """
    profile = get_or_create_user_profile(user_id)

    # Helper to update a list
    def update_list(current: List[str], add: Optional[List[str]], remove: Optional[List[str]]) -> List[str]:
        result = set(current)
        if add:
            result.update(add)
        if remove:
            result -= set(remove)
        return sorted(result)  # Sort for consistency

    # Update lists
    new_conditions = update_list(profile.conditions, add_conditions, remove_conditions)
    new_allergies = update_list(profile.allergies, add_allergies, remove_allergies)
    new_regular_meds = update_list(profile.regular_medications, add_regular_medications, remove_regular_medications)
    new_surgeries = update_list(profile.surgeries, add_surgeries, remove_surgeries)
    new_family_history = update_list(profile.family_history, add_family_history, remove_family_history)
    new_social_history = update_list(profile.social_history, add_social_history, remove_social_history)
    new_patterns = update_list(profile.patterns, add_patterns, remove_patterns)

    # Update demographics if provided
    new_name = name if name is not None else profile.name
    new_age = age if age is not None else profile.age
    new_gender = gender if gender is not None else profile.gender

    # Update health summary if provided
    new_summary = health_summary if health_summary is not None else profile.health_summary
    new_intake_completed = intake_completed if intake_completed is not None else profile.intake_completed
    new_intake_questions_asked = intake_questions_asked if intake_questions_asked is not None else profile.intake_questions_asked
    new_intake_answered = (
        sorted(set(intake_answered_question_ids))
        if intake_answered_question_ids is not None
        else profile.intake_answered_question_ids
    )
    new_intake_last_question = (
        intake_last_question_id if intake_last_question_id is not None else profile.intake_last_question_id
    )
    new_intake_started_at = intake_started_at if intake_started_at is not None else profile.intake_started_at
    new_intake_completed_at = intake_completed_at if intake_completed_at is not None else profile.intake_completed_at
    new_intake_pending_raw = intake_pending_raw if intake_pending_raw is not None else profile.intake_pending_raw

    with _get_connection() as conn:
        conn.execute(
            """
            UPDATE user_profiles
            SET name = ?,
                age = ?,
                gender = ?,
                conditions_json = ?,
                allergies_json = ?,
                regular_medications_json = ?,
                surgeries_json = ?,
                family_history_json = ?,
                social_history_json = ?,
                patterns_json = ?,
                health_summary = ?,
                intake_completed = ?,
                intake_questions_asked = ?,
                intake_answered_json = ?,
                intake_last_question_id = ?,
                intake_started_at = ?,
                intake_completed_at = ?,
                intake_pending_raw_json = ?,
                updated_at = ?
            WHERE user_id = ?
            """,
            (
                new_name,
                new_age,
                new_gender,
                json.dumps(new_conditions) if new_conditions else None,
                json.dumps(new_allergies) if new_allergies else None,
                json.dumps(new_regular_meds) if new_regular_meds else None,
                json.dumps(new_surgeries) if new_surgeries else None,
                json.dumps(new_family_history) if new_family_history else None,
                json.dumps(new_social_history) if new_social_history else None,
                json.dumps(new_patterns) if new_patterns else None,
                new_summary,
                1 if new_intake_completed else 0,
                new_intake_questions_asked,
                json.dumps(new_intake_answered) if new_intake_answered else None,
                new_intake_last_question,
                new_intake_started_at.isoformat() if new_intake_started_at else None,
                new_intake_completed_at.isoformat() if new_intake_completed_at else None,
                json.dumps(new_intake_pending_raw) if new_intake_pending_raw else None,
                _utcnow().isoformat(),
                user_id,
            ),
        )

    return get_user_profile(user_id)


def reset_user_data(user_id: str) -> dict:
    """Permanently delete ALL data for a user. Dev/test reset only."""
    tables = [
        "logs",
        "medication_logs",
        "medication_reminder_actions",
        "medications",
        "ambient_events",
        "ambient_chunks",
        "ambient_sessions",
        "scheduled_checkins",
        "user_profiles",
        "cycle_day_logs",
    ]
    counts = {}
    with _get_connection() as conn:
        for table in tables:
            cursor = conn.execute(
                f"DELETE FROM {table} WHERE user_id = ?", (user_id,)
            )
            counts[table] = cursor.rowcount
    return counts


# === Cycle Day Log Storage ===

CYCLE_COLUMNS = "id, user_id, date, flow_level, is_period_day, notes, created_at, updated_at"


def _row_to_cycle_day_log(row: tuple) -> "CycleDayLog":
    from ..models import CycleDayLog
    return CycleDayLog(
        id=row[0],
        user_id=row[1],
        date=row[2],
        flow_level=row[3],
        is_period_day=bool(row[4]),
        notes=row[5],
        created_at=datetime.fromisoformat(row[6]),
        updated_at=datetime.fromisoformat(row[7]),
    )


def upsert_cycle_day_log(entry: "CycleDayLog") -> None:
    """Upsert a cycle day log (one entry per user per date)."""
    with _get_connection() as conn:
        conn.execute(
            f"""
            INSERT INTO cycle_day_logs ({CYCLE_COLUMNS})
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, date) DO UPDATE SET
                flow_level = excluded.flow_level,
                is_period_day = excluded.is_period_day,
                notes = excluded.notes,
                updated_at = excluded.updated_at
            """,
            (
                entry.id, entry.user_id, entry.date, entry.flow_level.value,
                1 if entry.is_period_day else 0, entry.notes,
                entry.created_at.isoformat(), entry.updated_at.isoformat(),
            ),
        )


def list_cycle_day_logs(user_id: str, limit: int = 365) -> list:
    """List cycle day logs for a user, most recent first."""
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT {CYCLE_COLUMNS} FROM cycle_day_logs WHERE user_id = ? ORDER BY date DESC LIMIT ?",
            (user_id, limit),
        )
        rows = cursor.fetchall()
        return [_row_to_cycle_day_log(row) for row in rows]


def list_cycle_day_logs_in_range(user_id: str, start_date: str, end_date: str) -> list:
    """List cycle day logs for a user within a date range."""
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT {CYCLE_COLUMNS} FROM cycle_day_logs WHERE user_id = ? AND date >= ? AND date <= ? ORDER BY date ASC",
            (user_id, start_date, end_date),
        )
        rows = cursor.fetchall()
        return [_row_to_cycle_day_log(row) for row in rows]


def get_cycle_day_log(user_id: str, date: str):
    """Get a specific cycle day log."""
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT {CYCLE_COLUMNS} FROM cycle_day_logs WHERE user_id = ? AND date = ?",
            (user_id, date),
        )
        row = cursor.fetchone()
        return _row_to_cycle_day_log(row) if row else None


def delete_cycle_day_log(user_id: str, date: str) -> bool:
    """Delete a cycle day log entry."""
    deleted = False
    with _get_connection() as conn:
        cursor = conn.execute(
            "DELETE FROM cycle_day_logs WHERE user_id = ? AND date = ?",
            (user_id, date),
        )
        deleted = cursor.rowcount > 0
    return deleted
