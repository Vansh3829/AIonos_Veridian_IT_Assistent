"""
SQLite database layer for the Veridian IT Service Agent.

Tables:
  1. ticket_queue - Existing ticket history from the Data Pack.
  2. requests     - Employee requests submitted through the agent.
  3. tickets      - New tickets created by the agent during escalation.
  4. audit_logs   - One row per processed request.

Kept intentionally simple: plain sqlite3, no ORM.
"""

import sqlite3
import os
import pandas as pd
from datetime import datetime


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DB_PATH = os.path.join(
    BASE_DIR,
    "veridian.db"
)

TICKETS_CSV = os.path.join(
    BASE_DIR,
    "data",
    "tickets.csv"
)


# ---------------------------------------------------------------------------
# Database Connection
# ---------------------------------------------------------------------------

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# Database Initialization
# ---------------------------------------------------------------------------

def init_db():
    """
    Create all required tables.

    If an older version of the database already exists, migrate the
    tickets table by adding any newly required columns.
    """

    conn = get_connection()
    cur = conn.cursor()

    # -----------------------------------------------------------------------
    # Existing Data Pack ticket queue
    # -----------------------------------------------------------------------

    cur.execute("""
        CREATE TABLE IF NOT EXISTS ticket_queue (
            ticket_id TEXT PRIMARY KEY,
            employee TEXT,
            issue_summary TEXT,
            status TEXT,
            state TEXT
        )
    """)

    # -----------------------------------------------------------------------
    # Employee Requests
    # -----------------------------------------------------------------------

    cur.execute("""
        CREATE TABLE IF NOT EXISTS requests (
            request_id TEXT PRIMARY KEY,
            employee_id TEXT NOT NULL,
            employee_name TEXT NOT NULL,
            employee_email TEXT NOT NULL,
            issue TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    # -----------------------------------------------------------------------
    # Agent-created Tickets
    # -----------------------------------------------------------------------

    cur.execute("""
        CREATE TABLE IF NOT EXISTS tickets (
            ticket_id TEXT PRIMARY KEY,
            request_id TEXT,
            employee_id TEXT,
            employee_name TEXT,
            employee_email TEXT,
            category TEXT,
            issue TEXT,
            reason TEXT,
            status TEXT,
            created_at TEXT
        )
    """)

    # -----------------------------------------------------------------------
    # Audit Logs
    # -----------------------------------------------------------------------

    cur.execute("""
        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id TEXT,
            timestamp TEXT,
            intent TEXT,
            category TEXT,
            policy_used TEXT,
            ticket_context TEXT,
            grounding_result TEXT,
            grounding_reason TEXT,
            decision TEXT,
            action TEXT,
            ticket_id TEXT,
            response TEXT
        )
    """)

    conn.commit()

    # -----------------------------------------------------------------------
    # Migrate old tickets table if necessary
    # -----------------------------------------------------------------------

    cur.execute("PRAGMA table_info(tickets)")

    existing_columns = {
        row["name"]
        for row in cur.fetchall()
    }

    required_columns = {
        "request_id": "TEXT",
        "employee_id": "TEXT",
        "employee_name": "TEXT",
        "employee_email": "TEXT",
        "category": "TEXT",
        "issue": "TEXT",
        "reason": "TEXT",
        "status": "TEXT",
        "created_at": "TEXT",
    }

    for column_name, column_type in required_columns.items():

        if column_name not in existing_columns:

            cur.execute(
                f"""
                ALTER TABLE tickets
                ADD COLUMN {column_name} {column_type}
                """
            )

    conn.commit()

    # -----------------------------------------------------------------------
    # Load Data Pack ticket queue if it is empty
    # -----------------------------------------------------------------------

    cur.execute(
        "SELECT COUNT(*) AS c FROM ticket_queue"
    )

    count = cur.fetchone()["c"]

    if count == 0 and os.path.exists(TICKETS_CSV):

        df = pd.read_csv(TICKETS_CSV)

        df.to_sql(
            "ticket_queue",
            conn,
            if_exists="append",
            index=False
        )

        conn.commit()

    conn.close()


# ---------------------------------------------------------------------------
# Request ID
# ---------------------------------------------------------------------------

def get_next_request_id():
    """
    Generate the next sequential request ID.

    Example:
        REQ-001
        REQ-002
        REQ-003
    """

    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT request_id
        FROM requests
        ORDER BY rowid DESC
        LIMIT 1
    """)

    row = cur.fetchone()

    conn.close()

    if row is None:
        return "REQ-001"

    try:

        number = int(
            row["request_id"].split("-")[1]
        )

        return f"REQ-{number + 1:03d}"

    except (ValueError, IndexError):

        return "REQ-001"


# ---------------------------------------------------------------------------
# Create Employee Request
# ---------------------------------------------------------------------------

def create_request(
    employee_id,
    employee_name,
    employee_email,
    issue
):
    """
    Store a new employee request.

    The request ID is generated automatically.
    """

    request_id = get_next_request_id()

    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO requests (
            request_id,
            employee_id,
            employee_name,
            employee_email,
            issue,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            request_id,
            employee_id,
            employee_name,
            employee_email,
            issue,
            datetime.now().isoformat(
                timespec="seconds"
            )
        )
    )

    conn.commit()
    conn.close()

    return request_id


# ---------------------------------------------------------------------------
# Ticket ID
# ---------------------------------------------------------------------------

def get_next_ticket_id():
    """
    Generate the next agent-created ticket ID.

    Example:
        TKT-1001
        TKT-1002
        TKT-1003
    """

    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        "SELECT COUNT(*) AS c FROM tickets"
    )

    count = cur.fetchone()["c"]

    conn.close()

    return f"TKT-{1001 + count}"


# ---------------------------------------------------------------------------
# Create Agent Ticket
# ---------------------------------------------------------------------------

def create_ticket(
    request_id,
    employee_id,
    employee_name,
    employee_email,
    category,
    issue,
    reason,
    status="ESCALATED"
):
    """
    Create a new ticket when the agent escalates a request.
    """

    ticket_id = get_next_ticket_id()

    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO tickets (
            ticket_id,
            request_id,
            employee_id,
            employee_name,
            employee_email,
            category,
            issue,
            reason,
            status,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            ticket_id,
            request_id,
            employee_id,
            employee_name,
            employee_email,
            category,
            issue,
            reason,
            status,
            datetime.now().isoformat(
                timespec="seconds"
            )
        )
    )

    conn.commit()
    conn.close()

    return ticket_id


# ---------------------------------------------------------------------------
# Audit Log
# ---------------------------------------------------------------------------

def log_audit(entry: dict):
    """
    Insert one audit log row.
    """

    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO audit_logs (
            request_id,
            timestamp,
            intent,
            category,
            policy_used,
            ticket_context,
            grounding_result,
            grounding_reason,
            decision,
            action,
            ticket_id,
            response
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            entry.get("request_id"),
            datetime.now().isoformat(
                timespec="seconds"
            ),
            entry.get("intent"),
            entry.get("category"),
            entry.get("policy_used"),
            entry.get("ticket_context"),
            entry.get("grounding_result"),
            entry.get("grounding_reason"),
            entry.get("decision"),
            entry.get("action"),
            entry.get("ticket_id"),
            entry.get("response"),
        ),
    )

    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Get Employee Requests
# ---------------------------------------------------------------------------

def get_all_requests():

    conn = get_connection()

    df = pd.read_sql_query(
        """
        SELECT *
        FROM requests
        ORDER BY created_at DESC
        """,
        conn
    )

    conn.close()

    return df


# ---------------------------------------------------------------------------
# Get Agent-created Tickets
# ---------------------------------------------------------------------------

def get_all_tickets():

    conn = get_connection()

    df = pd.read_sql_query(
        """
        SELECT
            ticket_id,
            request_id,
            employee_id,
            employee_name,
            employee_email,
            category,
            issue,
            reason,
            status,
            created_at
        FROM tickets
        ORDER BY created_at DESC
        """,
        conn
    )

    conn.close()

    return df

# ---------------------------------------------------------------------------
# Get Audit Logs
# ---------------------------------------------------------------------------

def get_all_audit_logs():

    conn = get_connection()

    df = pd.read_sql_query(
        """
        SELECT *
        FROM audit_logs
        ORDER BY id DESC
        """,
        conn
    )

    conn.close()

    return df


# ---------------------------------------------------------------------------
# Get Existing Data Pack Ticket Queue
# ---------------------------------------------------------------------------

def get_ticket_queue():

    conn = get_connection()

    df = pd.read_sql_query(
        """
        SELECT *
        FROM ticket_queue
        """,
        conn
    )

    conn.close()

    return df


# ---------------------------------------------------------------------------
# Search Existing Ticket Queue
# ---------------------------------------------------------------------------

def search_ticket_queue(keywords):
    """
    Search the existing Data Pack ticket queue.

    Active tickets are returned before closed tickets.
    """

    if not keywords:
        return []

    conn = get_connection()

    df = pd.read_sql_query(
        """
        SELECT *
        FROM ticket_queue
        """,
        conn
    )

    conn.close()

    if df.empty:
        return []

    keywords_lower = [
        k.lower()
        for k in keywords
        if k
    ]

    def matches(row):

        text = str(
            row["issue_summary"]
        ).lower()

        return any(
            keyword in text
            for keyword in keywords_lower
        )

    matched = df[
        df.apply(matches, axis=1)
    ].copy()

    if matched.empty:
        return []

    # Active tickets first, historical/closed tickets second.
    matched["sort_key"] = matched["state"].apply(
        lambda state: 0
        if str(state).upper() == "ACTIVE"
        else 1
    )

    matched = (
        matched
        .sort_values("sort_key")
        .drop(columns=["sort_key"])
    )

    return matched.head(3).to_dict(
        orient="records"
    )


# ---------------------------------------------------------------------------
# Direct execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    init_db()

    print(
        f"Database initialized at: {DB_PATH}"
    )

    print(
        get_ticket_queue()
    )