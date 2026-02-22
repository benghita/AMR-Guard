"""Database connection and initialization for AMR-Guard."""

import sqlite3
from pathlib import Path
from contextlib import contextmanager

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DOCS_DIR = PROJECT_ROOT / "docs"
DB_PATH = DATA_DIR / "medic.db"
SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def init_database() -> None:
    """Initialize the database with schema."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with get_connection() as conn:
        with open(SCHEMA_PATH, 'r') as f:
            conn.executescript(f.read())
        conn.commit()
    print(f"Database initialized at {DB_PATH}")


@contextmanager
def get_connection():
    """Context manager for database connections."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def execute_query(query: str, params: tuple = ()) -> list[dict]:
    """Execute a query and return results as list of dicts."""
    with get_connection() as conn:
        cursor = conn.execute(query, params)
        columns = [description[0] for description in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]


def execute_insert(query: str, params: tuple = ()) -> int:
    """Execute an insert and return the last row id."""
    with get_connection() as conn:
        cursor = conn.execute(query, params)
        conn.commit()
        return cursor.lastrowid


def execute_many(query: str, params_list: list[tuple]) -> None:
    """Execute many inserts."""
    with get_connection() as conn:
        conn.executemany(query, params_list)
        conn.commit()


if __name__ == "__main__":
    init_database()
