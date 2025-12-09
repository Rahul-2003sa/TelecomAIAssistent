# utils/database.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


# Base directory of the project
BASE_DIR = Path(__file__).parent.parent

# Path to the SQLite database
DB_PATH = BASE_DIR / "data" / "telecom.db"


def get_db_path() -> Path:
    """Return the path to the telecom SQLite database."""
    return DB_PATH


def get_db_uri() -> str:
    """Return the SQLAlchemy connection URI for the telecom DB."""
    return f"sqlite:///{get_db_path().as_posix()}"


def get_engine(echo: bool = False) -> Engine:
    """Create and return a SQLAlchemy engine for the telecom DB."""
    return create_engine(get_db_uri(), echo=echo, future=True)


def run_query(query: str, params: dict = None) -> List[Dict[str, Any]]:
    """
    Run raw SQL on the telecom DB and return rows as dictionaries.
    This is critical for CrewAI, LangChain, and the plan agent.
    """
    import sqlite3

    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row  # rows become dict-like
    cur = conn.cursor()

    cur.execute(query, params or {})
    rows = cur.fetchall()

    results = [dict(row) for row in rows]

    conn.close()
    return results


def get_tables() -> List[str]:
    """List all table names in the telecom DB."""
    rows = run_query("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    return [r["name"] for r in rows]
