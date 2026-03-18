"""
Shared Snowflake connection and SQL utilities.
"""

import os
from contextlib import contextmanager
from pathlib import Path

from dotenv import load_dotenv
import snowflake.connector
from snowflake.connector import DictCursor

load_dotenv()

SQL_DIR = Path(__file__).parent


def get_config() -> dict:
    return {
        "account": os.environ["SNOWFLAKE_ACCOUNT"],
        "user": os.environ["SNOWFLAKE_USER"],
        "password": os.environ["SNOWFLAKE_TOKEN"],
        "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
        "database": os.getenv("SNOWFLAKE_DATABASE", "TESSAN_HACKATON"),
        "schema": os.getenv("SNOWFLAKE_SCHEMA", "ASTHMA_DETECTION"),
    }


@contextmanager
def get_connection():
    """Context manager that yields an open Snowflake connection."""
    conn = snowflake.connector.connect(**get_config())
    try:
        yield conn
    finally:
        conn.close()


def execute_sql_file(conn, sql_file: Path) -> None:
    """Execute all statements in a .sql file, ignoring blank lines."""
    sql = sql_file.read_text()
    statements = [s.strip() for s in sql.split(";") if s.strip()]
    with conn.cursor() as cur:
        for stmt in statements:
            cur.execute(stmt)


def fetchall_as_dicts(conn, query: str, params=None) -> list[dict]:
    """Run a query and return results as a list of dicts."""
    with conn.cursor(DictCursor) as cur:
        cur.execute(query, params)
        return cur.fetchall()
