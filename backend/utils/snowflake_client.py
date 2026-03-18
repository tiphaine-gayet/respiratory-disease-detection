"""
Shared Snowflake connection and SQL utilities.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
import snowflake.connector

load_dotenv()


def _get_config() -> dict:
    return {
        "account": os.environ["SNOWFLAKE_ACCOUNT"],
        "user": os.environ["SNOWFLAKE_USER"],
        "password": os.environ["SNOWFLAKE_TOKEN"],
        "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
        "database": os.getenv("SNOWFLAKE_DATABASE", "TESSAN_HACKATON"),
        "schema": os.getenv("SNOWFLAKE_SCHEMA", "ASTHMA_DETECTION"),
    }


class SnowflakeClient:
    """Context manager wrapping a Snowflake connection with utility methods."""

    def __init__(self):
        self._conn = snowflake.connector.connect(**_get_config())

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._conn.close()

    def cursor(self, cursor_class=None):
        if cursor_class:
            return self._conn.cursor(cursor_class)
        return self._conn.cursor()

    def execute(self, sql: str, params=None):
        with self._conn.cursor() as cur:
            cur.execute(sql, params)

    def execute_sql_file(self, sql_file: Path) -> None:
        """Execute all statements in a .sql file, ignoring blank lines."""
        sql = sql_file.read_text()
        statements = [s.strip() for s in sql.split(";") if s.strip()]
        with self._conn.cursor() as cur:
            for stmt in statements:
                cur.execute(stmt)
