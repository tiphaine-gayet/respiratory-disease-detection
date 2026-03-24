"""
Deploy the RUN_INFERENCE stored procedure to Snowflake.
Reads proc_run_inference.sql and substitutes hardcoded paths with env-driven variables.
"""

import os
from pathlib import Path
from .stg_model import DATABASE, SCHEMA, STAGE_NAME, MODEL_VERSION
from ...utils.snowflake_client import SnowflakeClient

SCHEMA_INGESTED = os.getenv("SNOWFLAKE_SCHEMA_INGESTED")
SCHEMA_APP      = os.getenv("SNOWFLAKE_SCHEMA_APP")

PROC_NAME  = "RUN_INFERENCE"
SQL_FILE   = Path(__file__).parent / "proc_run_inference.sql"

#TODO: Refactor proc_run_inference.sql to use variables instead of hardcoded paths, then simplify this deployment script to just execute the SQL file without manual string manipulation.

def deploy_procedure(client):
    sql = SQL_FILE.read_text()
    client.execute(sql)
    print(f"✅ Procedure {DATABASE}.{SCHEMA}.{PROC_NAME} deployed.")


if __name__ == "__main__":
    print(f"🚀 Deploying {PROC_NAME} procedure...")
    with SnowflakeClient() as client:
        deploy_procedure(client)
    print("✅ Procedure deployed.")
