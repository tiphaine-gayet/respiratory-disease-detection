"""
Snowflake internal stage for raw audio recordings submitted by patients via the app.
Mirrors RAW.STG_RESPIRATORY_SOUNDS for training data.
"""

import os
from ...utils.snowflake_client import SnowflakeClient

DATABASE = os.getenv("SNOWFLAKE_DATABASE")
SCHEMA = os.getenv("SNOWFLAKE_SCHEMA_INGESTED")
STAGE_NAME = "STG_INGESTED_SOUNDS"
STAGE_FULL_PATH = f"{DATABASE}.{SCHEMA}.{STAGE_NAME}"


def create_schema(client):
    client.execute(f"CREATE SCHEMA IF NOT EXISTS {DATABASE}.{SCHEMA}")
    print(f"✅ Schema {DATABASE}.{SCHEMA} created (if not exists).")


def create_stage(client):
    client.execute(f"CREATE STAGE IF NOT EXISTS {STAGE_FULL_PATH}")
    print(f"✅ Stage {STAGE_FULL_PATH} created (if not exists).")


if __name__ == "__main__":
    print("🚀 Setting up ingested sounds stage...")
    with SnowflakeClient() as client:
        create_schema(client)
        create_stage(client)
    print("✅ Ingested sounds stage ready.")
