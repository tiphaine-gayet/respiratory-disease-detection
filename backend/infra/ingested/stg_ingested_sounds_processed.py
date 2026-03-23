"""
Snowflake internal stage for preprocessed audio recordings from the app.
Mirrors PROCESSED.STG_RESPIRATORY_SOUNDS for training data.
"""

import os
from ...utils.snowflake_client import SnowflakeClient

DATABASE = os.getenv("SNOWFLAKE_DATABASE")
SCHEMA = os.getenv("SNOWFLAKE_SCHEMA_INGESTED")
STAGE_NAME = "STG_INGESTED_SOUNDS_PROCESSED"
STAGE_FULL_PATH = f"{DATABASE}.{SCHEMA}.{STAGE_NAME}"


def create_stage(client):
    client.execute(f"CREATE STAGE IF NOT EXISTS {STAGE_FULL_PATH}")
    print(f"✅ Stage {STAGE_FULL_PATH} created (if not exists).")


if __name__ == "__main__":
    print("🚀 Setting up ingested processed sounds stage...")
    with SnowflakeClient() as client:
        create_stage(client)
    print("✅ Ingested processed sounds stage ready.")
