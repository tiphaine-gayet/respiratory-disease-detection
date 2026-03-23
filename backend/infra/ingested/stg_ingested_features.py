"""
Snowflake internal stage for extracted feature arrays (.npy) from ingested patient audio.
Mirrors PROCESSED.STG_RESPIRATORY_FEATURES for training data.
"""

import os
from ...utils.snowflake_client import SnowflakeClient

DATABASE = os.getenv("SNOWFLAKE_DATABASE")
SCHEMA = os.getenv("SNOWFLAKE_SCHEMA_INGESTED")
STAGE_NAME = "STG_INGESTED_FEATURES"
STAGE_FULL_PATH = f"{DATABASE}.{SCHEMA}.{STAGE_NAME}"


def create_stage(client):
    client.execute(f"CREATE STAGE IF NOT EXISTS {STAGE_FULL_PATH}")
    print(f"✅ Stage {STAGE_FULL_PATH} created (if not exists).")


if __name__ == "__main__":
    print("🚀 Setting up ingested features stage...")
    with SnowflakeClient() as client:
        create_stage(client)
    print("✅ Ingested features stage ready.")
