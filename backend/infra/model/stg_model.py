"""
Snowflake internal stage for models and inference procedures.
"""

import subprocess
import os
from pathlib import Path
from ...utils.snowflake_client import SnowflakeClient

DATABASE = os.getenv("SNOWFLAKE_DATABASE")
SCHEMA = os.getenv("SNOWFLAKE_SCHEMA_MODEL")
STAGE_NAME = "STG_MODEL"
STAGE_PATH = f"{DATABASE}.{SCHEMA}.{STAGE_NAME}"

SNOWSQL_PATH = os.getenv("SNOWSQL_PATH", "snowsql")

MODEL_VERSION = "v0"
MODEL_LOCAL_PATH = Path(__file__).parent.parent.parent / "models" / MODEL_VERSION
MODEL_STAGE_PATH = f"{DATABASE}.{SCHEMA}.{STAGE_NAME}/{MODEL_VERSION}"


def create_schema(client):
    client.execute(f"CREATE SCHEMA IF NOT EXISTS {DATABASE}.{SCHEMA}")
    print(f"✅ Schema {DATABASE}.{SCHEMA} created (if not exists).")


def create_stage(client):
    client.execute(f"CREATE STAGE IF NOT EXISTS {STAGE_PATH}")
    print(f"✅ Stage {STAGE_PATH} created (if not exists).")
    cmd = [
        SNOWSQL_PATH,
            "-q", f"PUT file://{MODEL_LOCAL_PATH}/* @{MODEL_STAGE_PATH} AUTO_COMPRESS=FALSE OVERWRITE=TRUE"
        ]
    
    try:        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✅ Uploaded model files to {MODEL_STAGE_PATH}")
        if result.stdout:
            print(f"   {result.stdout.strip()}")
    except subprocess.CalledProcessError as e:
            print(f"❌ Failed to upload model files to {MODEL_STAGE_PATH}: {e.stderr}")
            raise


if __name__ == "__main__":
    print("🚀 Setting up model stage...")
    with SnowflakeClient() as client:
        create_schema(client)
        create_stage(client)
    print("✅ Model stage ready.")
