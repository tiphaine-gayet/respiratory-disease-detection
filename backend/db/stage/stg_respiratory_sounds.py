import subprocess
import os
from pathlib import Path
from ...utils.snowflake_client import SnowflakeClient

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATASET_ROOT = PROJECT_ROOT / "asthma_detection_dataset" / "audio"

DATABASE = os.getenv("SNOWFLAKE_DATABASE", "TESSAN_HACKATHON")
SCHEMA = "DATASETS"
STAGE_NAME = "STG_RESPIRATORY_SOUNDS"
STAGE_FULL_PATH = f"{DATABASE}.{SCHEMA}.{STAGE_NAME}"


SNOWSQL_PATH = os.getenv("SNOWSQL_PATH", "snowsql")

FOLDERS = {
    "asthma": "asthma",
    "Bronchial": "bronchial",
    "copd": "copd",
    "healthy": "healthy",
    "pneumonia": "pneumonia",
}

def create_stage(client):
    """Create the Snowflake stage if it doesn't exist."""
    client.execute(f"""
        CREATE OR ALTER STAGE {STAGE_FULL_PATH}
    """)
    print(f"✅ Stage {STAGE_FULL_PATH} created (if not exists).")

def upload_audio_files_to_stage():
    """Upload audio files to Snowflake stage using snowsql."""
    for folder, path in FOLDERS.items():
        local_path = str(DATASET_ROOT / folder)
        stage_path = f"{STAGE_FULL_PATH}/{path}"

        cmd = [
            SNOWSQL_PATH,
            "-q", f"PUT file://{local_path}/* @{stage_path} AUTO_COMPRESS=FALSE"
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"✅ Uploaded {folder} to {stage_path}")
            if result.stdout:
                print(f"   {result.stdout.strip()}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to upload {folder}: {e.stderr}")
            raise

if __name__ == "__main__":
    print("🚀 Transferring audio files to Snowflake stage...")
    with SnowflakeClient() as client:
        create_stage(client)
    upload_audio_files_to_stage()
    print("✅ Files transferred to stage successfully!")
