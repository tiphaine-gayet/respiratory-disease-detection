"""
Extract audio metadata from dataset and ingest into Snowflake table.
"""

from pathlib import Path
import librosa
from ...utils.snowflake_client import SnowflakeClient

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATASET_ROOT = PROJECT_ROOT / "asthma_detection_dataset" / "audio"

DATABASE = "TESSAN_HACKATHON"
SCHEMA = "DATASETS"
TABLE = "RESPIRATORY_SOUNDS_METADATA"

FOLDERS = {
    "asthma": "asthma",
    "Bronchial": "bronchial",
    "copd": "copd",
    "healthy": "healthy",
    "pneumonia": "pneumonia",
}


def create_metadata_table(client):
    """Create the metadata table if it doesn't exist."""
    client.execute(f"""
        CREATE TABLE IF NOT EXISTS {DATABASE}.{SCHEMA}.{TABLE} (
            file_name VARCHAR,
            class VARCHAR,
            sample_rate INTEGER,
            duration_seconds FLOAT,
            PRIMARY KEY (file_name)
        )
    """)
    print(f"✅ Table {DATABASE}.{SCHEMA}.{TABLE} created (if not exists).")


def extract_metadata(audio_path: Path) -> dict:
    """Extract metadata from an audio file."""
    y, sr = librosa.load(str(audio_path), sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    return {
        "sample_rate": sr,
        "duration_seconds": duration,
    }


def ingest_metadata(client):
    """Read audio files and ingest metadata into Snowflake."""
    rows_inserted = 0

    for class_name, folder_path in FOLDERS.items():
        audio_dir = DATASET_ROOT / folder_path

        if not audio_dir.exists():
            print(f"⚠️  Directory not found: {audio_dir}")
            continue

        audio_files = list(audio_dir.glob("*.wav"))
        print(f"📁 Found {len(audio_files)} files in {class_name}/")

        for audio_file in audio_files:
            try:
                metadata = extract_metadata(audio_file)

                # Insert into table
                client.execute(f"""
                    INSERT INTO {DATABASE}.{SCHEMA}.{TABLE}
                    (file_name, class, sample_rate, duration_seconds)
                    VALUES ('{audio_file.name}', '{class_name}', {metadata['sample_rate']}, {metadata['duration_seconds']})
                """)

                rows_inserted += 1
                if rows_inserted % 50 == 0:
                    print(f"   ✓ Ingested {rows_inserted} files...")

            except Exception as e:
                print(f"❌ Error processing {audio_file.name}: {e}")

    print(f"✅ Ingested {rows_inserted} audio metadata records.")


if __name__ == "__main__":
    print("🚀 Ingesting audio metadata to Snowflake...")
    with SnowflakeClient() as client:
        create_metadata_table(client)
        ingest_metadata(client)
    print("✅ Metadata ingestion complete!")
