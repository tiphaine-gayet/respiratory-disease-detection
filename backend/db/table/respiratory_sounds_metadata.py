"""
Extract audio metadata from dataset and ingest into Snowflake table.
Based on pipeline.ipynb's extract_metadata function.
"""

# TODO: refactor to use Snowflake stage files instead of local files (will require changes to extract_metadata to read from stage instead of local path)

from pathlib import Path
from scipy.io import wavfile


PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATASET_ROOT = PROJECT_ROOT / "asthma_detection_dataset" / "audio"

DATABASE = "TESSAN_HACKATHON"
SCHEMA = "RAW"
TABLE = "RESPIRATORY_SOUNDS_METADATA"


FOLDERS = {
    "asthma": "Asthma",
    "Bronchial": "Bronchial",
    "copd": "COPD",
    "healthy": "Healthy",
    "pneumonia": "Pneumonia",
}


def create_metadata_table(client):
    """Create the metadata table if it doesn't exist."""
    client.execute(f"""
        CREATE TABLE IF NOT EXISTS {DATABASE}.{SCHEMA}.{TABLE} (
            file_name VARCHAR,
            class VARCHAR,
            sample_rate INTEGER,
            duration_s FLOAT,
            n_samples INTEGER,
            amplitude_max FLOAT,
            rms FLOAT,
            PRIMARY KEY (file_name)
        )
    """)
    print(f"✅ Table {DATABASE}.{SCHEMA}.{TABLE} created (if not exists).")


def extract_metadata(filepath):
    """Extract audio metadata - returns dict with all properties and error handling."""
    try:
        sr, y = wavfile.read(filepath)  # sr=sample rate, y=audio data
        duration = len(y) / sr  # Calculate duration from samples and sample rate
        # Calculate amplitude max without numpy
        amplitude = float(max(abs(sample) for sample in y))
        # Calculate RMS without numpy: sqrt(mean(y^2))
        mean_square = sum(sample**2 for sample in y) / len(y)
        rms = float(mean_square ** 0.5)
        return {
            'sample_rate': sr,
            'duration_s': round(duration, 3),
            'n_samples': len(y),
            'amplitude_max': round(amplitude, 4),
            'rms': round(rms, 6),
            'error': None
        }
    except Exception as e:
        return {
            'sample_rate': None, 'duration_s': None,
            'n_samples': None, 'amplitude_max': None,
            'rms': None, 'error': str(e)
        }


def ingest_metadata(client):
    """Read audio files and ingest metadata into Snowflake."""
    rows_inserted = 0
    rows_error = 0

    for folder_name, class_label in FOLDERS.items():
        audio_dir = DATASET_ROOT / folder_name

        if not audio_dir.exists():
            print(f"⚠️  Directory not found: {audio_dir}")
            continue

        audio_files = list(audio_dir.glob("*.wav"))
        print(f"📁 Found {len(audio_files)} files in {folder_name}/")

        for audio_file in audio_files:
            metadata = extract_metadata(str(audio_file))

            if metadata['error'] is not None:
                print(f"❌ Error processing {audio_file.name}: {metadata['error']}")
                rows_error += 1
                continue

            # Insert into table
            client.execute(f"""
                INSERT INTO {DATABASE}.{SCHEMA}.{TABLE}
                (file_name, class, sample_rate, duration_s, n_samples, amplitude_max, rms)
                VALUES ('{audio_file.name}', '{class_label}', {metadata['sample_rate']},
                        {metadata['duration_s']}, {metadata['n_samples']},
                        {metadata['amplitude_max']}, {metadata['rms']})
            """)

            rows_inserted += 1
            if rows_inserted % 100 == 0:
                print(f"   ✓ Ingested {rows_inserted} files...")

    print(f"\n✅ Ingested {rows_inserted} audio metadata records.")
    if rows_error > 0:
        print(f"⚠️  {rows_error} files had errors and were skipped.")


if __name__ == "__main__":
    from backend.utils.snowflake_client import SnowflakeClient
    print("🚀 Ingesting audio metadata to Snowflake...")
    with SnowflakeClient() as client:
        create_metadata_table(client)
        ingest_metadata(client)
    print("✅ Metadata ingestion complete!")
