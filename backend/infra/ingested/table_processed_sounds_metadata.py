"""
Metadata table for preprocessed patient audio files.
Mirrors PROCESSED.RESPIRATORY_SOUNDS_METADATA, extended with patient and pharmacy context.
Populated by the preprocessing pipeline when a new patient recording is processed.
"""

import os
from ...utils.snowflake_client import SnowflakeClient

DATABASE = os.getenv("SNOWFLAKE_DATABASE")
SCHEMA = os.getenv("SNOWFLAKE_SCHEMA_INGESTED")
TABLE = "PROCESSED_SOUNDS_METADATA"


def create_table(client):
    client.execute(f"""
        CREATE TABLE IF NOT EXISTS {DATABASE}.{SCHEMA}.{TABLE} (
            file_name              VARCHAR        NOT NULL,   -- processed filename in STG_INGESTED_SOUNDS_PROCESSED
            original_file_name     VARCHAR        NOT NULL,   -- FK → INGESTED_SOUNDS_METADATA.file_name
            patient_id             VARCHAR(15)    NOT NULL,
            pharmacie_id           VARCHAR(14),
            action                 VARCHAR,                   -- preprocessing action (e.g. 'strip_silence', 'pad')
            original_duration_s    FLOAT,
            stripped_duration_s    FLOAT,
            final_duration_s       FLOAT,
            leading_silence_s      FLOAT,
            trailing_silence_s     FLOAT,
            sample_rate            INTEGER,
            n_samples              INTEGER,
            amplitude_max          FLOAT,
            rms                    FLOAT,
            mel_npy_filename       VARCHAR,                   -- filename in STG_MEL_NPY (e.g. <patient_id>_<timestamp>.npy)
            processed_at           TIMESTAMP_NTZ  NOT NULL DEFAULT CURRENT_TIMESTAMP(),
            PRIMARY KEY (file_name),
            FOREIGN KEY (original_file_name) REFERENCES {DATABASE}.{SCHEMA}.INGESTED_SOUNDS_METADATA(file_name)
        )
    """)
    print(f"✅ Table {DATABASE}.{SCHEMA}.{TABLE} created (if not exists).")

    # Migration: add mel_npy_filename if table already exists without it
    client.execute(f"""
        ALTER TABLE {DATABASE}.{SCHEMA}.{TABLE}
        ADD COLUMN IF NOT EXISTS mel_npy_filename VARCHAR
    """)

    # Migration: remove old mel_spectrogram VARIANT column if still present
    client.execute(f"""
        ALTER TABLE {DATABASE}.{SCHEMA}.{TABLE}
        DROP COLUMN IF EXISTS mel_spectrogram
    """)
    print(f"✅ Table {DATABASE}.{SCHEMA}.{TABLE} schema up to date.")


if __name__ == "__main__":
    print("🚀 Setting up ingested processed sounds metadata table...")
    with SnowflakeClient() as client:
        create_table(client)
    print("✅ Ingested processed sounds metadata table ready.")
