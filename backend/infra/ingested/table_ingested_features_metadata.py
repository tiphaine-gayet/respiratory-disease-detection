"""
Metadata table for extracted feature arrays (.npy) from ingested patient audio.
Mirrors PROCESSED.RESPIRATORY_FEATURES_METADATA, extended with patient and pharmacy context.
Populated by the feature extraction pipeline after preprocessing.
"""

import os
from ...utils.snowflake_client import SnowflakeClient

DATABASE = os.getenv("SNOWFLAKE_DATABASE")
SCHEMA = os.getenv("SNOWFLAKE_SCHEMA_INGESTED")
TABLE = "INGESTED_FEATURES_METADATA"

# Feature types produced by the extraction pipeline
# FEATURE_TYPES = ("mel", "mfcc", "chroma", "centroid", "bandwidth", "zcr")


def create_table(client):
    client.execute(f"""
        CREATE TABLE IF NOT EXISTS {DATABASE}.{SCHEMA}.{TABLE} (
            file_name              VARCHAR        NOT NULL,   -- FK → INGESTED_SOUNDS_PROCESSED_METADATA.file_name
            original_file_name     VARCHAR        NOT NULL,   -- FK → INGESTED_SOUNDS_METADATA.file_name
            patient_id             VARCHAR(15)    NOT NULL,
            pharmacie_id           VARCHAR(14),
            feature_type           VARCHAR        NOT NULL,   -- mel | mfcc | chroma | centroid | bandwidth | zcr
            npy_filename           VARCHAR        NOT NULL,   -- .npy file path in STG_INGESTED_FEATURES
            extracted_at           TIMESTAMP_NTZ  NOT NULL DEFAULT CURRENT_TIMESTAMP(),
            PRIMARY KEY (file_name, feature_type),
            FOREIGN KEY (original_file_name) REFERENCES {DATABASE}.{SCHEMA}.INGESTED_SOUNDS_METADATA(file_name)
        )
    """)
    print(f"✅ Table {DATABASE}.{SCHEMA}.{TABLE} created (if not exists).")


if __name__ == "__main__":
    print("🚀 Setting up ingested features metadata table...")
    with SnowflakeClient() as client:
        create_table(client)
    print("✅ Ingested features metadata table ready.")
