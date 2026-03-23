"""
Metadata table for raw audio files recorded by patients via the app.
Mirrors RAW.RESPIRATORY_SOUNDS_METADATA, extended with patient and pharmacy context.
Populated at recording time by the app backend.
"""

import os
from ...utils.snowflake_client import SnowflakeClient

DATABASE = os.getenv("SNOWFLAKE_DATABASE")
SCHEMA = os.getenv("SNOWFLAKE_SCHEMA_INGESTED")
TABLE = "INGESTED_SOUNDS_METADATA"


def create_table(client):
    client.execute(f"""
        CREATE TABLE IF NOT EXISTS {DATABASE}.{SCHEMA}.{TABLE} (
            file_name         VARCHAR        NOT NULL,
            patient_id        VARCHAR(15)    NOT NULL,  -- numéro de sécurité sociale (13 digits + 2-char key)
            pharmacie_siret   VARCHAR(14),              -- SIRET from PUBLIC.PHARMACIES_FRANCE (NULL if unavailable)
            recorded_at       TIMESTAMP_NTZ  NOT NULL DEFAULT CURRENT_TIMESTAMP(),
            sample_rate       INTEGER,
            duration_s        FLOAT,
            n_samples         INTEGER,
            amplitude_max     FLOAT,
            rms               FLOAT,
            PRIMARY KEY (file_name)
        )
    """)
    print(f"✅ Table {DATABASE}.{SCHEMA}.{TABLE} created (if not exists).")


if __name__ == "__main__":
    print("🚀 Setting up ingested sounds metadata table...")
    with SnowflakeClient() as client:
        create_table(client)
    print("✅ Ingested sounds metadata table ready.")
