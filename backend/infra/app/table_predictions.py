"""
Predictions table — records every inference made by the model from the app.
Each row captures the patient, pharmacy, model output probabilities, confidence,
and the resulting clinical action.
"""

import os
from ...utils.snowflake_client import SnowflakeClient

DATABASE = os.getenv("SNOWFLAKE_DATABASE")
SCHEMA = os.getenv("SNOWFLAKE_SCHEMA_APP")
TABLE = "PREDICTIONS"

# Clinical action thresholds are enforced at app level; only valid labels stored here.
# VALID_ACTIONS = ("RAS", "SUIVI_48H", "URGENCE")

def create_table(client):
    client.execute(f"""
        CREATE TABLE IF NOT EXISTS {DATABASE}.{SCHEMA}.{TABLE} (
            prediction_id     VARCHAR        NOT NULL DEFAULT UUID_STRING(),
            predicted_at      TIMESTAMP_NTZ  NOT NULL DEFAULT CURRENT_TIMESTAMP(),

            -- Patient & pharmacy
            patient_id        VARCHAR(15)    NOT NULL,  -- numéro de sécurité sociale
            pharmacie_id   VARCHAR(14),                 -- OSM_ID from APP.PHARMACIES_FRANCE

            -- Source audio (FK to ingested layer)
            audio_file_name   VARCHAR        NOT NULL,  -- FK → INGESTED.INGESTED_SOUNDS_METADATA.file_name

            -- Model output: per-class probabilities (0–1, should sum to 1)
            pct_asthma        FLOAT          NOT NULL,
            pct_copd          FLOAT          NOT NULL,
            pct_bronchial     FLOAT          NOT NULL,
            pct_pneumonia     FLOAT          NOT NULL,
            pct_healthy       FLOAT          NOT NULL,

            -- Overall model confidence (0–1)
            pct_confiance     FLOAT          NOT NULL,

            -- Clinical action derived from prediction
            action            VARCHAR        NOT NULL,  -- RAS | SUIVI_48H | URGENCE
            detailed_action   VARCHAR                   -- e.g. "Aucun suivi nécessaire", "Suivi dans les 48h", "Orientation vers urgence"

            -- Model traceability
            model_version     VARCHAR,

            PRIMARY KEY (prediction_id)
        )
    """)
    print(f"✅ Table {DATABASE}.{SCHEMA}.{TABLE} created (if not exists).")


if __name__ == "__main__":
    print("🚀 Setting up predictions table...")
    with SnowflakeClient() as client:
        create_table(client)
    print("✅ Predictions table ready.")
