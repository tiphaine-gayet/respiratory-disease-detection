"""
INGESTED.INFERENCE_DATA_V — one row per ingested patient recording,
with audio characteristics and feature file paths ready for model input.

Mirrors the shape of PROCESSED.TRAINING_DATA_V (same feature columns)
so the same loading code works for both training and inference.
"""

import os
from ...utils.snowflake_client import SnowflakeClient

DATABASE = os.getenv("SNOWFLAKE_DATABASE")
SCHEMA = os.getenv("SNOWFLAKE_SCHEMA_INGESTED")
VIEW = "INFERENCE_DATA_V"


def create_view(client):
    client.execute(f"""
        CREATE OR REPLACE VIEW {DATABASE}.{SCHEMA}.{VIEW} AS

        WITH features_wide AS (
            SELECT
                original_file_name,
                MAX(CASE WHEN feature_type = 'mel'       THEN npy_filename END) AS feat_mel,
                MAX(CASE WHEN feature_type = 'mfcc'      THEN npy_filename END) AS feat_mfcc,
                MAX(CASE WHEN feature_type = 'chroma'    THEN npy_filename END) AS feat_chroma,
                MAX(CASE WHEN feature_type = 'centroid'  THEN npy_filename END) AS feat_centroid,
                MAX(CASE WHEN feature_type = 'bandwidth' THEN npy_filename END) AS feat_bandwidth,
                MAX(CASE WHEN feature_type = 'zcr'       THEN npy_filename END) AS feat_zcr
            FROM {DATABASE}.{SCHEMA}.INGESTED_FEATURES_METADATA
            GROUP BY original_file_name
        )

        SELECT
            p.file_name            AS processed_file_name,
            p.original_file_name,
            p.patient_id,
            p.pharmacie_id,
            p.sample_rate,
            p.final_duration_s,
            p.n_samples,
            p.amplitude_max,
            p.rms,
            f.feat_mel,
            f.feat_mfcc,
            f.feat_chroma,
            f.feat_centroid,
            f.feat_bandwidth,
            f.feat_zcr
        FROM {DATABASE}.{SCHEMA}.INGESTED_SOUNDS_PROCESSED_METADATA p
        LEFT JOIN features_wide f ON p.original_file_name = f.original_file_name
    """)
    print(f"✅ View {DATABASE}.{SCHEMA}.{VIEW} created.")


if __name__ == "__main__":
    print("🚀 Creating inference data view...")
    with SnowflakeClient() as client:
        create_view(client)
    print("✅ Done.")
