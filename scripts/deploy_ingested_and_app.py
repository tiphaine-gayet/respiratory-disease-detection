"""
Deploy all INGESTED and APP schema resources:
  - INGESTED.STG_INGESTED_SOUNDS
  - INGESTED.INGESTED_SOUNDS_METADATA
  - INGESTED.STG_INGESTED_SOUNDS_PROCESSED
  - INGESTED.INGESTED_SOUNDS_PROCESSED_METADATA
  - INGESTED.STG_INGESTED_FEATURES
  - INGESTED.INGESTED_FEATURES_METADATA
  - APP.PREDICTIONS

Required env vars (in addition to the standard SNOWFLAKE_* credentials):
  SNOWFLAKE_SCHEMA_INGESTED  — name of the ingested schema (e.g. INGESTED)
  SNOWFLAKE_SCHEMA_APP       — name of the app schema      (e.g. APP)
"""

from backend.infra.ingested.stg_ingested_sounds import create_schema, create_stage as create_stg_raw
from backend.infra.ingested.table_ingested_sounds_metadata import create_table as create_sounds_meta
from backend.infra.ingested.stg_ingested_sounds_processed import create_stage as create_stg_processed
from backend.infra.ingested.table_ingested_sounds_processed_metadata import create_table as create_sounds_processed_meta
from backend.infra.ingested.stg_ingested_features import create_stage as create_stg_features
from backend.infra.ingested.table_ingested_features_metadata import create_table as create_features_meta
from backend.infra.app.table_predictions import (
    create_schema as create_app_schema,
    create_table as create_predictions,
)
from backend.utils.snowflake_client import SnowflakeClient


def deploy():
    with SnowflakeClient() as client:
        print("\n── INGESTED schema ──────────────────────────────")
        create_schema(client)
        create_stg_raw(client)
        create_sounds_meta(client)
        create_stg_processed(client)
        create_sounds_processed_meta(client)
        create_stg_features(client)
        create_features_meta(client)

        print("\n── APP schema ───────────────────────────────────")
        create_app_schema(client)
        create_predictions(client)

    print("\n✅ All ingested & app resources deployed successfully.")


if __name__ == "__main__":
    deploy()
