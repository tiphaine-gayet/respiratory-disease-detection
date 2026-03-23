#!/usr/bin/env python
"""Deploy dual audio UDFs to Snowflake.

This script deploys two UDFs from distinct files:
1. PROCESS_FILE_UDF from infra.processed.udf_process_file
2. EXTRACT_FEATURES_UDF from infra.processed.udf_extract_features

Usage:
    python backend/infra/deploy/deploy_udf_audio_processing.py

Environment variables:
    SNOWFLAKE_ACCOUNT: Snowflake account identifier
    SNOWFLAKE_USER: Snowflake user
    SNOWFLAKE_PASSWORD: Snowflake password
    SNOWFLAKE_WAREHOUSE: Warehouse to use
    SNOWFLAKE_DATABASE: Database (default: M2_ISD_EQUIPE_1_DB)
    SNOWFLAKE_SCHEMA: Schema (default: PROCESSED)
"""

import os
import sys
from pathlib import Path

# Add project root to path so imports like infra.processed.* work.
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from infra.processed.udf_process_file import deploy_udf_process_file
from infra.processed.udf_extract_features import deploy_udf_extract_features


def main():
    """Deploy both UDFs."""
    try:
        from snowflake.snowpark import Session
        
        # Get credentials from environment or config
        connection_params = {
            "account": os.getenv("SNOWFLAKE_ACCOUNT"),
            "user": os.getenv("SNOWFLAKE_USER"),
            "password": os.getenv("SNOWFLAKE_PASSWORD"),
            "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
            "database": os.getenv("SNOWFLAKE_DATABASE", "M2_ISD_EQUIPE_1_DB"),
            "schema": os.getenv("SNOWFLAKE_SCHEMA", "PROCESSED"),
        }
        
        # Validate credentials
        if not all([connection_params["account"], connection_params["user"], connection_params["password"]]):
            print("❌ Missing Snowflake credentials")
            print("   Set: SNOWFLAKE_ACCOUNT, SNOWFLAKE_USER, SNOWFLAKE_PASSWORD")
            return 1
        
        print("Connecting to Snowflake...")
        session = Session.builder.configs(connection_params).create()
        print(f"✓ Connected to {connection_params['account']}")

        # Deploy UDF 1 from udf_process_file.py
        print("\nDeploying UDF 1/2: PROCESS_FILE_UDF")
        deploy_udf_process_file(session, udf_name="PROCESS_FILE_UDF")

        # Deploy UDF 2 from udf_extract_features.py
        print("\nDeploying UDF 2/2: EXTRACT_FEATURES_UDF")
        deploy_udf_extract_features(session, udf_name="EXTRACT_FEATURES_UDF")
                
        # Display usage
        print(f"\n" + "="*70)
        print("Dual UDF Deployment Complete")
        print("="*70)
        print(f"\nUsage:")
        print("  SELECT PROCESS_FILE_UDF(file_name, stage_path, class_name) FROM your_table;")
        print("  SELECT EXTRACT_FEATURES_UDF(processed_path, stage_path, file_name, class_name, save_to_stage, output_stage) FROM your_table;")
        print(f"\nExample:")
        print("  SELECT PROCESS_FILE_UDF(")
        print(f"      'patient_001.wav',")
        print(f"      '@M2_ISD_EQUIPE_1_DB.PROCESSED.STG_RESPIRATORY_SOUNDS/asthma/',")
        print(f"      'Asthma'")
        print(f"  ) AS result;")
        print("\nSecond UDF example:")
        print("  SELECT EXTRACT_FEATURES_UDF(")
        print("      'patient_001.wav',")
        print("      '@M2_ISD_EQUIPE_1_DB.PROCESSED.STG_RESPIRATORY_SOUNDS/asthma/',")
        print("      'patient_001.wav',")
        print("      'Asthma',")
        print("      FALSE,")
        print("      '@M2_ISD_EQUIPE_1_DB.PROCESSED.STG_RESPIRATORY_FEATURES/asthma/'")
        print("  ) AS result;")

        print("\nUDF sources:")
        print("  - infra/processed/udf_process_file.py")
        print("  - infra/processed/udf_extract_features.py")
        print("="*70)
        
        session.close()
        return 0
        
    except Exception as e:
        print(f"❌ Deployment failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
