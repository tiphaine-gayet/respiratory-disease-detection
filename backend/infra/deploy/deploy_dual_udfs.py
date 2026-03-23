"""
Deployment script for dual audio processing UDFs
=================================================

This script deploys two separate UDFs to Snowflake:
1. PROCESS_FILE_UDF - Handles audio preprocessing
2. EXTRACT_FEATURES_UDF - Handles feature extraction

Usage:
    python scripts/deploy_dual_udfs.py

Environment variables required:
    SNOWFLAKE_ACCOUNT
    SNOWFLAKE_USER
    SNOWFLAKE_PASSWORD
    SNOWFLAKE_DATABASE (optional, defaults to M2_ISD_EQUIPE_1_DB)
    SNOWFLAKE_SCHEMA (optional, defaults to PROCESSED)
    SNOWFLAKE_WAREHOUSE (optional, defaults to COMPUTE_WH)
"""

import os
import sys
import json
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "infra" / "processed"))

from snowflake.snowpark import Session
from udf_process_file import deploy_udf_process_file
from udf_extract_features import (
    deploy_udf_extract_features,
    deploy_udf_extract_features_simple
)


def setup_snowflake_session():
    """
    Create a Snowflake session from environment variables.
    
    Returns:
        Session object
    
    Raises:
        ValueError if required environment variables are missing
    """
    required_vars = ["SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD"]
    missing = [var for var in required_vars if var not in os.environ]
    
    if missing:
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
    
    connection_params = {
        "account": os.environ["SNOWFLAKE_ACCOUNT"],
        "user": os.environ["SNOWFLAKE_USER"],
        "password": os.environ["SNOWFLAKE_PASSWORD"],
        "database": os.environ.get("SNOWFLAKE_DATABASE", "M2_ISD_EQUIPE_1_DB"),
        "schema": os.environ.get("SNOWFLAKE_SCHEMA", "PROCESSED"),
        "warehouse": os.environ.get("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
    }
    
    return Session.builder.configs(connection_params).create()


def setup_metadata_tables(session):
    """
    Create additional metadata tables for processing results.
    
    Note: RESPIRATORY_SOUNDS_METADATA is created automatically by PROCESS_FILE_UDF.
    
    Args:
        session: Snowflake session
    """
    # RESPIRATORY_FEATURES_METADATA table
    session.sql("""
        CREATE TABLE IF NOT EXISTS M2_ISD_EQUIPE_1_DB.PROCESSED.RESPIRATORY_FEATURES_METADATA (
            FILE_NAME VARCHAR,
            CLASS VARCHAR,
            ACTION VARCHAR,
            ORIGINAL_DURATION_S FLOAT,
            STRIPPED_DURATION_S FLOAT,
            FINAL_DURATION_S FLOAT,
            LEADING_SILENCE_S FLOAT,
            TRAILING_SILENCE_S FLOAT,
            SAMPLE_RATE INT,
            N_SAMPLES INT,
            AMPLITUDE_MAX FLOAT,
            RMS FLOAT,
            CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (FILE_NAME, CLASS)
        )
    """).collect()
    print("✓ Table RESPIRATORY_FEATURES_METADATA created/verified")
    
    # RESPIRATORY_FEATURES_EXTRACTED table
    session.sql("""
        CREATE TABLE IF NOT EXISTS M2_ISD_EQUIPE_1_DB.PROCESSED.RESPIRATORY_FEATURES_EXTRACTED (
            FILE_NAME VARCHAR,
            CLASS VARCHAR,
            FEATURE_TYPE VARCHAR,
            FEATURE_SHAPE ARRAY,
            FEATURE_DTYPE VARCHAR,
            N_FRAMES INT,
            N_COEFFICIENTS INT,
            NPY_FILENAME VARCHAR,
            CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (FILE_NAME, CLASS, FEATURE_TYPE)
        )
    """).collect()
    print("✓ Table RESPIRATORY_FEATURES_EXTRACTED created/verified")


def main():
    """Deploy both UDFs to Snowflake."""
    try:
        print("=" * 70)
        print("DEPLOYING DUAL AUDIO PROCESSING UDFs")
        print("=" * 70)
        
        # Create session
        print("\n[1/5] Connecting to Snowflake...")
        session = setup_snowflake_session()
        print("✓ Connected to Snowflake")
        
        # Display connection info
        info = session.sql("SELECT CURRENT_DATABASE(), CURRENT_SCHEMA(), CURRENT_WAREHOUSE()").collect()
        print(f"  Database: {info[0][0]}")
        print(f"  Schema: {info[0][1]}")
        print(f"  Warehouse: {info[0][2]}")
        
        # Deploy UDF 1: Process File
        print("\n[2/5] Deploying PROCESS_FILE_UDF...")
        deploy_udf_process_file(session, "PROCESS_FILE_UDF")
        
        # Deploy UDF 2: Extract Features (full version)
        print("\n[3/5] Deploying EXTRACT_FEATURES_UDF...")
        deploy_udf_extract_features(session, "EXTRACT_FEATURES_UDF")
        
        # Deploy UDF 3: Extract Features (simple version)
        print("\n[4/5] Deploying EXTRACT_FEATURES_SIMPLE_UDF...")
        deploy_udf_extract_features_simple(session, "EXTRACT_FEATURES_SIMPLE_UDF")
        
        # Setup metadata tables
        print("\n[5/5] Setting up metadata tables...")
        setup_metadata_tables(session)
        
        # Print summary
        print("\n" + "=" * 70)
        print("DEPLOYMENT COMPLETE ✓")
        print("=" * 70)
        
        print("\nRegistered UDFs:")
        print("  1. PROCESS_FILE_UDF(file_name, stage_name, class_name)")
        print("     → Preprocesses audio + auto-inserts metadata into RESPIRATORY_SOUNDS_METADATA")
        print("  2. EXTRACT_FEATURES_UDF(path, stage, file_name, class_name, save, out_stage)")
        print("     → Returns full feature data with optional stage save")
        print("  3. EXTRACT_FEATURES_SIMPLE_UDF(path, stage_name)")
        print("     → Returns feature metadata only (lightweight)")
        
        print("\nMetadata tables created:")
        print("  - RESPIRATORY_SOUNDS_METADATA (auto-populated by PROCESS_FILE_UDF)")
        print("  - RESPIRATORY_FEATURES_METADATA")
        print("  - RESPIRATORY_FEATURES_EXTRACTED")
        
        print("\nUsage examples:")
        print("""
  -- Process single audio file (metadata auto-inserted)
  SELECT PROCESS_FILE_UDF(
    'filename.wav',
    '@STG_RESPIRATORY_SOUNDS/asthma/',
    'Asthma'
  );

  -- Extract features from processed audio
  SELECT EXTRACT_FEATURES_SIMPLE_UDF(
    'filename.wav',
    '@STG_RESPIRATORY_SOUNDS/asthma/'
  );

  -- Batch processing (100 files)
  SELECT 
    FILE_NAME,
    PROCESS_FILE_UDF(FILE_NAME, '@STG/', CLASS) AS PROC_RESULT,
    EXTRACT_FEATURES_SIMPLE_UDF(FILE_NAME, '@STG/') AS FEATURES
  FROM SOURCE_METADATA
  WHERE CLASS = 'Asthma'
  LIMIT 100;
  
  -- Check metadata inserted by UDF
  SELECT * FROM M2_ISD_EQUIPE_1_DB.PROCESSED.RESPIRATORY_SOUNDS_METADATA
  LIMIT 10;
        """)
        
        session.close()
        return 0
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
