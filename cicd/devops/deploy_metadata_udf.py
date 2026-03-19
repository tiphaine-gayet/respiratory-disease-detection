"""
Deploy the extract_metadata function as a Snowflake UDF (User Defined Function).
"""

from snowflake.snowpark import Session
from snowflake.snowpark.functions import udf
from snowflake.snowpark.types import StringType, StructType, StructField, IntegerType, FloatType
import os
import json
from dotenv import load_dotenv

load_dotenv()


def deploy_metadata_udf():
    """Deploy extract_metadata as a Snowflake UDF."""
    
    # Load configuration
    with open('config/snowflake_config.json', 'r') as f:
        config = json.load(f)

    # Create Snowpark session
    session = Session.builder.configs({
        "account": os.getenv("SNOWFLAKE_ACCOUNT"),
        "user": os.getenv("SNOWFLAKE_USER"),
        "password": os.getenv("SNOWFLAKE_TOKEN"),
        "database": config.get("database", "TESSAN_HACKATHON"),
        "schema": config.get("schema", "DATASETS"),
        "warehouse": config.get("warehouse", "COMPUTE_WH")
    }).create()

    # Define the UDF return type (struct matching extract_metadata output)
    return_type = StructType([
        StructField("sample_rate", IntegerType(), nullable=True),
        StructField("duration_s", FloatType(), nullable=True),
        StructField("n_samples", IntegerType(), nullable=True),
        StructField("amplitude_max", FloatType(), nullable=True),
        StructField("rms", FloatType(), nullable=True),
        StructField("error", StringType(), nullable=True),
    ])

    # Register the UDF from the Python file
    session.udf.register_from_file(
        file_path="../../backend/db/table/respiratory_sounds_metadata.py",
        func_name="extract_metadata",
        name="EXTRACT_AUDIO_METADATA",
        return_type=return_type,
        input_types=[StringType()],
        is_permanent=True,
        replace=True,
        stage_location="@STG_RESPIRATORY_SOUNDS",
        packages=["librosa", "numpy", "scikit-learn"]
    )
    
    print("✅ UDF 'EXTRACT_AUDIO_METADATA' deployed successfully!")
    print("   Usage in SQL: SELECT EXTRACT_AUDIO_METADATA(filepath) FROM table;")
    
    session.close()


if __name__ == "__main__":
    deploy_metadata_udf()
