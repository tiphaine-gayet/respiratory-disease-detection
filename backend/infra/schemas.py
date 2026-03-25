import os
from ..utils.snowflake_client import SnowflakeClient

DATABASE = os.getenv("SNOWFLAKE_DATABASE")
SCHEMAS = {
    os.getenv("SNOWFLAKE_SCHEMA_RAW"),
    os.getenv("SNOWFLAKE_SCHEMA_PROCESSED"),
    os.getenv("SNOWFLAKE_SCHEMA_INGESTED"),
    os.getenv("SNOWFLAKE_SCHEMA_APP"),
    os.getenv("SNOWFLAKE_SCHEMA_MODEL"),
}

def create_schema(client, schema):
    client.execute(f"CREATE SCHEMA IF NOT EXISTS {DATABASE}.{schema}")
    print(f"✅ Schema {DATABASE}.{schema} created (if not exists).")

if __name__ == "__main__":
    print("🚀 Setting up Snowflake schemas...")
    with SnowflakeClient() as client:
        for schema in SCHEMAS:
            create_schema(client, schema)
    print("✅ All schemas ready.")