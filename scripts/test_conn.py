import snowflake.connector
import os

def deploy_infrastructure():
    # Récupération des variables injectées par le YAML
    db = os.getenv('SNOWFLAKE_DATABASE').upper()
    warehouse = os.getenv('SNOWFLAKE_WAREHOUSE').upper()
    
    ctx = snowflake.connector.connect(
        user=os.getenv('SNOWFLAKE_USERNAME'),
        password=os.getenv('SNOWFLAKE_PASSWORD'),
        account=os.getenv('SNOWFLAKE_ACCOUNT'),
        role='ACCOUNTADMIN' 
    )
    
    try:
        cs = ctx.cursor()
        
        # Liste des commandes utilisant les f-strings pour injecter tes variables Git
        commands = [
            f"CREATE WAREHOUSE IF NOT EXISTS {warehouse} WITH WAREHOUSE_SIZE='XSMALL'",
            f"USE WAREHOUSE {warehouse}",
            f"CREATE DATABASE IF NOT EXISTS {db}",
            f"USE DATABASE {db}",
            "CREATE SCHEMA IF NOT EXISTS PUBLIC",
            "CREATE STAGE IF NOT EXISTS STG_RESPIRATORY_SOUNDS DIRECTORY = (ENABLE = TRUE)",
            """CREATE TABLE IF NOT EXISTS RAW_RESPIRATORY_METADATA (
                FILE_NAME STRING, DIAGNOSIS STRING, DURATION_SECONDS FLOAT, 
                SAMPLE_RATE INT, UPLOADED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
            )"""
        ]
        
        for cmd in commands:
            cs.execute(cmd)
            print(f"✅ Exécuté avec succès")
            
    finally:
        cs.close()
        ctx.close()

if __name__ == "__main__":
    deploy_infrastructure()