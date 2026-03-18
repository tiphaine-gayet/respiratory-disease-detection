import snowflake.connector
import os
import json

def deploy():
    # 1. Charger la config partagée
    with open('config/snowflake_config.json', 'r') as f:
        config = json.load(f)

    # 2. Connexion via Secrets (Identifiants)
    conn = snowflake.connector.connect(
        user=os.getenv('SNOWFLAKE_USERNAME'),
        password=os.getenv('SNOWFLAKE_PASSWORD'),
        account=os.getenv('SNOWFLAKE_ACCOUNT'),
        role='ACCOUNTADMIN'
    )

    try:
        cur = conn.cursor()
        db = config['database'].upper()
        wh = config['warehouse'].upper()

        # Configuration de l'environnement
        cur.execute(f"CREATE WAREHOUSE IF NOT EXISTS {wh} WITH WAREHOUSE_SIZE='XSMALL'")
        cur.execute(f"USE WAREHOUSE {wh}")
        cur.execute(f"CREATE DATABASE IF NOT EXISTS {db}")
        cur.execute(f"USE DATABASE {db}")
        cur.execute(f"CREATE SCHEMA IF NOT EXISTS {config['schema']}")
        cur.execute(f"USE SCHEMA {config['schema']}")
        
        # Lecture et exécution du SQL
        with open('scripts/setup_tessan.sql', 'r') as f:
            sql_commands = f.read().split(';')
            for cmd in sql_commands:
                if cmd.strip():
                    # On peut même faire du remplacement dynamique ici si besoin
                    cur.execute(cmd)
        
        print(f"🚀 Infrastructure déployée dans {db} via fichier de config !")
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    deploy()