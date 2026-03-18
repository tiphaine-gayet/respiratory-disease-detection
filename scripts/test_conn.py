import snowflake.connector
import os

def deploy():
    # Récupération des variables Git
    db = os.getenv('SNOWFLAKE_DATABASE').upper()
    wh = os.getenv('SNOWFLAKE_WAREHOUSE').upper()

    conn = snowflake.connector.connect(
        user=os.getenv('SNOWFLAKE_USERNAME'),
        password=os.getenv('SNOWFLAKE_PASSWORD'),
        account=os.getenv('SNOWFLAKE_ACCOUNT'),
        role='ACCOUNTADMIN'
    )

    try:
        cur = conn.cursor()
        # On définit le contexte d'abord
        cur.execute(f"CREATE WAREHOUSE IF NOT EXISTS {wh} WITH WAREHOUSE_SIZE='XSMALL'")
        cur.execute(f"USE WAREHOUSE {wh}")
        cur.execute(f"CREATE DATABASE IF NOT EXISTS {db}")
        cur.execute(f"USE DATABASE {db}")
        
        # On lit et exécute le reste du fichier SQL (Tables, Stages)
        with open('scripts/setup_tessan.sql', 'r') as f:
            full_sql = f.read()
            # On sépare par point-virgule pour exécuter commande par commande
            for cmd in full_sql.split(';'):
                if cmd.strip():
                    cur.execute(cmd)
        print("🚀 Infrastructure déployée avec succès !")
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    deploy()