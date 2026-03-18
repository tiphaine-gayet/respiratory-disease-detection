import snowflake.connector
import os

def deploy():
    # Récupération des variables configurées sur GitHub
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
        # Préparation de l'environnement selon les variables Git
        cur.execute(f"CREATE WAREHOUSE IF NOT EXISTS {wh} WITH WAREHOUSE_SIZE='XSMALL'")
        cur.execute(f"USE WAREHOUSE {wh}")
        cur.execute(f"CREATE DATABASE IF NOT EXISTS {db}")
        cur.execute(f"USE DATABASE {db}")
        cur.execute("CREATE SCHEMA IF NOT EXISTS PUBLIC")
        cur.execute("USE SCHEMA PUBLIC")
        
        # Lecture et exécution du fichier de structure
        with open('scripts/setup_tessan.sql', 'r') as f:
            sql_commands = f.read().split(';')
            for cmd in sql_commands:
                if cmd.strip():
                    cur.execute(cmd)
        
        print(f"🚀 Infrastructure Tessan prête dans la base {db} !")
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    deploy()