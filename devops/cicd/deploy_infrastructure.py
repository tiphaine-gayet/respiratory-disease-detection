import snowflake.connector
import os
from dotenv import load_dotenv

load_dotenv()

def deploy():


    db = os.getenv('SNOWFLAKE_DATABASE')
    wh = os.getenv('SNOWFLAKE_WAREHOUSE')

    # 2. Connexion
    conn = snowflake.connector.connect(
        user=os.getenv('SNOWFLAKE_USERNAME'),
        password=os.getenv('SNOWFLAKE_PASSWORD'),
        account=os.getenv('SNOWFLAKE_ACCOUNT'),
        role=os.getenv('SNOWFLAKE_ROLE')
    )

    cur = None
    try:
        cur = conn.cursor()
        
        # 3. Setup initial (Python injecte les variables ici)
        cur.execute(f"CREATE WAREHOUSE IF NOT EXISTS {wh} WITH WAREHOUSE_SIZE='XSMALL'")
        cur.execute(f"USE WAREHOUSE {wh}")
        cur.execute(f"CREATE DATABASE IF NOT EXISTS {db}")
        cur.execute(f"USE DATABASE {db}")
        cur.execute(f"CREATE SCHEMA IF NOT EXISTS {os.getenv('SNOWFLAKE_RAW_SCHEMA')}")
        cur.execute(f"USE SCHEMA {os.getenv('SNOWFLAKE_RAW_SCHEMA')}")
        
        # 4. Lecture du fichier SQL et remplacement des variables
        with open('devops/cicd/setup_tessan.sql', 'r') as f:
            content = f.read()
            # On remplace les placeholders par les vraies valeurs
            content = content.replace('{db}', db).replace('{warehouse}', wh)
            
            sql_commands = content.split(';')
            for cmd in sql_commands:
                if cmd.strip():
                    cur.execute(cmd)
        
        print(f"🚀 Infrastructure Tessan déployée avec succès !")
    finally:
        if cur is not None:
            cur.close()
        conn.close()

if __name__ == "__main__":
    deploy()