import snowflake.connector
import os

def run_test():
    # Récupération des variables
    db = os.getenv('SNOWFLAKE_DATABASE')
    schema = 'PUBLIC'
    
    ctx = snowflake.connector.connect(
        user=os.getenv('SNOWFLAKE_USERNAME'),
        password=os.getenv('SNOWFLAKE_PASSWORD'),
        account=os.getenv('SNOWFLAKE_ACCOUNT'),
        warehouse=os.getenv('SNOWFLAKE_WAREHOUSE'),
        role='ACCOUNTADMIN'
    )
    
    try:
        cs = ctx.cursor()
        
        # On force l'utilisation de la DB et du Schéma 
        cs.execute(f"USE DATABASE {db}")
        cs.execute(f"USE SCHEMA {schema}")
        
        # Création de la table avec le nom complet
        cs.execute(f"""
            CREATE TABLE IF NOT EXISTS {db}.{schema}.HACKATHON_LOG (
                ID INT IDENTITY(1,1),
                MESSAGE STRING,
                CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
            )
        """)
        
        cs.execute(f"INSERT INTO {db}.{schema}.HACKATHON_LOG (MESSAGE) VALUES ('Test réussi avec DATABASE explicite !')")
        print(f"✅ Succès : Table mise à jour dans {db}.{schema}")
        
    finally:
        cs.close()
        ctx.close()

if __name__ == "__main__":
    run_test()