import snowflake.connector
import os

def run_test():
    # Connexion via les variables d'environnement (GitHub Secrets)
    ctx = snowflake.connector.connect(
        user=os.getenv('SNOWFLAKE_USERNAME'),
        password=os.getenv('SNOWFLAKE_PASSWORD'),
        account=os.getenv('SNOWFLAKE_ACCOUNT'),
        warehouse=os.getenv('SNOWFLAKE_WAREHOUSE'),
        database=os.getenv('SNOWFLAKE_DATABASE'),
        schema='PUBLIC',
        role='ACCOUNTADMIN'
    )
    
    try:
        cs = ctx.cursor()
        # Création de la table de log pour le projet Tessan
        cs.execute("""
            CREATE TABLE IF NOT EXISTS HACKATHON_LOG (
                ID INT IDENTITY(1,1),
                MESSAGE STRING,
                CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
            )
        """)
        
        # Insertion du test
        cs.execute("INSERT INTO HACKATHON_LOG (MESSAGE) VALUES ('Test réussi via Python Connector !')")
        print("✅ Succès : La table a été mise à jour dans Snowflake.")
        
    finally:
        cs.close()
        ctx.close()

if __name__ == "__main__":
    run_test()