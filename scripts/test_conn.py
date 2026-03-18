import snowflake.connector
import os

def run_test():
    # Récupération des variables
    # On force en majuscules au cas où
    db = os.getenv('SNOWFLAKE_DATABASE', 'TESSAN_HACKATHON').upper()
    warehouse = os.getenv('SNOWFLAKE_WAREHOUSE', 'COMPUTE_WH').upper()
    
    ctx = snowflake.connector.connect(
        user=os.getenv('SNOWFLAKE_USERNAME'),
        password=os.getenv('SNOWFLAKE_PASSWORD'),
        account=os.getenv('SNOWFLAKE_ACCOUNT'),
        role='ACCOUNTADMIN' # On reste en ACCOUNTADMIN pour le setup
    )
    
    try:
        cs = ctx.cursor()
        
        # 1. Vérification/Création du Warehouse
        cs.execute(f"CREATE WAREHOUSE IF NOT EXISTS {warehouse}")
        cs.execute(f"USE WAREHOUSE {warehouse}")
        
        # 2. Vérification/Création de la Database
        cs.execute(f"CREATE DATABASE IF NOT EXISTS {db}")
        cs.execute(f"USE DATABASE {db}")
        
        # 3. Création du schéma et de la table
        cs.execute("CREATE SCHEMA IF NOT EXISTS PUBLIC")
        cs.execute("USE SCHEMA PUBLIC")
        
        cs.execute("""
            CREATE TABLE IF NOT EXISTS HACKATHON_LOG (
                ID INT IDENTITY(1,1),
                MESSAGE STRING,
                CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
            )
        """)
        
        cs.execute("INSERT INTO HACKATHON_LOG (MESSAGE) VALUES ('Connexion et création auto réussies !')")
        print(f"✅ Succès total dans la base {db} !")
        
    except Exception as e:
        print(f"❌ Erreur détectée : {e}")
        raise e
    finally:
        cs.close()
        ctx.close()

if __name__ == "__main__":
    run_test()