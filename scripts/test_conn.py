import snowflake.connector
import os

def deploy_sql():
    ctx = snowflake.connector.connect(
        user=os.getenv('SNOWFLAKE_USERNAME'),
        password=os.getenv('SNOWFLAKE_PASSWORD'),
        account=os.getenv('SNOWFLAKE_ACCOUNT'),
        database=os.getenv('SNOWFLAKE_DATABASE'),
        warehouse=os.getenv('SNOWFLAKE_WAREHOUSE'),
        role='ACCOUNTADMIN'
    )
    
    try:
        cs = ctx.cursor()
        # Lecture du fichier SQL
        with open('scripts/setup_tessan.sql', 'r') as f:
            sql_commands = f.read().split(';')
            
        for command in sql_commands:
            if command.strip():
                cs.execute(command)
                print(f"✅ Exécuté : {command[:50]}...")
                
        print("\n🚀 Infrastructure Tessan prête sur Snowflake !")
        
    finally:
        cs.close()
        ctx.close()

if __name__ == "__main__":
    deploy_sql()