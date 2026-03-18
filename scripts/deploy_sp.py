# scripts/deploy_sp.py
from snowflake.snowpark import Session
from snowflake.snowpark.functions import sproc
import os
import json

def deploy_training_sp():
    # Connexion via Snowpark
    with open('config/snowflake_config.json', 'r') as f:
        config = json.load(f)

    session = Session.builder.configs({
        "account": os.getenv("SNOWFLAKE_ACCOUNT"),
        "user": os.getenv("SNOWFLAKE_USERNAME"),
        "password": os.getenv("SNOWFLAKE_PASSWORD"),
        "database": config["database"],
        "schema": config["schema"],
        "warehouse": config["warehouse"]
    }).create()

    # Enregistrement de la Stored Procedure
    session.sproc.register_from_file(
        file_path="scripts/train_logic.py",
        func_name="train_respiratory_model",
        name="TRAIN_MODEL_SP",
        return_type=StringType(),
        input_types=[],
        is_permanent=True,
        replace=True,
        stage_location="@STG_RESPIRATORY_SOUNDS",
        # Liste simplifiée et stable 
        packages=["snowflake-snowpark-python", "scikit-learn", "pandas", "joblib"]
    )
    print("🚀 Stored Procedure d'entraînement déployée !")

if __name__ == "__main__":
    deploy_training_sp()