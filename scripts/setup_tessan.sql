-- 1. Création et activation du Warehouse (Puissance de calcul)
-- On utilise une taille XSMALL pour économiser les crédits du hackathon
CREATE WAREHOUSE IF NOT EXISTS {warehouse} 
WITH WAREHOUSE_SIZE = 'XSMALL' 
AUTO_SUSPEND = 60 
AUTO_RESUME = TRUE;

USE WAREHOUSE {warehouse};

-- 2. Création et activation de la Database (Stockage)
CREATE DATABASE IF NOT EXISTS {db};
USE DATABASE {db};

-- 3. Création du Schéma par défaut
CREATE SCHEMA IF NOT EXISTS PUBLIC;
USE SCHEMA {db}.PUBLIC;

-- 4. Création du Stage Interne pour les fichiers audio .wav [cite: 64, 65]
-- On active le DIRECTORY pour que Streamlit puisse lister les sons plus tard [cite: 112, 114]
CREATE STAGE IF NOT EXISTS STG_RESPIRATORY_SOUNDS
DIRECTORY = (ENABLE = TRUE)
ENCRYPTION = (TYPE = 'SNOWFLAKE_SSE');

-- 5. Table pour les métadonnées du dataset Kaggle [cite: 38, 63]
CREATE TABLE IF NOT EXISTS RAW_RESPIRATORY_METADATA (
    FILE_NAME STRING,
    DIAGNOSIS STRING,        -- Asthme, BPCO, Bronchique, Pneumonie, Sain [cite: 36, 38]
    DURATION_SECONDS FLOAT,  -- Utile pour filtrer les sons trop courts [cite: 70, 71]
    SAMPLE_RATE INT,         -- Pour vérifier l'hétérogénéité du dataset [cite: 38, 70]
    UPLOADED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);