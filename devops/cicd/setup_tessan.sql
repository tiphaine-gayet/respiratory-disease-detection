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

CREATE SCHEMA IF NOT EXISTS RAW;
USE SCHEMA {db}.RAW;

-- 4. Création du Stage Interne pour les fichiers audio .wav [cite: 64, 65]
-- On active le DIRECTORY pour que Streamlit puisse lister les sons plus tard [cite: 112, 114]
-- Création du Stage pour les sons respiratoires [cite: 31, 51]
CREATE STAGE IF NOT EXISTS STG_RESPIRATORY_SOUNDS
DIRECTORY = (ENABLE = TRUE);

-- Table pour les métadonnées du dataset (Asthma, COPD, Bronchial, Pneumonia, Healthy) [cite: 35, 38]
CREATE TABLE IF NOT EXISTS RAW_RESPIRATORY_METADATA (
    FILE_NAME STRING,
    DIAGNOSIS STRING,
    DURATION_SECONDS FLOAT,
    SAMPLE_RATE INT,
    UPLOADED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

CREATE SCHEMA IF NOT EXISTS PROCESSED;
USE SCHEMA {db}.PROCESSED;

CREATE SCHEMA IF NOT EXISTS TEST;
USE SCHEMA {db}.TEST;