-- Création d'une table de test pour le Hackathon Tessan
CREATE TABLE IF NOT EXISTS HACKATHON_LOG (
    ID INT IDENTITY(1,1),
    MESSAGE STRING,
    CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP()
);

INSERT INTO HACKATHON_LOG (MESSAGE) 
VALUES ('Connexion réussie depuis GitHub Actions !');