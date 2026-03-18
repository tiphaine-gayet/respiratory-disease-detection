# 🫁 Asthma Detection Dataset - Snowflake Integration

Guide complet pour charger le **Asthma Detection Dataset: Version 2** dans Snowflake.

## 📋 Contenu du Dataset

Le dataset comprend **1,211 fichiers audio WAV** répartis en 5 catégories :

| Catégorie | Nombre de fichiers |
|-----------|-------------------|
| ASTHMA | 288 |
| COPD | 401 |
| PNEUMONIA | 285 |
| BRONCHIAL | 104 |
| HEALTHY | 133 |

**Durée des fichiers** : 1.5-5 secondes
**Format** : WAV (audio binaire)

---

## 🚀 Installation

### 1. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 2. Configurer les credentials Snowflake

Remplir le fichier `.env` à la racine du projet :

```bash
cp .env.example .env
```

Éditer `.env` avec vos identifiants Snowflake :

```ini
SNOWFLAKE_ACCOUNT=xyz12345        # votre account ID
SNOWFLAKE_USER=votre_utilisateur
SNOWFLAKE_TOKEN=votre_mot_de_passe
SNOWFLAKE_ROLE=ACCOUNTADMIN
SNOWFLAKE_WAREHOUSE=COMPUTE_WH
SNOWFLAKE_DATABASE=TESSAN_HACKATON
SNOWFLAKE_SCHEMA=ASTHMA_DETECTION
```

---

## 📦 Étapes de chargement

### Étape 1 : Initialiser le schéma Snowflake

Connectez-vous à Snowflake et exécutez le script d'initialisation :

```sql
-- Copier le contenu de snowflake/01_init_schema.sql
-- et l'exécuter dans Snowflake Web UI ou via snowsql
```

Ou via CLI :

```bash
cd snowflake/
snowsql -c votre_connection -f 01_init_schema.sql
```

**Cet script crée :**
- ✅ Database : `TESSAN_HACKATON`
- ✅ Schema : `ASTHMA_DETECTION`
- ✅ Table `ASTHMA_AUDIO_FILES` (métadonnées)
- ✅ Table `DATASET_STATISTICS` (statistiques)
- ✅ Stage interne `ASTHMA_AUDIO_STAGE` (stockage des fichiers)
- ✅ 2 vues : `VW_DATASET_OVERVIEW` et `VW_AUDIO_FILES`

### Étape 2 : Charger les données

```bash
python snowflake/02_load_dataset.py
```

**Le script va :**
1. 🗂️ Scanner le dataset local (1,211 fichiers)
2. 📤 Uploader les fichiers audio vers la stage Snowflake
3. 📊 Insérer les métadonnées dans la table
4. 📈 Mettre à jour les statistiques
5. ✅ Afficher un résumé du chargement

**Temps estimé :** ~5-10 minutes (dépend du réseau/warehouse)

---

## 🔍 Utiliser le dataset

### Option A : Via Snowflake Web UI

1. Aller sur [Snowflake Web UI](https://app.snowflake.com)
2. Sélectionner la database `TESSAN_HACKATON`
3. Utiliser les vues et tables :

```sql
SELECT * FROM VW_DATASET_OVERVIEW;
SELECT * FROM VW_AUDIO_FILES;
```

### Option B : Via Python/SQL

```python
from utils import get_connection, fetchall_as_dicts

with get_connection() as conn:
    rows = fetchall_as_dicts(conn, "SELECT * FROM VW_DATASET_OVERVIEW")
    for row in rows:
        print(row)
```

### Option C : Exécuter les requêtes d'exemple

```bash
# Copier le contenu de snowflake/03_usage_examples.sql
# et l'exécuter dans Snowflake
```

---

## 📊 Requêtes utiles

### Vue d'ensemble du dataset

```sql
SELECT * FROM VW_DATASET_OVERVIEW;
```

**Résultat attendu :**
| DIAGNOSIS | FILE_COUNT | TOTAL_SIZE_GB | AVG_FILE_SIZE_MB | LAST_UPDATED |
|-----------|-----------|---------------|------------------|--------------|
| COPD | 401 | ~X.XX | ~Y.YY | 2026-03-18 |
| ASTHMA | 288 | ~X.XX | ~Y.YY | 2026-03-18 |
| ... | ... | ... | ... | ... |

### Accéder aux fichiers dans la stage

```sql
-- Lister tous les fichiers
LIST @ASTHMA_AUDIO_STAGE;

-- Lister les fichiers d'une catégorie
LIST @ASTHMA_AUDIO_STAGE/asthma;

-- Télécharger un fichier localement
GET @ASTHMA_AUDIO_STAGE/asthma/P1Asthma13S.wav file:///tmp/;
```

### Filtrer par catégorie

```sql
SELECT FILE_NAME, FILE_SIZE_BYTES, STAGE_PATH
FROM ASTHMA_AUDIO_FILES
WHERE DIAGNOSIS = 'ASTHMA'
LIMIT 10;
```

### Analyser les tailles de fichiers

```sql
SELECT
    DIAGNOSIS,
    COUNT(*) as FILE_COUNT,
    ROUND(AVG(FILE_SIZE_BYTES) / 1024, 2) as AVG_SIZE_KB,
    ROUND(MAX(FILE_SIZE_BYTES) / 1024, 2) as MAX_SIZE_KB
FROM ASTHMA_AUDIO_FILES
GROUP BY DIAGNOSIS
ORDER BY FILE_COUNT DESC;
```

---

## 🏗️ Architecture

```
Snowflake Database Structure
├── Database: TESSAN_HACKATON
│   └── Schema: ASTHMA_DETECTION
│       ├── Tables:
│       │   ├── ASTHMA_AUDIO_FILES (1,211 rows)
│       │   └── DATASET_STATISTICS (5 rows)
│       ├── Stage:
│       │   └── @ASTHMA_AUDIO_STAGE (fichiers audio binaires)
│       └── Views:
│           ├── VW_DATASET_OVERVIEW
│           └── VW_AUDIO_FILES
```

### Table ASTHMA_AUDIO_FILES

| Colonne | Type | Description |
|---------|------|-------------|
| FILE_ID | VARCHAR | ID unique (UUID) |
| FILE_NAME | VARCHAR | Nom du fichier (ex: P1Asthma13S.wav) |
| DIAGNOSIS | VARCHAR | Catégorie (ASTHMA, COPD, etc.) |
| FILE_PATH | VARCHAR | Chemin local du fichier |
| FILE_SIZE_BYTES | NUMBER | Taille en octets |
| UPLOAD_TIMESTAMP | TIMESTAMP | Date d'upload |
| STAGE_PATH | VARCHAR | Chemin dans la stage Snowflake |

---

## ⚠️ Troubleshooting

### Erreur de connexion Snowflake

```
snowflake.connector.errors.ProgrammingError: 250001: Authentication failed
```

**Solution :** Vérifier les credentials dans `.env` :
- Account ID correct ? (ex: `xy12345.us-east-1`)
- Utilisateur et password corrects ?
- Role a les permissions ?

### Fichiers non trouvés

```
FileNotFoundError: Dataset path not found
```

**Solution :** Vérifier que le dataset est bien extrait :
```bash
ls -la "Asthma Detection Dataset Version 2/"
```

### Dépassement de quota stage

```
SQL compilation error: Insufficient quota
```

**Solution :** Augmenter le quota de la stage ou supprimer les anciens fichiers.

### Timeout pendant l'upload

```
socket timeout | Connection closed
```

**Solution :**
- Augmenter le warehouse (ex: M au lieu de XSMALL)
- Redémarrer le warehouse
- Réessayer avec moins de fichiers

---

## 🧹 Maintenance

### Nettoyer les données

```sql
-- Supprimer tous les enregistrements de métadonnées
DELETE FROM ASTHMA_AUDIO_FILES;

-- Vider la stage (ATTENTION: destructif)
REMOVE @ASTHMA_AUDIO_STAGE;
```

### Vérifier l'intégrité des données

```sql
-- Compter le nombre total de fichiers
SELECT COUNT(*) as TOTAL_FILES FROM ASTHMA_AUDIO_FILES;

-- Chercher les doublons
SELECT FILE_NAME, COUNT(*)
FROM ASTHMA_AUDIO_FILES
GROUP BY FILE_NAME
HAVING COUNT(*) > 1;

-- Vérifier les fichiers manquants (taille = 0)
SELECT COUNT(*) FROM ASTHMA_AUDIO_FILES WHERE FILE_SIZE_BYTES = 0;
```

---

## 📚 Ressources

- **Dataset source :** Kaggle - Asthma Detection Dataset Version 2
- **Documentation Snowflake :** https://docs.snowflake.com
- **Python Snowflake Connector :** https://github.com/snowflakedb/snowflake-connector-python

---

## ✅ Checklist de configuration

- [ ] `.env` rempli avec les credentials Snowflake
- [ ] `requirements.txt` installé (`pip install -r requirements.txt`)
- [ ] Script `01_init_schema.sql` exécuté dans Snowflake
- [ ] Dataset extrait dans `Asthma Detection Dataset Version 2/`
- [ ] Script `02_load_dataset.py` exécuté avec succès
- [ ] Tables et fichiers visibles dans Snowflake

---

**Questions ou problèmes ?** Consultez la documentation Snowflake ou les logs du script de chargement.
