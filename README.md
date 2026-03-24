# TESSAN — Détection de Maladies Respiratoires

Plateforme médicale de diagnostic des maladies respiratoires par analyse audio, utilisant des modèles de deep learning (CNN et ResNet34).

## Sommaire

- [Présentation du projet](#présentation-du-projet)
- [Architecture](#architecture)
- [Prérequis](#prérequis)
- [Installation](#installation)
- [Configuration](#configuration)
- [Lancement de l'application](#lancement-de-lapplication)
- [Commandes disponibles](#commandes-disponibles)
- [Structure du projet](#structure-du-projet)

---

## Présentation du projet

TESSAN permet :

- Aux **patients** de déposer un enregistrement audio respiratoire pour obtenir un diagnostic préliminaire (asthme, BPCO, bronchite, pneumonie, sain).
- Aux **médecins** de consulter un tableau de bord interactif avec la distribution géographique des cas en France, les métriques de santé et les prédictions filtrables.

Les modèles ML analysent les spectrogrammes Mel extraits des fichiers audio pour classer les maladies.

---

## Architecture

```
Frontend (Streamlit)  ←→  Backend (Python)  ←→  Snowflake (Data Warehouse)
                               ↕
                         Modèles ML
                    (Simple CNN / ResNet34)
```

**Pipeline de traitement audio :**
1. Rééchantillonnage à 22 050 Hz
2. Suppression des silences (seuil 30 dB)
3. Filtrage passe-bande (100–2000 Hz)
4. Normalisation à 6 secondes
5. Extraction du spectrogramme Mel

---

## Prérequis

- Python **3.10+**
- Un compte **Snowflake** avec les droits appropriés
- Un compte **Kaggle** (pour le téléchargement du dataset)
- `make` (disponible sur Linux/macOS)

---

## Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/<organisation>/respiratory-disease-detection.git
cd respiratory-disease-detection
```

### 2. Créer l'environnement virtuel et installer les dépendances

```bash
make venv
```

Cette commande crée un environnement virtuel Python et installe toutes les dépendances listées dans `requirements.txt`.

Pour activer manuellement l'environnement :

```bash
source .venv/bin/activate
```

### 3. Télécharger le dataset

Assurez-vous que vos identifiants Kaggle sont configurés (`~/.kaggle/kaggle.json`), puis :

```bash
make download_dataset
```

---

## Configuration

### Fichier `.env`

Copiez le fichier d'exemple et renseignez vos valeurs :

```bash
cp .env.example .env
```

Éditez `.env` avec vos informations Snowflake :

```env
SNOWFLAKE_ACCOUNT=<votre_compte>          # ex: xy12345.eu-west-1
SNOWFLAKE_USER=<votre_utilisateur>
SNOWFLAKE_TOKEN=<votre_token_jwt>
SNOWFLAKE_PWD=<votre_mot_de_passe>
SNOWFLAKE_ROLE=M2_ISD_EQUIPE_1_ROLE
SNOWFLAKE_WAREHOUSE=M2_ISD_E1_WH
SNOWFLAKE_DATABASE=M2_ISD_EQUIPE_1_DB
SNOWFLAKE_SCHEMA=PUBLIC
SNOWFLAKE_SCHEMA_RAW=RAW
SNOWFLAKE_SCHEMA_PROCESSED=PROCESSED
SNOWFLAKE_SCHEMA_INGESTED=INGESTED
SNOWFLAKE_SCHEMA_APP=APP
SNOWSQL_PATH=<chemin_vers_snowsql>        # ex: /usr/local/bin/snowsql
```

### Déployer l'infrastructure Snowflake

```bash
# Initialiser tous les schémas (RAW, INGESTED, APP)
make infra

# Ou étape par étape :
python -m backend.infra.schemas   # Créer les schémas
make raw                          # Transférer les fichiers audio bruts
make ingested                     # Créer la couche INGESTED avec les sons traités
make app                          # Créer la table de prédictions et les données pharmacies
```

---

## Lancement de l'application

```bash
make streamlit
```

Ou directement :

```bash
streamlit run frontend/app.py
```

L'application est accessible à l'adresse : [http://localhost:8501](http://localhost:8501)

### Utilisation

1. **Connexion / Inscription** : Créez un compte en choisissant le rôle *Patient* ou *Médecin*.
2. **Patient** : Déposez un fichier audio (WAV, MP3, FLAC), visualisez le spectrogramme et consultez les probabilités de diagnostic.
3. **Médecin** : Accédez au tableau de bord avec la carte des cas en France et les statistiques agrégées.

---

## Commandes disponibles

| Commande | Description |
|---|---|
| `make venv` | Crée l'environnement virtuel et installe les dépendances |
| `make download_dataset` | Télécharge le dataset depuis Kaggle |
| `make infra` | Déploie toute l'infrastructure Snowflake |
| `make raw` | Transfère les fichiers audio bruts vers Snowflake |
| `make ingested` | Crée le schéma INGESTED avec les sons traités |
| `make app` | Configure la table de prédictions et les pharmacies |
| `make streamlit` | Lance l'application web Streamlit |

---

## Structure du projet

```
respiratory-disease-detection/
├── backend/
│   ├── infra/          # Déploiement infrastructure Snowflake
│   ├── models/         # Modèles ML (Simple CNN, ResNet34)
│   ├── router/         # Logique métier (auth, prédictions, preprocessing)
│   └── utils/          # Client Snowflake partagé
├── frontend/
│   ├── app.py          # Point d'entrée Streamlit
│   ├── pages/          # Pages de l'interface (auth, diagnostic, dashboard)
│   ├── components/     # Composants réutilisables (audio, charts, styles)
│   └── assets/         # Fichiers audio de référence par maladie
├── config/
│   └── snowflake_config.json
├── scripts/            # Scripts de déploiement SQL
├── .env.example        # Template des variables d'environnement
├── requirements.txt    # Dépendances Python
└── Makefile            # Commandes de gestion du projet
```

---

## Stack technique

| Composant | Technologies |
|---|---|
| Frontend | Streamlit, Pydeck, Matplotlib |
| Traitement audio | Librosa, SoundFile |
| Deep Learning | PyTorch, Torchvision |
| Data Warehouse | Snowflake Snowpark |
| Sécurité | PyJWT, cryptography (PBKDF2-HMAC-SHA256) |
| Qualité du code | Ruff, Bandit, pre-commit |
