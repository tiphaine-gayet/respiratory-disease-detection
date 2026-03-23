import os
import subprocess
import tempfile
import requests
from pathlib import Path
from ...utils.snowflake_client import SnowflakeClient

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
DATABASE     = os.getenv("SNOWFLAKE_DATABASE")
SCHEMA       = os.getenv("SNOWFLAKE_SCHEMA")
SNOWSQL_PATH = os.getenv("SNOWSQL_PATH", "snowsql")

STAGE_NAME       = "STG_PHARMACIES_OSM"
STAGE_FULL_PATH  = f"{DATABASE}.{SCHEMA}.{STAGE_NAME}"
OSM_TABLE_NAME   = f"{DATABASE}.{SCHEMA}.PHARMACIES_OSM_RAW"
FINAL_TABLE_NAME = f"{DATABASE}.{SCHEMA}.PHARMACIES_FRANCE"

# Extrait quotidien OSM des pharmacies — magellium / data.gouv.fr
# Mis à jour le 13 février 2026 — Format CSV, 5.9 Mo
OSM_CSV_URL    = "https://www.data.gouv.fr/api/1/datasets/r/2bb8cbe9-0291-4705-9cbc-092e6513dcc3"
LOCAL_FILENAME = "pharmacies_osm.csv"

# ─────────────────────────────────────────────
# Étape 1 — Téléchargement local
# ─────────────────────────────────────────────
def download_file(dest_path: Path) -> Path:
    print(f"⬇️  Téléchargement du fichier OSM pharmacies...")
    response = requests.get(OSM_CSV_URL, stream=True, timeout=120)
    response.raise_for_status()
    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    size_mb = dest_path.stat().st_size / 1024 / 1024
    print(f"✅ Fichier téléchargé : {dest_path} ({size_mb:.1f} MB)")
    return dest_path


# ─────────────────────────────────────────────
# Étape 2 — Stage Snowflake
# ─────────────────────────────────────────────
def create_stage(client):
    print(f"🏗️  Création du stage {STAGE_FULL_PATH}...")
    client.execute(f"CREATE OR REPLACE STAGE {STAGE_FULL_PATH}")
    print(f"✅ Stage prêt.")


# ─────────────────────────────────────────────
# Étape 3 — Upload via snowsql PUT
# ─────────────────────────────────────────────
def upload_to_stage(local_file: Path):
    print(f"📤 Upload de {local_file.name} vers @{STAGE_FULL_PATH}...")
    cmd = [
        SNOWSQL_PATH,
        "-q", f"PUT file://{local_file} @{STAGE_FULL_PATH} AUTO_COMPRESS=TRUE OVERWRITE=TRUE"
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✅ Upload terminé.")
        if result.stdout:
            print(f"   {result.stdout.strip()}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Échec de l'upload : {e.stderr}")
        raise


# ─────────────────────────────────────────────
# Étape 4 — Table RAW avec colonnes nommées
# ─────────────────────────────────────────────
def create_raw_table(client):
    print(f"🏗️  Création de la table brute {OSM_TABLE_NAME}...")
    client.execute(f"""
        CREATE OR REPLACE TABLE {OSM_TABLE_NAME} (
            fid               VARCHAR,   -- Identifiant OSM composite  ex: pharmacies_point.4534591198
            osm_id            VARCHAR,   -- ID OpenStreetMap
            amenity           VARCHAR,   -- Toujours 'pharmacy' ici
            name              VARCHAR,   -- Nom de la pharmacie
            short_name        VARCHAR,
            official_name     VARCHAR,
            alt_name          VARCHAR,
            old_name          VARCHAR,
            operator          VARCHAR,   -- Gestionnaire / enseigne
            operator_type     VARCHAR,   -- 'operator:type' dans OSM
            dispensing        VARCHAR,   -- 'yes' si dispensation médicaments
            emergency         VARCHAR,
            capacity          VARCHAR,
            wheelchair        VARCHAR,   -- Accessibilité PMR
            social_facility   VARCHAR,
            ref_fr_finess     VARCHAR,   -- 'ref:FR:FINESS' — numéro FINESS ET
            type_fr_finess    VARCHAR,   -- 'type:FR:FINESS'
            ref_fr_naf        VARCHAR,   -- 'ref:FR:NAF'  — code APE
            ref_fr_siret      VARCHAR,   -- 'ref:FR:SIRET' — numéro SIRET (14 chiffres)
            website           VARCHAR,
            contact_website   VARCHAR,
            url               VARCHAR,
            phone             VARCHAR,   -- Téléphone principal
            contact_phone     VARCHAR,
            fax               VARCHAR,
            contact_fax       VARCHAR,
            email             VARCHAR,
            contact_email     VARCHAR,
            addr_housename    VARCHAR,
            addr_housenumber  VARCHAR,   -- Numéro de rue
            addr_street       VARCHAR,   -- Nom de la rue
            addr_city         VARCHAR,   -- Commune
            addr_postcode     VARCHAR,   -- Code postal
            wikidata          VARCHAR,
            wikipedia         VARCHAR,
            description       VARCHAR,
            opening_hours     VARCHAR,   -- Horaires d'ouverture (format OSM)
            source            VARCHAR,
            note              VARCHAR,
            osm_version       VARCHAR,
            osm_timestamp     TIMESTAMP_NTZ,
            the_geom          VARCHAR,   -- Géométrie projetée EPSG:3857 (POINT WKT)
            osm_original_geom VARCHAR,   -- Géométrie originale (polygone si way)
            osm_type          VARCHAR    -- 'node', 'way' ou 'relation'
        )
    """)
    print(f"✅ Table {OSM_TABLE_NAME} créée.")


def copy_into_raw(client):
    print(f"📥 COPY INTO {OSM_TABLE_NAME}...")
    client.execute(f"""
        COPY INTO {OSM_TABLE_NAME}
        FROM @{STAGE_FULL_PATH}
        FILE_FORMAT = (
            TYPE                           = 'CSV'
            FIELD_DELIMITER                = ','
            RECORD_DELIMITER               = '\\n'
            SKIP_HEADER                    = 1
            FIELD_OPTIONALLY_ENCLOSED_BY   = '"'
            NULL_IF                        = ('', 'NULL')
            EMPTY_FIELD_AS_NULL            = TRUE
            ERROR_ON_COLUMN_COUNT_MISMATCH = FALSE
        )
        ON_ERROR = 'CONTINUE'
    """)
    nb = client.query(f"SELECT COUNT(*) FROM {OSM_TABLE_NAME}")
    print(f"✅ {nb} lignes chargées dans {OSM_TABLE_NAME}.")


# ─────────────────────────────────────────────
# Étape 5 — Table finale PHARMACIES_FRANCE
#
# Extraction de lat/long depuis the_geom :
#   the_geom est en EPSG:3857 (mètres), format : POINT (x y)
#   → on extrait x et y puis on convertit en WGS84 (degrés)
#   Formule : lon = x / 20037508.34 * 180
#             lat = atan(exp(y / 20037508.34 * pi)) * 360 / pi - 90
# ─────────────────────────────────────────────
def create_pharmacies_table(client):
    print(f"🏗️  Création de la table finale {FINAL_TABLE_NAME}...")
    client.execute(f"""
        CREATE OR REPLACE TABLE {FINAL_TABLE_NAME} AS

        WITH parsed_geom AS (
            SELECT
                *,
                -- Extraction des coordonnées X/Y depuis 'POINT (x y)'
                TRY_TO_DOUBLE(
                    SPLIT_PART(REGEXP_SUBSTR(the_geom, '\\\\((.+)\\\\)', 1, 1, 'e', 1), ' ', 1)
                ) AS geom_x,
                TRY_TO_DOUBLE(
                    SPLIT_PART(REGEXP_SUBSTR(the_geom, '\\\\((.+)\\\\)', 1, 1, 'e', 1), ' ', 2)
                ) AS geom_y
            FROM {OSM_TABLE_NAME}
        )

        SELECT
            -- Identifiants
            COALESCE(ref_fr_siret, '')      AS siret,
            ref_fr_finess                   AS nofinesset,
            osm_id,

            -- Nom
            name                            AS nom,
            operator                        AS enseigne,

            -- Adresse
            TRIM(
                COALESCE(addr_housenumber || ' ', '') ||
                COALESCE(addr_street, '')
            )                               AS adresse,
            addr_city                       AS commune,
            addr_postcode                   AS code_postal,
            LEFT(addr_postcode, 2)          AS code_departement,

            -- Contact
            COALESCE(phone, contact_phone)  AS telephone,
            COALESCE(fax,   contact_fax)    AS telecopie,
            COALESCE(email, contact_email)  AS email,
            COALESCE(website, contact_website, url) AS site_web,

            -- Horaires
            opening_hours,

            -- Accessibilité
            wheelchair,

            -- Géolocalisation WGS84
            ROUND(geom_x / 20037508.34 * 180, 6)                      AS loc_long,
            ROUND(
                DEGREES(ATAN(EXP(geom_y / 20037508.34 * PI()))) * 2 - 90
            , 6)                                                        AS loc_lat,

            -- Métadonnées OSM
            osm_type,
            osm_timestamp,
            source                          AS source_donnee,
            note

        FROM parsed_geom
        ORDER BY code_departement, code_postal, nom
    """)
    nb = client.query(f"SELECT COUNT(*) FROM {FINAL_TABLE_NAME}")
    print(f"✅ Table {FINAL_TABLE_NAME} créée avec {nb} pharmacies.")


# ─────────────────────────────────────────────
# Étape 6 — Contrôle qualité
# ─────────────────────────────────────────────
def quality_check(client):
    print("\n📊 Contrôle qualité :")

    rows = client.query(f"""
        SELECT
            COUNT(*)                                              AS total,
            COUNT(loc_lat)                                        AS avec_coordonnees,
            ROUND(COUNT(loc_lat) * 100.0 / COUNT(*), 1)          AS pct_geolocalisees,
            COUNT(CASE WHEN siret  = '' THEN 1 END)               AS sans_siret,
            COUNT(CASE WHEN nofinesset IS NOT NULL THEN 1 END)    AS avec_finess,
            COUNT(CASE WHEN opening_hours IS NOT NULL THEN 1 END) AS avec_horaires,
            COUNT(CASE WHEN telephone IS NOT NULL THEN 1 END)     AS avec_telephone
        FROM {FINAL_TABLE_NAME}
    """)
    row = rows[0]
    print(f"   Total pharmacies       : {row[0]}")
    print(f"   Géolocalisées          : {row[1]} ({row[2]}%)")
    print(f"   Sans SIRET             : {row[3]}")
    print(f"   Avec FINESS            : {row[4]}")
    print(f"   Avec horaires          : {row[5]}")
    print(f"   Avec téléphone         : {row[6]}")

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("🚀 Chargement des pharmacies OSM dans Snowflake\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        local_file = Path(tmpdir) / LOCAL_FILENAME
        download_file(local_file)

        with SnowflakeClient() as client:
            create_stage(client)
            upload_to_stage(local_file)
            create_raw_table(client)
            copy_into_raw(client)
            create_pharmacies_table(client)
            quality_check(client)

    print("\n✅ Pipeline terminé — table PHARMACIES_FRANCE disponible dans Snowflake !")
