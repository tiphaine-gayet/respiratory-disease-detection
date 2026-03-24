"""
backend/router/predictions.py
──────────────────────────────────────────────────────────────────────────────
Snowflake query layer for the doctor dashboard.

Returns a flat DataFrame joining APP.PREDICTIONS with APP.PHARMACIES_FRANCE
so that every row carries the coordinates needed for the pydeck map.

Usage
-----
from backend.router.predictions import load_predictions

df = load_predictions(date_from, date_to)
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from datetime import date
from datetime import datetime
import re

import numpy as np
from datetime import date

import pandas as pd

from backend.utils.snowflake_client import SnowflakeClient

# ── Snowflake identifiers (mirror the pattern used in infra/) ─────────────────
_DB     = os.getenv("SNOWFLAKE_DATABASE")
_SCHEMA = os.getenv("SNOWFLAKE_SCHEMA_APP")
_INGESTED_SCHEMA = os.getenv("SNOWFLAKE_SCHEMA_INGESTED") or "INGESTED"

_PREDICTIONS_TABLE   = f"{_DB}.{_SCHEMA}.PREDICTIONS"
_PHARMACIES_TABLE    = f"{_DB}.{_SCHEMA}.PHARMACIES_FRANCE"
_INGESTED_STAGE      = f"{_DB}.{_INGESTED_SCHEMA}.STG_INGESTED_SOUNDS"
_INGESTED_METADATA_TABLE = f"{_DB}.{_INGESTED_SCHEMA}.INGESTED_SOUNDS_METADATA"

_PREDICTIONS_TABLE   = f"{_DB}.{_SCHEMA}.PREDICTIONS"
_PHARMACIES_TABLE    = f"{_DB}.{_SCHEMA}.PHARMACIES_FRANCE"


# ── Public API ────────────────────────────────────────────────────────────────

def load_predictions(date_from: date, date_to: date) -> pd.DataFrame:
    """
    Return predictions for [date_from, date_to] enriched with pharmacy
    coordinates and metadata from PHARMACIES_FRANCE.

    Columns returned
    ----------------
    prediction_id, predicted_at, patient_id, pharmacie_id,
    audio_file_name, model_version, action,
    pct_asthma, pct_copd, pct_bronchial, pct_pneumonia, pct_healthy,
    pct_confiance,
    pharmacie_nom, commune, code_postal, code_departement,
    loc_lat, loc_long
    """
    sql = f"""
        SELECT
            -- Prediction core
            p.prediction_id,
            p.predicted_at,
            p.patient_id,
            p.pharmacie_id,
            p.audio_file_name,
            p.model_version,
            p.action,

            -- Per-class probabilities
            p.pct_asthma,
            p.pct_copd,
            p.pct_bronchial,
            p.pct_pneumonia,
            p.pct_healthy,
            p.pct_confiance,

            -- Pharmacy metadata (LEFT JOIN — predictions without a known
            -- pharmacy still appear but without geographic columns)
            ph.nom        AS pharmacie_nom,
            ph.commune,
            ph.code_postal,
            ph.code_departement,
            ph.loc_lat,
            ph.loc_long

        FROM {_PREDICTIONS_TABLE} p
        LEFT JOIN {_PHARMACIES_TABLE} ph
            ON ph.osm_id = p.pharmacie_id

        WHERE p.predicted_at::DATE BETWEEN '{date_from}' AND '{date_to}'
        ORDER BY p.predicted_at DESC
    """

    with SnowflakeClient() as client:
        rows = client.query(sql)

    if not rows:
        return pd.DataFrame(columns=[
            "prediction_id", "predicted_at", "patient_id", "pharmacie_id",
            "audio_file_name", "model_version", "action",
            "pct_asthma", "pct_copd", "pct_bronchial", "pct_pneumonia",
            "pct_healthy", "pct_confiance",
            "pharmacie_nom", "commune", "code_postal", "code_departement",
            "loc_lat", "loc_long",
        ])

    return pd.DataFrame(rows, columns=[
        "prediction_id", "predicted_at", "patient_id", "pharmacie_id",
        "audio_file_name", "model_version", "action",
        "pct_asthma", "pct_copd", "pct_bronchial", "pct_pneumonia",
        "pct_healthy", "pct_confiance",
        "pharmacie_nom", "commune", "code_postal", "code_departement",
        "loc_lat", "loc_long",
    ])


def load_pharmacies_for_select() -> pd.DataFrame:
    """Return pharmacy options for UI selectbox: id + display label."""
    sql = f"""
        SELECT
            osm_id AS pharmacie_id,
            nom AS pharmacie_nom,
            commune,
            code_postal
        FROM {_PHARMACIES_TABLE}
        WHERE osm_id IS NOT NULL
          AND nom IS NOT NULL
        ORDER BY nom ASC, commune ASC
    """

    with SnowflakeClient() as client:
        rows = client.query(sql)

    if not rows:
        return pd.DataFrame(columns=["pharmacie_id", "label"])

    df = pd.DataFrame(rows, columns=["pharmacie_id", "pharmacie_nom", "commune", "code_postal"])
    df["pharmacie_id"] = df["pharmacie_id"].astype(str)

    def _label(row: pd.Series) -> str:
        commune = row["commune"] if pd.notna(row["commune"]) else "Commune inconnue"
        cp = str(row["code_postal"]) if pd.notna(row["code_postal"]) else "--"
        return f"{row['pharmacie_nom']} ({cp}, {commune})"

    df["label"] = df.apply(_label, axis=1)
    return df[["pharmacie_id", "label"]].drop_duplicates(subset=["pharmacie_id"]).reset_index(drop=True)


def upload_patient_audio_to_stage(audio_bytes: bytes, patient_id: str, original_filename: str | None = None) -> str:
    """
    Upload a patient audio payload to INGESTED.STG_INGESTED_SOUNDS.

    The uploaded file name is normalized to: patient_id_timestamp.ext
    (timestamp in UTC, format: YYYYMMDDHHMMSS).
    """
    if not audio_bytes:
        raise ValueError("Audio payload is empty.")

    if not patient_id or not str(patient_id).strip():
        raise ValueError("patient_id is required.")

    ext = ".wav"
    if original_filename:
        candidate_ext = Path(original_filename).suffix.lower()
        if candidate_ext in {".wav", ".mp3", ".flac"}:
            ext = candidate_ext

    safe_patient_id = re.sub(r"[^A-Za-z0-9_-]", "", str(patient_id).strip())
    if not safe_patient_id:
        raise ValueError("patient_id contains no valid characters.")

    timestamp_utc = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    stage_file_name = f"{safe_patient_id}_{timestamp_utc}{ext}"

    temp_dir = Path(tempfile.mkdtemp(prefix="resp-audio-"))
    local_file = temp_dir / stage_file_name

    try:
        local_file.write_bytes(audio_bytes)

        with SnowflakeClient() as client:
            cur = client.cursor()
            try:
                cur.execute(f"CREATE STAGE IF NOT EXISTS {_INGESTED_STAGE}")
                cur.execute(
                    f"PUT 'file://{local_file.as_posix()}' @{_INGESTED_STAGE} AUTO_COMPRESS=FALSE OVERWRITE=FALSE"
                )
            finally:
                cur.close()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return stage_file_name


def calculate_audio_metadata(audio: np.ndarray, sr: int) -> dict:
    """
    Calculates physical metadata from a normalized audio signal.

    Required fields for INGESTED_SOUNDS_METADATA:
    - SAMPLE_RATE: Number of samples per second
    - DURATION_S: Total length in seconds
    - N_SAMPLES: Total number of digital samples
    - AMPLITUDE_MAX: Peak absolute value
    - RMS: Root Mean Square (average power)
    """
    if sr <= 0:
        raise ValueError("Sample rate must be > 0.")
    if audio is None or len(audio) == 0:
        raise ValueError("Audio signal is empty.")

    duration_s = float(len(audio) / sr)
    n_samples = int(len(audio))
    amplitude_max = float(np.max(np.abs(audio)))
    # RMS calculation: square root of the arithmetic mean of the squares of the values
    rms = float(np.sqrt(np.mean(audio**2)))

    return {
        "sample_rate": sr,
        "duration_s": round(duration_s, 4),
        "n_samples": n_samples,
        "amplitude_max": round(amplitude_max, 6),
        "rms": round(rms, 6),
    }


def insert_ingested_sound_metadata(
    file_name: str,
    patient_id: str,
    pharmacie_id: str | None,
    audio: np.ndarray,
    sr: int,
) -> dict:
    """Insert one row in INGESTED.INGESTED_SOUNDS_METADATA for a patient audio file."""
    if not file_name:
        raise ValueError("file_name is required.")
    if not patient_id or not str(patient_id).strip():
        raise ValueError("patient_id is required.")

    metadata = calculate_audio_metadata(audio=audio, sr=sr)

    with SnowflakeClient() as client:
        cur = client.cursor()
        try:
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {_INGESTED_METADATA_TABLE} (
                    FILE_NAME VARCHAR(16777216) NOT NULL,
                    PATIENT_ID VARCHAR(15) NOT NULL,
                    PHARMACIE_ID VARCHAR(14),
                    RECORDED_AT TIMESTAMP_NTZ(9) NOT NULL DEFAULT CURRENT_TIMESTAMP(),
                    SAMPLE_RATE NUMBER(38,0),
                    DURATION_S FLOAT,
                    N_SAMPLES NUMBER(38,0),
                    AMPLITUDE_MAX FLOAT,
                    RMS FLOAT,
                    PRIMARY KEY (FILE_NAME)
                )
                """
            )
            cur.execute(
                f"""
                INSERT INTO {_INGESTED_METADATA_TABLE}
                    (file_name, patient_id, pharmacie_id, sample_rate, duration_s, n_samples, amplitude_max, rms)
                VALUES
                    (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    file_name,
                    str(patient_id).strip(),
                    str(pharmacie_id).strip() if pharmacie_id else None,
                    metadata["sample_rate"],
                    metadata["duration_s"],
                    metadata["n_samples"],
                    metadata["amplitude_max"],
                    metadata["rms"],
                ),
            )
        finally:
            cur.close()

    return metadata


def upload_patient_audio_with_metadata(
    audio_bytes: bytes,
    audio: np.ndarray,
    sr: int,
    patient_id: str,
    pharmacie_id: str | None,
    original_filename: str | None = None,
) -> tuple[str, dict]:
    """
    Upload patient audio to stage then persist its metadata row.
    Returns (stage_file_name, metadata).
    """
    stage_file_name = upload_patient_audio_to_stage(
        audio_bytes=audio_bytes,
        patient_id=patient_id,
        original_filename=original_filename,
    )
    metadata = insert_ingested_sound_metadata(
        file_name=stage_file_name,
        patient_id=patient_id,
        pharmacie_id=pharmacie_id,
        audio=audio,
        sr=sr,
    )
    return stage_file_name, metadata
