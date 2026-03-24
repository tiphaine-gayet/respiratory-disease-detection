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
from datetime import date

import pandas as pd

from backend.utils.snowflake_client import SnowflakeClient

# ── Snowflake identifiers (mirror the pattern used in infra/) ─────────────────
_DB     = os.getenv("SNOWFLAKE_DATABASE")
_SCHEMA = os.getenv("SNOWFLAKE_SCHEMA_APP")

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
