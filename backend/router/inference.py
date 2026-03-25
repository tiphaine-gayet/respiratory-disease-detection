"""
Inference router: calls the RUN_INFERENCE Snowflake procedure on a single mel .npy,
maps probabilities to an action label, and inserts the result into APP.PREDICTIONS.
"""

from __future__ import annotations

import json
import os

from backend.utils.snowflake_client import SnowflakeClient

DATABASE     = os.getenv("SNOWFLAKE_DATABASE")
SCHEMA_MODEL = os.getenv("SNOWFLAKE_SCHEMA_MODEL")
SCHEMA_APP   = os.getenv("SNOWFLAKE_SCHEMA_APP")

PROC_PATH         = f"{DATABASE}.{SCHEMA_MODEL}.RUN_INFERENCE"
PREDICTIONS_TABLE = f"{DATABASE}.{SCHEMA_APP}.PREDICTIONS"

# Maps result dict keys → class names used by get_action
_PCT_TO_CLASS: dict[str, str] = {
    "pct_asthma":    "Asthma",
    "pct_copd":      "COPD",
    "pct_bronchial": "Bronchial",
    "pct_pneumonia": "Pneumonia",
    "pct_healthy":   "Healthy",
}

_ACTIONS: dict[str, list[tuple[float, str, str]]] = {
    "Healthy": [
        (0.60, "RAS",
         "Aucune anomalie détectée. Suivi préventif annuel suffisant. "
         "Consulter si dyspnée ou toux persistante > 3 semaines."),
        (0.00, "surveillance_7j",
         "Profil sain probable mais certitude faible. Consultation de contrôle conseillée si symptômes persistants."),
    ],
    "Asthma": [
        (0.85, "consultation_24h",
         "Asthme très probable. Consultation sous 24h, traitement de fond (CSI ± LABA) à évaluer. "
         "Vérifier disponibilité d'un bronchodilatateur d'urgence (salbutamol). "
         "⚠ SpO₂ < 92% ou crise nocturne → SAMU (15)."),
        (0.50, "surveillance_48h",
         "Profil asthmatique probable. Suivi médical 48–72h, spirométrie pour confirmer. "
         "⚠ Sibilances ou dyspnée aiguë → consultation urgente."),
        (0.00, "surveillance_7j",
         "Asthme possible mais incertain. Téléconsultation ou spirométrie recommandée. "
         "Éviter déclencheurs connus (froid, poussière, effort)."),
    ],
    "Pneumonia": [
        (0.85, "urgent_6h",
         "Pneumonie très probable. Prise en charge sous 6h, score CRB-65 à évaluer pour hospitalisation. "
         "⚠ SpO₂ < 90% ou confusion → SAMU (15) immédiat."),
        (0.50, "consultation_24h",
         "Pneumonie probable. Consultation dans les 24h, antibiothérapie probable (amoxicilline 1ère intention). "
         "⚠ SpO₂ < 94% ou FR > 25/min → urgences."),
        (0.00, "surveillance_48h",
         "Pneumonie possible. Radio thoracique + bilan biologique (NFS, CRP) nécessaires. "
         "Surveiller température, fréquence respiratoire et état général."),
    ],
    "COPD": [
        (0.85, "consultation_24h",
         "BPCO très probable. Consultation pneumologique sous 48h, stadification GOLD indispensable. "
         "⚠ SpO₂ < 88% → oxygénothérapie + urgences ; exacerbation sévère → hospitalisation."),
        (0.50, "surveillance_7j",
         "BPCO probable. Consultation pneumologique sous 7 jours, arrêt du tabac prioritaire. "
         "Évaluation bronchodilatateurs longue durée (LAMA/LABA), vaccination grippe + pneumocoque. "
         "⚠ Exacerbation (↑ dyspnée + ↑ expectoration) → consultation sous 24h."),
        (0.00, "surveillance_7j",
         "BPCO possible. Spirométrie nécessaire pour confirmer (VEMS/CVF < 0.70 post-bronchodilatateur). "
         "Évaluer tabagisme et expositions professionnelles."),
    ],
    "Bronchial": [
        (0.85, "surveillance_48h",
         "Bronchite aiguë très probable. Consultation sous 48h pour écarter surinfection. "
         "⚠ FR > 25/min ou SpO₂ < 94% → éliminer pneumonie en urgence."),
        (0.50, "surveillance_7j",
         "Bronchite aiguë probable. Généralement virale, résolution en 7–14j. "
         "Pas d'antibiotiques en 1ère intention sauf expectoration purulente > 10j ou terrain fragile."),
        (0.00, "surveillance_7j",
         "Bronchite possible. Traitement symptomatique (repos, hydratation). "
         "Consulter si fièvre > 38.5°C au-delà de 5 jours."),
    ],
}


def get_action(pred_class: str, confidence: float) -> tuple[str, str]:
    """
    Map a predicted class + confidence (0.0–1.0) to (action_code, detailed_text).
    action_code: "RAS" | "surveillance_7j" | "surveillance_48h" | "consultation_24h" | "urgent_6h"
    """
    levels = _ACTIONS.get(pred_class)
    if not levels:
        return "surveillance_7j", f"Classe non reconnue : {pred_class}. Téléconsultation conseillée."

    for threshold, action, detail in levels:
        if confidence >= threshold:
            return action, f"Détection à {confidence:.0%} : {detail}"

    return "surveillance_7j", f"Résultat incertain ({pred_class}, {confidence:.0%}). Téléconsultation conseillée."


def run_inference_and_store(
    mel_npy_filename: str,
    patient_id: str,
    pharmacie_id: str | None,
    audio_file_name: str,
) -> dict:
    """
    1. Call MODEL.RUN_INFERENCE(mel_npy_filename) → probabilities + model_version
    2. Derive predicted class and action label in Python
    3. Insert one row into APP.PREDICTIONS
    Returns the full prediction dict.
    """
    with SnowflakeClient() as client:
        cur = client.cursor()
        try:
            cur.execute(f"CALL {PROC_PATH}(%s)", (mel_npy_filename,))
            row = cur.fetchone()
            if row is None:
                raise RuntimeError(f"RUN_INFERENCE returned no result for {mel_npy_filename!r}")

            result: dict = row[0] if isinstance(row[0], dict) else json.loads(row[0])

            # Derive predicted class from the highest probability
            pred_class = max(_PCT_TO_CLASS, key=lambda k: result.get(k, 0.0))
            pred_class = _PCT_TO_CLASS[pred_class]

            action, detailed_action = get_action(pred_class, result["pct_confiance"] / 100)

            cur.execute(
                f"""
                INSERT INTO {PREDICTIONS_TABLE} (
                    patient_id, pharmacie_id, audio_file_name,
                    pct_asthma, pct_copd, pct_bronchial, pct_pneumonia, pct_healthy,
                    pct_confiance, action, detailed_action, model_version
                ) VALUES (
                    %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s
                )
                """,
                (
                    str(patient_id).strip(),
                    str(pharmacie_id).strip() if pharmacie_id else None,
                    audio_file_name,
                    result["pct_asthma"],
                    result["pct_copd"],
                    result["pct_bronchial"],
                    result["pct_pneumonia"],
                    result["pct_healthy"],
                    result["pct_confiance"],
                    action,
                    detailed_action,
                    result.get("model_version"),
                ),
            )
        finally:
            cur.close()

    return {
        **result,
        "pred_class":      pred_class,
        "action":          action,
        "detailed_action": detailed_action,
    }
