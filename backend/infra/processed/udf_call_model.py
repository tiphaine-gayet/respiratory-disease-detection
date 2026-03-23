"""
Snowflake UDF for model inference payload generation.

This module exposes a deployable UDF that:
1. Accepts MEL and MFCC arrays as JSON strings.
2. Produces class probabilities and prediction metadata.
3. Returns a VARIANT payload ready to be logged.

Notes:
- The UDF is deployable as-is in Snowflake (only depends on numpy).
- Logging to a table is done outside the UDF via helper functions below.
"""

from __future__ import annotations

import textwrap
from typing import Any, Dict

PREDICTION_TABLE = "M2_ISD_EQUIPE_1_DB.PROCESSED.PREDICTIONS"


def create_prediction_table(session, table_name: str = PREDICTION_TABLE) -> None:
    """Create prediction logging table if it does not exist."""
    session.sql(
        f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            PREDICTION_ID     VARCHAR(36)     NOT NULL,
            USER_ID           VARCHAR(100),
            DEVICE_ID         VARCHAR(100),
            PREDICTED_CLASS   VARCHAR(50)     NOT NULL,
            CONFIDENCE        FLOAT           NOT NULL,
            PROB_ASTHMA       FLOAT           NOT NULL,
            PROB_BRONCHIAL    FLOAT           NOT NULL,
            PROB_COPD         FLOAT           NOT NULL,
            PROB_HEALTHY      FLOAT           NOT NULL,
            PROB_PNEUMONIA    FLOAT           NOT NULL,
            USE_TTA           BOOLEAN         NOT NULL,
            N_TTA_PASSES      INTEGER         NOT NULL,
            MODEL_VERSION     VARCHAR(50)     NOT NULL,
            INFERENCE_TIME_MS FLOAT           NOT NULL,
            GROUND_TRUTH      VARCHAR(50),
            IS_CORRECT        BOOLEAN,
            CREATED_AT        TIMESTAMP_NTZ   NOT NULL
        )
        """
    ).collect()


def log_prediction_result(session, result: Dict[str, Any], table_name: str = PREDICTION_TABLE) -> None:
    """Insert one prediction payload returned by CALL_MODEL_UDF into the logging table."""
    probs = result.get("probabilities", {})
    row = {
        "PREDICTION_ID": result.get("prediction_id"),
        "USER_ID": result.get("user_id"),
        "DEVICE_ID": result.get("device_id", "0000"),
        "PREDICTED_CLASS": result.get("predicted_class"),
        "CONFIDENCE": result.get("confidence", 0.0),
        "PROB_ASTHMA": probs.get("Asthma", 0.0),
        "PROB_BRONCHIAL": probs.get("Bronchial", 0.0),
        "PROB_COPD": probs.get("COPD", 0.0),
        "PROB_HEALTHY": probs.get("Healthy", 0.0),
        "PROB_PNEUMONIA": probs.get("Pneumonia", 0.0),
        "USE_TTA": result.get("use_tta", False),
        "N_TTA_PASSES": result.get("n_tta_passes", 1),
        "MODEL_VERSION": result.get("model_version", "v2_mel_mfcc"),
        "INFERENCE_TIME_MS": result.get("inference_time_ms", 0.0),
        "GROUND_TRUTH": result.get("ground_truth"),
        "IS_CORRECT": result.get("is_correct"),
        "CREATED_AT": result.get("created_at"),
    }

    session.create_dataframe([row]).write.mode("append").save_as_table(
        table_name,
        create_temp_table=False,
    )


def deploy_udf_call_model(session, udf_name: str = "CALL_MODEL_UDF") -> None:
    """
    Deploy CALL_MODEL_UDF as a permanent Snowflake Python UDF.

    UDF signature:
      CALL_MODEL_UDF(
        mel_json STRING,
        mfcc_json STRING,
        user_id STRING,
        device_id STRING,
        ground_truth STRING,
        use_tta BOOLEAN,
        n_passes INTEGER
      ) -> VARIANT
    """
    udf_sql = textwrap.dedent(
        """
        CREATE OR REPLACE FUNCTION __UDF_NAME__(
            mel_json STRING,
            mfcc_json STRING,
            user_id STRING,
            device_id STRING,
            ground_truth STRING,
            use_tta BOOLEAN,
            n_passes INTEGER
        )
        RETURNS VARIANT
        LANGUAGE PYTHON
        RUNTIME_VERSION = '3.11'
        PACKAGES = ('numpy')
        HANDLER = 'call_model_udf_handler'
        AS
        $$
        import json
        import time
        import uuid
        from datetime import datetime

        import numpy as np


        def _safe_array(payload):
            arr = np.asarray(payload, dtype=np.float32)
            if arr.ndim == 0:
                arr = arr.reshape(1, 1)
            return arr


        def _softmax(x):
            z = x - np.max(x)
            e = np.exp(z)
            return e / np.sum(e)


        def _compute_probs(mel, mfcc, use_tta, n_passes):
            # Baseline deterministic logits from summary stats.
            mel_energy = float(np.mean(np.abs(mel)))
            mfcc_mean = float(np.mean(mfcc))
            mfcc_std = float(np.std(mfcc))

            base_logits = np.array([
                0.80 * mel_energy + 0.20 * mfcc_mean,   # Asthma
                0.60 * mfcc_std - 0.10 * mfcc_mean,     # Bronchial
                0.50 * mel_energy + 0.50 * mfcc_std,    # COPD
                -0.30 * mel_energy + 0.40 * mfcc_mean,  # Healthy
                -0.20 * mfcc_std + 0.10 * mel_energy,   # Pneumonia
            ], dtype=np.float64)

            passes = int(n_passes) if use_tta else 1
            passes = max(1, min(passes, 20))

            rng = np.random.default_rng(42)
            probs_list = []
            for _ in range(passes):
                if use_tta:
                    noise = rng.normal(0.0, 0.01, size=base_logits.shape)
                    logits = base_logits + noise
                else:
                    logits = base_logits
                probs_list.append(_softmax(logits))

            return np.mean(np.stack(probs_list), axis=0), passes


        def call_model_udf_handler(
            mel_json,
            mfcc_json,
            user_id,
            device_id,
            ground_truth,
            use_tta,
            n_passes,
        ):
            t0 = time.time()

            try:
                mel = _safe_array(json.loads(mel_json))
                mfcc = _safe_array(json.loads(mfcc_json))
            except Exception as exc:
                return {
                    "status": "ERROR",
                    "error": f"Invalid JSON input: {str(exc)[:300]}",
                    "prediction_id": str(uuid.uuid4()),
                    "created_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                }

            probs, used_passes = _compute_probs(mel, mfcc, bool(use_tta), int(n_passes or 1))

            labels = ["Asthma", "Bronchial", "COPD", "Healthy", "Pneumonia"]
            prob_map = {labels[i]: float(probs[i]) for i in range(len(labels))}
            predicted = max(prob_map, key=prob_map.get)
            confidence = float(prob_map[predicted])

            gt = ground_truth if ground_truth not in (None, "", "NULL") else None
            is_correct = (predicted == gt) if gt else None

            return {
                "status": "SUCCESS",
                "prediction_id": str(uuid.uuid4()),
                "user_id": user_id,
                "device_id": "0000",
                "predicted_class": predicted,
                "confidence": round(confidence, 6),
                "probabilities": {k: round(v, 6) for k, v in prob_map.items()},
                "use_tta": bool(use_tta),
                "n_tta_passes": used_passes,
                "model_version": "v2_mel_mfcc",
                "inference_time_ms": round((time.time() - t0) * 1000.0, 2),
                "ground_truth": gt,
                "is_correct": is_correct,
                "created_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            }
        $$
        """
    ).replace("__UDF_NAME__", udf_name)

    session.sql(udf_sql).collect()


def deploy_prediction_udf_stack(session, udf_name: str = "CALL_MODEL_UDF") -> None:
    """Convenience function: create table + deploy CALL_MODEL_UDF."""
    create_prediction_table(session)
    deploy_udf_call_model(session, udf_name=udf_name)


if __name__ == "__main__":
    from snowflake.snowpark.context import get_active_session

    active_session = get_active_session()
    deploy_prediction_udf_stack(active_session)
    print("CALL_MODEL_UDF deployed successfully.")
