import io
import json
import os
import shutil
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, filtfilt

from backend.utils.snowflake_client import SnowflakeClient


DATABASE = os.getenv("SNOWFLAKE_DATABASE")
SCHEMA_APP = os.getenv("SNOWFLAKE_SCHEMA_APP")
SCHEMA_INGESTED = os.getenv("SNOWFLAKE_SCHEMA_INGESTED")


# ── Stages ────────────────────────────────────────────────────────────────────
SOURCE_STAGE      = f"@{DATABASE}.{SCHEMA_INGESTED}.STG_INGESTED_SOUNDS"
PROCESSED_STAGE   = f"@{DATABASE}.{SCHEMA_INGESTED}.STG_PROCESSED_SOUNDS"

# ── Tables ────────────────────────────────────────────────────────────────────
SOURCE_METADATA_TABLE  = f"{DATABASE}.{SCHEMA_INGESTED}.INGESTED_SOUNDS_METADATA"
PROC_METADATA_TABLE    = f"{DATABASE}.{SCHEMA_INGESTED}.PROCESSED_SOUNDS_METADATA"

# ── Audio params ──────────────────────────────────────────────────────────────
TARGET_SR          = 22050
TARGET_DURATION_S  = 6
SILENCE_DB        = 30
BANDPASS_LOW       = 100
BANDPASS_HIGH      = 2000

# ── Filtre passe-bande ────────────────────────────────────────────────────────
def bandpass_filter(y, sr, low=BANDPASS_LOW, high=BANDPASS_HIGH, order=4):
    """
    Butterworth passe-bande 100-2000 Hz.
    Guard : filtfilt nécessite au moins padlen+1 samples (≈ 3*order*2 = 24).
    En dessous on retourne le signal brut plutôt que de crasher.
    """
    min_samples = 3 * order * 2 + 1
    if len(y) < min_samples:
        return y  # signal trop court pour filtrer proprement
    nyq = sr / 2
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, y)


# ── Trim silence (début ET fin) ───────────────────────────────────────────────
def trim_silence(y, sr, top_db=SILENCE_DB):
    """
    Supprime le silence en début ET en fin.
    Retourne (y_trimmed, leading_s, trailing_s).

    Fix v2 : l'ancienne version ne supprimait que le début
    (audio[trim_index[0]:] ignorait trim_index[1]).
    """
    _, indices = librosa.effects.trim(y, top_db=top_db)
    y_trimmed    = y[indices[0]:indices[1]]
    leading_s    = indices[0] / sr
    trailing_s   = (len(y) - indices[1]) / sr
    return y_trimmed, leading_s, trailing_s


# ── Pad / crop ────────────────────────────────────────────────────────────────
def pad_or_crop(y, sr, target_s=TARGET_DURATION_S):
    """Padding à droite ou crop au début pour atteindre target_s secondes."""
    target = int(target_s * sr)
    if len(y) < target:
        return np.pad(y, (0, target - len(y)))
    return y[:target]


# ── Normalisation amplitude ───────────────────────────────────────────────────
def normalize_amplitude(y, target_peak=0.95):
    """
    Ramène le pic d'amplitude à target_peak.
    Fix v2 : absent de v1, causait des disparités de volume entre fichiers.
    """
    peak = np.max(np.abs(y))
    if peak > 0:
        return y / peak * target_peak
    return y


# ── Pipeline complet pour UN fichier ─────────────────────────────────────────
def process_file(y_raw, sr_raw, file_name, class_name):
    """
    Pipeline unifié (remplace process_one_file + process_audio).
    Retourne (y_processed, sr, meta_dict) ou (None, sr, meta_dict) si rejeté.

    Étapes :
      1. Resample → TARGET_SR
      2. Bandpass filter 100-2000 Hz
      3. Trim silence début ET fin
      4. Rejet si durée < MIN_DURATION_S
      5. Pad / crop → TARGET_DURATION_S
      6. Normalisation amplitude
    """
    original_duration = len(y_raw) / sr_raw

    # 1. Resample
    if sr_raw != TARGET_SR:
        y = librosa.resample(y_raw, orig_sr=sr_raw, target_sr=TARGET_SR)
        sr = TARGET_SR
    else:
        y, sr = y_raw.copy(), sr_raw

    # 2. Bandpass
    y = bandpass_filter(y, sr)

    # 3. Trim silence début ET fin
    y, leading_s, trailing_s = trim_silence(y, sr)
    stripped_duration = len(y) / sr

    # 4. Rejet si trop court
    # if stripped_duration < MIN_DURATION_S:
    #     return None, sr, {
    #         "FILE_NAME":           file_name,
    #         "CLASS":               class_name,
    #         "ACTION":              "SKIPPED_TOO_SHORT",
    #         "ORIGINAL_DURATION_S": round(original_duration, 4),
    #         "STRIPPED_DURATION_S": round(stripped_duration, 4),
    #         "FINAL_DURATION_S":    None,
    #         "LEADING_SILENCE_S":   round(leading_s, 4),
    #         "TRAILING_SILENCE_S":  round(trailing_s, 4),
    #         "AMPLITUDE_MAX":       None,
    #         "RMS":                 None,
    #     }

    # 5. Pad / crop
    y = pad_or_crop(y, sr)

    # 6. Normalisation
    y = normalize_amplitude(y)

    # Labelling action
    target_samples = int(TARGET_DURATION_S * sr)
    if leading_s > 0 and trailing_s > 0:
        action = "STRIPPED_BOTH_ENDS"
    elif leading_s > 0:
        action = "STRIPPED_LEADING"
    elif trailing_s > 0:
        action = "STRIPPED_TRAILING"
    elif len(y) == target_samples and original_duration * sr > target_samples:
        action = "CROPPED"
    elif len(y) == target_samples and original_duration * sr < target_samples:
        action = "PADDED"
    else:
        action = "PROCESSED"

    meta = {
        "FILE_NAME":           file_name,
        "CLASS":               class_name,
        "ACTION":              action,
        "ORIGINAL_DURATION_S": round(original_duration, 4),
        "STRIPPED_DURATION_S": round(stripped_duration, 4),
        "FINAL_DURATION_S":    round(len(y) / sr, 4),
        "LEADING_SILENCE_S":   round(leading_s, 4),
        "TRAILING_SILENCE_S":  round(trailing_s, 4),
        "SAMPLE_RATE":         sr,
        "N_SAMPLES":           len(y),
        "AMPLITUDE_MAX":       round(float(np.max(np.abs(y))), 6),
        "RMS":                 round(float(np.sqrt(np.mean(y**2))), 6),
    }
    return y, sr, meta


# ── Full pipeline: process + store ────────────────────────────────────────────
def process_and_store_ingested_audio(
    original_file_name: str,
    patient_id: str,
    pharmacie_id: str | None,
    audio_bytes: bytes,
) -> str:
    """
    Full preprocessing pipeline for one patient audio upload:
      1. Load raw audio from bytes (original SR, no prior processing)
      2. Run process_file: resample → bandpass → trim → pad/crop → normalize
      3. Compute mel spectrogram → serialize as JSON (stored in VARIANT column)
      4. Upload processed WAV to STG_PROCESSED_SOUNDS
      5. Insert one row in PROCESSED_SOUNDS_METADATA

    Returns the processed file name (key in STG_PROCESSED_SOUNDS).
    """
    # 1. Load raw audio
    y_raw, sr_raw = librosa.load(io.BytesIO(audio_bytes), sr=None)

    # 2. Full processing pipeline
    y, sr, meta = process_file(y_raw, sr_raw, original_file_name, class_name=None)

    # 3. Mel spectrogram → JSON-serialisable list[128][T]
    mel = librosa.power_to_db(
        librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, n_fft=2048, hop_length=512),
        ref=np.max,
    )
    mel_json_str = json.dumps(mel.tolist())

    # 4. Upload processed WAV to STG_PROCESSED_SOUNDS
    stem = os.path.splitext(original_file_name)[0]
    processed_file_name = f"{stem}_processed.wav"

    temp_dir = Path(tempfile.mkdtemp(prefix="resp-proc-"))
    local_file = temp_dir / processed_file_name
    try:
        sf.write(str(local_file), y, sr, subtype="PCM_16")
        with SnowflakeClient() as client:
            cur = client.cursor()
            try:
                cur.execute(f"CREATE STAGE IF NOT EXISTS {PROCESSED_STAGE.lstrip('@')}")
                cur.execute(
                    f"PUT 'file://{local_file.as_posix()}' {PROCESSED_STAGE}"
                    " AUTO_COMPRESS=FALSE OVERWRITE=FALSE"
                )
            finally:
                cur.close()
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    # 5. Insert metadata row
    with SnowflakeClient() as client:
        cur = client.cursor()
        try:
            cur.execute(
                f"""
                INSERT INTO {PROC_METADATA_TABLE}
                    (file_name, original_file_name, patient_id, pharmacie_id,
                     action, original_duration_s, stripped_duration_s, final_duration_s,
                     leading_silence_s, trailing_silence_s, sample_rate, n_samples,
                     amplitude_max, rms, mel_spectogram)
                VALUES
                    (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, PARSE_JSON(%s))
                """,
                (
                    processed_file_name,
                    original_file_name,
                    str(patient_id).strip(),
                    str(pharmacie_id).strip() if pharmacie_id else None,
                    meta["ACTION"],
                    meta["ORIGINAL_DURATION_S"],
                    meta["STRIPPED_DURATION_S"],
                    meta["FINAL_DURATION_S"],
                    meta["LEADING_SILENCE_S"],
                    meta["TRAILING_SILENCE_S"],
                    meta["SAMPLE_RATE"],
                    meta["N_SAMPLES"],
                    meta["AMPLITUDE_MAX"],
                    meta["RMS"],
                    mel_json_str,
                ),
            )
        finally:
            cur.close()

    return processed_file_name


