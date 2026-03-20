# ============================================================
# IMPORTS
# ============================================================
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import os
import tempfile
from scipy.signal import butter, filtfilt
from dotenv import load_dotenv
from snowflake.snowpark import Session

# ============================================================
# CONNEXION SNOWFLAKE
# ============================================================
load_dotenv()

connection_params = {
    "account": os.getenv("SNOWFLAKE_ACCOUNT"),
    "user": os.getenv("SNOWFLAKE_USER"),
    "password": os.getenv("SNOWFLAKE_TOKEN"),
    "role": os.getenv("SNOWFLAKE_ROLE"),
    "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE"),
    "database": os.getenv("SNOWFLAKE_DATABASE"),
    "schema": "RAW",
}

session = Session.builder.configs(connection_params).create()
print("Connected ✓")

session.sql(
    "CREATE STAGE IF NOT EXISTS M2_ISD_EQUIPE_1_DB.PROCESSED.STG_RESPIRATORY_SOUNDS"
).collect()
print("Stage ready ✓")

# ============================================================
# CONFIG
# ============================================================
SOURCE_STAGE = "@M2_ISD_EQUIPE_1_DB.RAW.STG_RESPIRATORY_SOUNDS"
TARGET_STAGE = "@M2_ISD_EQUIPE_1_DB.PROCESSED.STG_RESPIRATORY_SOUNDS"
SOURCE_METADATA_TABLE = "M2_ISD_EQUIPE_1_DB.RAW.RESPIRATORY_SOUNDS_METADATA"
TARGET_METADATA_TABLE = "M2_ISD_EQUIPE_1_DB.PROCESSED.RESPIRATORY_SOUNDS_METADATA"
TARGET_SR = 22050
MIN_DURATION_S = 4
TARGET_DURATION_S = 6
SILENCE_THRESHOLD_DB = 30
BANDPASS_LOW = 100
BANDPASS_HIGH = 2000

CLASS_TO_DIR = {
    "Asthma": "asthma",
    "Bronchial": "bronchial",
    "COPD": "copd",
    "Pneumonia": "pneumonia",
    "Healthy": "healthy",
}


def trim_leading_silence(audio, sr, threshold_db=SILENCE_THRESHOLD_DB):
    """
    Supprime le silence au début du signal.
    Retourne le signal trimé + durée de silence supprimée.
    """
    _, trim_index = librosa.effects.trim(audio, top_db=threshold_db)
    trimmed = audio[trim_index[0] :]
    leading_silence_s = trim_index[0] / sr
    return trimmed, leading_silence_s


def pad_or_crop(audio, sr, target_duration=TARGET_DURATION_S):
    """
    Standardise la durée à target_duration secondes.
    - Trop court → padding avec des zéros à droite
    - Trop long  → crop au début
    """
    target_samples = int(target_duration * sr)
    if len(audio) < target_samples:
        return np.pad(audio, (0, target_samples - len(audio)))
    else:
        return audio[:target_samples]


def bandpass_filter(signal, sr, lowcut=BANDPASS_LOW, highcut=BANDPASS_HIGH):
    """
    Filtre passe-bande Butterworth ordre 4 : 100-2000 Hz.
    - Supprime < 100 Hz  : vibrations stéthoscope, bruits de manipulation
    - Supprime > 2000 Hz : bruit électronique, respiration du soignant
    Appliqué sur signal numpy 1D avant trim et pad/crop.
    """
    nyquist = sr / 2
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(N=4, Wn=[low, high], btype="band")
    return filtfilt(b, a, signal)


def process_one_file(local_path, file_name, class_name):
    """
    Ordre des étapes :
      1. Filtre passe-bande
      2. Trim silence
      3. Pad / crop
    """
    audio, sr = librosa.load(local_path, sr=None)
    original_duration = len(audio) / sr

    # --- Étape 0 : resample  ---
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR

    # --- Étape 1 : filtre passe-bande ---
    filtered = bandpass_filter(audio, sr)

    # --- Étape 2 : trim silence ---
    trimmed, leading_silence_s = trim_leading_silence(filtered, sr)
    stripped_duration = len(trimmed) / sr

    if stripped_duration < MIN_DURATION_S:
        return (
            None,
            None,
            {
                "FILE_NAME": file_name,
                "CLASS": class_name,
                "ACTION": "SKIPPED_TOO_SHORT",
                "ORIGINAL_DURATION_S": round(original_duration, 4),
                "STRIPPED_DURATION_S": round(stripped_duration, 4),
                "FINAL_DURATION_S": None,
                "SILENCE_STRIPPED_S": round(leading_silence_s, 4),
            },
            None,
        )


    # --- Étape 3 : pad / crop  ---
    padded = pad_or_crop(trimmed, sr)
    final_duration = len(padded) / sr

    # Déterminer l'action
    target_samples = int(TARGET_DURATION_S * sr)
    if leading_silence_s == 0 and len(trimmed) >= target_samples:
        action = "TRIMMED_TO_TARGET"
    elif leading_silence_s == 0:
        action = "PADDED"
    elif len(trimmed) >= target_samples:
        action = "STRIPPED_AND_TRIMMED"
    else:
        action = "PROCESSED"

    amplitude_max = float(np.max(np.abs(padded)))
    rms = float(np.sqrt(np.mean(padded**2)))

    modification = {
        "FILE_NAME": file_name,
        "CLASS": class_name,
        "ACTION": action,
        "ORIGINAL_DURATION_S": round(original_duration, 4),
        "STRIPPED_DURATION_S": round(stripped_duration, 4),
        "FINAL_DURATION_S": round(final_duration, 4),
        "SILENCE_STRIPPED_S": round(leading_silence_s, 4),
    }
    processed_meta = {
        "FILE_NAME": file_name,
        "CLASS": class_name,
        "SAMPLE_RATE": sr,
        "DURATION_S": final_duration,
        "N_SAMPLES": len(padded),
        "AMPLITUDE_MAX": round(amplitude_max, 6),
        "RMS": round(rms, 6),
    }

    return padded, sr, modification, processed_meta

def process_audio(y, sr):
    # resample
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR

    # filter
    y = bandpass_filter(y, sr)

    # trim
    y, leading_silence_s = trim_leading_silence(y, sr)

    # skip short
    duration = len(y) / sr
    if duration < MIN_DURATION_S:
        return None, sr, {"skipped": True, "duration": duration}

    # pad/crop
    y = pad_or_crop(y, sr)

    return y, sr, {
        "skipped": False,
        "duration": duration,
        "leading_silence_s": leading_silence_s
    }

def compute_metadata(y, sr, file_name, class_name):
    duration = len(y) / sr
    amplitude = float(np.max(np.abs(y)))
    rms = float(np.sqrt(np.mean(y**2)))

    return {
        "FILE_NAME": file_name,
        "CLASS": class_name,
        "SAMPLE_RATE": sr,
        "DURATION_S": round(duration, 4),
        "N_SAMPLES": len(y),
        "AMPLITUDE_MAX": round(amplitude, 6),
        "RMS": round(rms, 6),
    }

def run_preprocessing_pipeline(
    session,
    source_stage,
    target_stage,
    source_metadata_table,
    target_metadata_table,
    class_to_dir
):
    metadata_df = session.sql(f"SELECT * FROM {source_metadata_table}").to_pandas()

    modifications = []
    processed_metadata = []

    with tempfile.TemporaryDirectory() as tmpdir:
        src_dir = os.path.join(tmpdir, "source")
        dst_dir = os.path.join(tmpdir, "processed")
        os.makedirs(src_dir, exist_ok=True)
        os.makedirs(dst_dir, exist_ok=True)

        for class_name, subdir in CLASS_TO_DIR.items():
            class_files = metadata_df[metadata_df["CLASS"] == class_name]
            if class_files.empty:
                continue

            class_src = os.path.join(src_dir, subdir)
            class_dst = os.path.join(dst_dir, subdir)
            os.makedirs(class_src, exist_ok=True)
            os.makedirs(class_dst, exist_ok=True)

            session.file.get(f"{SOURCE_STAGE}/{subdir}", class_src)
            print(f"Downloaded {class_name} files to {class_src}")

            for _, row in class_files.iterrows():
                file_name = row["FILE_NAME"]
                local_path = os.path.join(class_src, file_name)

                if not os.path.exists(local_path):
                    modifications.append(
                        {
                            "FILE_NAME": file_name,
                            "CLASS": class_name,
                            "ACTION": "SKIPPED_NOT_FOUND",
                            "ORIGINAL_DURATION_S": row["DURATION_S"],
                            "STRIPPED_DURATION_S": None,
                            "FINAL_DURATION_S": None,
                            "SILENCE_STRIPPED_S": None,
                        }
                    )
                    continue

                try:
                    padded, sr, mod, meta = process_one_file(
                        local_path, file_name, class_name
                    )
                    modifications.append(mod)

                    if padded is not None:
                        out_path = os.path.join(class_dst, file_name)
                        sf.write(out_path, padded, sr)
                        processed_metadata.append(meta)

                except Exception as e:
                    modifications.append(
                        {
                            "FILE_NAME": file_name,
                            "CLASS": class_name,
                            "ACTION": f"ERROR: {str(e)[:100]}",
                            "ORIGINAL_DURATION_S": row["DURATION_S"],
                            "STRIPPED_DURATION_S": None,
                            "FINAL_DURATION_S": None,
                            "SILENCE_STRIPPED_S": None,
                        }
                    )

            for f in os.listdir(class_dst):
                try:
                    session.file.put(
                        os.path.join(class_dst, f),
                        f"{TARGET_STAGE}/{subdir}",
                        auto_compress=False,
                        overwrite=True
                    )
                except Exception as e:
                    print(f"❌ Upload failed for {f}: {e}")

            print(f"  {class_name}: {len(os.listdir(class_dst))} files uploaded")

    return pd.DataFrame(modifications), pd.DataFrame(processed_metadata)


modifications_df, processed_metadata_df = run_preprocessing_pipeline(
    session=session,
    source_stage=SOURCE_STAGE,
    target_stage=TARGET_STAGE,
    source_metadata_table=SOURCE_METADATA_TABLE,
    target_metadata_table=TARGET_METADATA_TABLE,
    class_to_dir=CLASS_TO_DIR
)
print(f"\nTotal files: {len(modifications_df)}")
print(modifications_df["ACTION"].value_counts().to_string())

# ============================================================
# SAUVEGARDE METADATA
# ============================================================
if not processed_metadata_df.empty:
    sp_df = session.create_dataframe(processed_metadata_df)
    sp_df.write.mode("overwrite").save_as_table(TARGET_METADATA_TABLE)
    session.sql(f"""
        ALTER TABLE {TARGET_METADATA_TABLE} 
        ADD PRIMARY KEY (FILE_NAME)
    """).collect()
    print(
        f"Metadata table created: {TARGET_METADATA_TABLE} ({len(processed_metadata_df)} rows)"
    )
else:
    print("No processed files to save.")

# ============================================================
# SUMMARY
# ============================================================
total = len(modifications_df)
kept = modifications_df[~modifications_df["ACTION"].str.startswith("SKIPPED")]
skipped_short = modifications_df[modifications_df["ACTION"] == "SKIPPED_TOO_SHORT"]
skipped_nf = modifications_df[modifications_df["ACTION"] == "SKIPPED_NOT_FOUND"]
errors = modifications_df[modifications_df["ACTION"].str.startswith("ERROR")]

print(f"\n{'='*80}\nPREPROCESSING SUMMARY\n{'='*80}")
print(f"Total source files:        {total}")
print(f"Files kept (processed):    {len(kept)}")
print(f"Skipped (too short <{MIN_DURATION_S}s):  {len(skipped_short)}")
print(f"Skipped (not found):       {len(skipped_nf)}")
print(f"Errors:                    {len(errors)}")

print(f"\n{'─'*80}\nACTION BREAKDOWN\n{'─'*80}")
print(modifications_df["ACTION"].value_counts().to_string())

print(f"\n{'─'*80}\nBREAKDOWN BY CLASS\n{'─'*80}")
summary = modifications_df.groupby(["CLASS", "ACTION"]).size().unstack(fill_value=0)
print(summary.to_string())

if not skipped_short.empty:
    print(f"\n{'─'*80}\nFILES SKIPPED (too short after stripping silence)\n{'─'*80}")
    print(
        skipped_short[
            [
                "FILE_NAME",
                "CLASS",
                "ORIGINAL_DURATION_S",
                "STRIPPED_DURATION_S",
                "SILENCE_STRIPPED_S",
            ]
        ].to_string(index=False)
    )

stripped = modifications_df[
    modifications_df["SILENCE_STRIPPED_S"].notna()
    & (modifications_df["SILENCE_STRIPPED_S"] > 0)
]
if not stripped.empty:
    print(f"\n{'─'*80}\nSILENCE STRIPPING STATS\n{'─'*80}")
    print(f"Files with leading silence stripped: {len(stripped)}")
    print(f"Avg silence stripped:  {stripped['SILENCE_STRIPPED_S'].mean():.4f}s")
    print(f"Max silence stripped:  {stripped['SILENCE_STRIPPED_S'].max():.4f}s")
    print(f"Min silence stripped:  {stripped['SILENCE_STRIPPED_S'].min():.4f}s")
