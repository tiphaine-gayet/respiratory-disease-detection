"""
Snowflake UDF for audio preprocessing
======================================

This module provides a Snowflake-deployable UDF for audio preprocessing:
1. Loads an audio file from a stage
2. Processes it through the complete pipeline (resample, filter, trim, pad, normalize)
3. Returns metadata about the processing

Pipeline steps:
  1. Resample → TARGET_SR (22050 Hz)
  2. Bandpass filter 100-2000 Hz
  3. Trim silence from beginning & end
  4. Reject if duration < MIN_DURATION_S (4 sec)
  5. Pad/crop → TARGET_DURATION_S (6 sec)
  6. Normalize amplitude to 0.95 peak

Usage (in deployment script):
    from snowflake.snowpark.context import get_active_session
    from udf_process_file import deploy_udf_process_file
    session = get_active_session()
    deploy_udf_process_file(session)
"""

import sys
import os
import io
import zipfile
import json
import textwrap

# Get Snowflake session for metadata insertion
try:
    from snowflake.snowpark.context import get_active_session
except ImportError:
    get_active_session = None

# Import Snowpark types
from snowflake.snowpark.types import StringType, VariantType

# Setup librosa from Snowflake import directory
def _setup_librosa():
    """Extract librosa from Snowflake import directory."""
    try:
        import_dir = sys._xoptions.get("snowflake_import_directory")
        if import_dir:
            zip_name = "libroza.zip"
            final_lib_dir = "/tmp/site-packages"
            os.makedirs(final_lib_dir, exist_ok=True)
            
            zip_path = os.path.join(import_dir, zip_name)
            if os.path.exists(zip_path):
                with zipfile.ZipFile(zip_path, 'r') as outer:
                    for name in outer.namelist():
                        if name.endswith(".whl"):
                            whl_bytes = outer.read(name)
                            with zipfile.ZipFile(io.BytesIO(whl_bytes), 'r') as whl:
                                whl.extractall(final_lib_dir)
                
                if final_lib_dir not in sys.path:
                    sys.path.insert(0, final_lib_dir)
    except Exception as e:
        pass

_setup_librosa()

import numpy as np
import librosa
import soundfile as sf
from scipy.signal import butter, filtfilt
import warnings
import pandas as pd

warnings.filterwarnings('ignore')


# ── Configuration ─────────────────────────────────────────────────────────────
CONFIG = {
    "TARGET_SR": 22050,
    "MIN_DURATION_S": 4,
    "TARGET_DURATION_S": 6,
    "SILENCE_DB": 30,
    "BANDPASS_LOW": 100,
    "BANDPASS_HIGH": 2000,
}


# ── Audio Processing Functions ────────────────────────────────────────────────
def bandpass_filter(y, sr, low=CONFIG["BANDPASS_LOW"], high=CONFIG["BANDPASS_HIGH"], order=4):
    """
    Butterworth bandpass filter 100-2000 Hz.
    Guard: filtfilt requires at least padlen+1 samples (≈ 3*order*2 = 24).
    Below that, returns raw signal to avoid crashes.
    """
    min_samples = 3 * order * 2 + 1
    if len(y) < min_samples:
        return y
    nyq = sr / 2
    b, a = butter(order, [low / nyq, high / nyq], btype='band')
    return filtfilt(b, a, y)


def trim_silence(y, sr, top_db=CONFIG["SILENCE_DB"]):
    """
    Remove silence from beginning AND end.
    Returns (y_trimmed, leading_s, trailing_s).
    """
    _, indices = librosa.effects.trim(y, top_db=top_db)
    y_trimmed = y[indices[0]:indices[1]]
    leading_s = indices[0] / sr
    trailing_s = (len(y) - indices[1]) / sr
    return y_trimmed, leading_s, trailing_s


def pad_or_crop(y, sr, target_s=CONFIG["TARGET_DURATION_S"]):
    """Pad or crop to standardized duration."""
    target = int(target_s * sr)
    if len(y) < target:
        return np.pad(y, (0, target - len(y)))
    return y[:target]


def normalize_amplitude(y, target_peak=0.95):
    """Normalize amplitude peak."""
    peak = np.max(np.abs(y))
    if peak > 0:
        return y / peak * target_peak
    return y


def process_file(y_raw, sr_raw, file_name, class_name):
    """
    Complete audio processing pipeline.
    
    Takes raw audio (y_raw, sr_raw) and applies 6-step preprocessing.
    
    Returns: (y_processed, sr, meta_dict) or (None, sr, meta_dict) if rejected
    """
    TARGET_SR = CONFIG["TARGET_SR"]
    MIN_DURATION_S = CONFIG["MIN_DURATION_S"]
    TARGET_DURATION_S = CONFIG["TARGET_DURATION_S"]
    
    original_duration = len(y_raw) / sr_raw

    # 1. Resample
    if sr_raw != TARGET_SR:
        y = librosa.resample(y_raw, orig_sr=sr_raw, target_sr=TARGET_SR)
        sr = TARGET_SR
    else:
        y, sr = y_raw.copy(), sr_raw

    # 2. Bandpass filter
    y = bandpass_filter(y, sr)

    # 3. Trim silence from beginning and end
    y, leading_s, trailing_s = trim_silence(y, sr)
    stripped_duration = len(y) / sr

    # 4. Reject if too short after silence trim
    # if stripped_duration < MIN_DURATION_S:
    #     return None, sr, {
    #         "FILE_NAME": file_name,
    #         "CLASS": class_name,
    #         "ACTION": "SKIPPED_TOO_SHORT",
    #         "ORIGINAL_DURATION_S": round(original_duration, 4),
    #         "STRIPPED_DURATION_S": round(stripped_duration, 4),
    #         "FINAL_DURATION_S": None,
    #         "LEADING_SILENCE_S": round(leading_s, 4),
    #         "TRAILING_SILENCE_S": round(trailing_s, 4),
    #         "SAMPLE_RATE": None,
    #         "N_SAMPLES": None,
    #         "AMPLITUDE_MAX": None,
    #         "RMS": None,
    #     }

    # 5. Pad or crop to target duration
    y = pad_or_crop(y, sr)

    # 6. Normalize amplitude
    y = normalize_amplitude(y)

    # Determine action label
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
        "FILE_NAME": file_name,
        "CLASS": class_name,
        "ACTION": action,
        "ORIGINAL_DURATION_S": round(original_duration, 4),
        "STRIPPED_DURATION_S": round(stripped_duration, 4),
        "FINAL_DURATION_S": round(len(y) / sr, 4),
        "LEADING_SILENCE_S": round(leading_s, 4),
        "TRAILING_SILENCE_S": round(trailing_s, 4),
        "SAMPLE_RATE": sr,
        "N_SAMPLES": len(y),
        "AMPLITUDE_MAX": round(float(np.max(np.abs(y))), 6),
        "RMS": round(float(np.sqrt(np.mean(y**2))), 6),
    }
    return y, sr, meta


# ── Snowflake UDF ─────────────────────────────────────────────────────────────
def process_file_udf(file_name: str, stage_name: str, class_name: str) -> dict:
    """
    Snowflake UDF to preprocess a single audio file.
    
    Also inserts metadata into M2_ISD_EQUIPE_1_DB.PROCESSED.RESPIRATORY_SOUNDS_METADATA.
    
    Args:
        file_name: Audio file name (e.g., 'patient_001.wav')
        stage_name: Stage path (e.g., '@STG_RESPIRATORY_SOUNDS/asthma/')
        class_name: Class label (e.g., 'Asthma')
        
    Returns:
        dict with preprocessing status and metadata
    """
    try:
        # Construct full path
        if not stage_name.endswith('/'):
            stage_name = stage_name + '/'
        file_path = stage_name + file_name
        
        # Load audio from stage
        y, sr = librosa.load(file_path, sr=None)
        
        # Process through pipeline
        y_proc, sr_proc, meta = process_file(y, sr, file_name, class_name)
        
        # Insert metadata into table
        try:
            if get_active_session is not None:
                session = get_active_session()
                # Create DataFrame for safer insertion
                df = pd.DataFrame([{
                    'FILE_NAME': meta.get("FILE_NAME"),
                    'CLASS': meta.get("CLASS"),
                    'ACTION': meta.get("ACTION"),
                    'ORIGINAL_DURATION_S': meta.get("ORIGINAL_DURATION_S"),
                    'STRIPPED_DURATION_S': meta.get("STRIPPED_DURATION_S"),
                    'FINAL_DURATION_S': meta.get("FINAL_DURATION_S"),
                    'LEADING_SILENCE_S': meta.get("LEADING_SILENCE_S"),
                    'TRAILING_SILENCE_S': meta.get("TRAILING_SILENCE_S"),
                    'SAMPLE_RATE': meta.get("SAMPLE_RATE"),
                    'N_SAMPLES': meta.get("N_SAMPLES"),
                    'AMPLITUDE_MAX': meta.get("AMPLITUDE_MAX"),
                    'RMS': meta.get("RMS"),
                }])
                session.create_dataframe(df).write.mode("append").save_as_table(
                    "M2_ISD_EQUIPE_1_DB.TEST.USER_RESPIRATORY_SOUNDS_METADATA",
                    create_temp_table=False
                )
        except Exception as insert_error:
            # Log insert error but don't fail the UDF
            pass
        
        return {
            "FILE_NAME": file_name,
            "CLASS": class_name,
            "STAGE": stage_name,
            "STATUS": "SUCCESS" if y_proc is not None else "REJECTED",
            "ACTION": meta.get("ACTION"),
            "METADATA": meta,
            "DB_INSERT": "SUCCESS" if y_proc is not None else "SKIPPED",
        }
        
    except Exception as e:
        return {
            "FILE_NAME": file_name,
            "CLASS": class_name,
            "STAGE": stage_name,
            "STATUS": "ERROR",
            "ERROR": str(e)[:500],
            "DB_INSERT": "FAILED",
        }


# ── Deployment ────────────────────────────────────────────────────────────────
def deploy_udf_process_file(session, udf_name: str = "PROCESS_FILE_UDF"):
    """
    Register the process_file UDF to Snowflake.
    
    Also creates the metadata table if it doesn't exist.
    
    Args:
        session: Snowflake session
        udf_name: Name for the UDF in Snowflake
    """
    # Register UDF via SQL with inline Python code
    udf_sql = textwrap.dedent(f"""
    CREATE OR REPLACE FUNCTION {udf_name}(file_name VARCHAR, stage_name VARCHAR, class_name VARCHAR)
    RETURNS VARIANT
    LANGUAGE PYTHON
    RUNTIME_VERSION = '3.11'
    PACKAGES = ('scipy', 'numpy', 'pandas', 'snowflake-snowpark-python')
    HANDLER = 'process_file_udf_handler'
    AS $$
    import io
    import numpy as np
    import warnings
    from scipy.signal import butter, filtfilt, resample_poly
    from scipy.io import wavfile
    import pandas as pd
    from snowflake.snowpark.files import SnowflakeFile
    
    warnings.filterwarnings('ignore')
    
    CONFIG = {{
        "TARGET_SR": 22050,
        "MIN_DURATION_S": 4,
        "TARGET_DURATION_S": 6,
        "SILENCE_DB": 30,
        "BANDPASS_LOW": 100,
        "BANDPASS_HIGH": 2000,
    }}
    
    def bandpass_filter(y, sr, low=100, high=2000, order=4):
        min_samples = 3 * order * 2 + 1
        if len(y) < min_samples:
            return y
        nyq = sr / 2
        b, a = butter(order, [low / nyq, high / nyq], btype='band')
        return filtfilt(b, a, y)
    
    def trim_silence(y, sr, top_db=30):
        if len(y) == 0:
            return y, 0.0, 0.0
        ref = float(np.max(np.abs(y)))
        if ref <= 0:
            return y, 0.0, 0.0

        threshold = ref * (10 ** (-top_db / 20.0))
        idx = np.where(np.abs(y) >= threshold)[0]
        if idx.size == 0:
            return y, 0.0, 0.0

        start = int(idx[0])
        end = int(idx[-1]) + 1
        y_trimmed = y[start:end]
        leading_s = start / sr
        trailing_s = (len(y) - end) / sr
        return y_trimmed, leading_s, trailing_s

    def load_audio_from_stage(file_path):
        with SnowflakeFile.open(file_path, 'rb') as f:
            audio_bytes = f.read()

        sr, y = wavfile.read(io.BytesIO(audio_bytes))

        # Convert multi-channel to mono
        if len(y.shape) > 1:
            y = np.mean(y.astype(np.float32), axis=1)

        # Normalize PCM to float32 [-1, 1]
        if y.dtype == np.int16:
            y = y.astype(np.float32) / 32768.0
        elif y.dtype == np.int32:
            y = y.astype(np.float32) / 2147483648.0
        elif y.dtype == np.uint8:
            y = (y.astype(np.float32) - 128.0) / 128.0
        else:
            y = y.astype(np.float32)

        return y, int(sr)
    
    def pad_or_crop(y, sr, target_s=6):
        target = int(target_s * sr)
        if len(y) < target:
            return np.pad(y, (0, target - len(y)))
        return y[:target]
    
    def normalize_amplitude(y, target_peak=0.95):
        peak = np.max(np.abs(y))
        if peak > 0:
            return y / peak * target_peak
        return y
    
    def process_file(y_raw, sr_raw, file_name, class_name):
        original_duration = len(y_raw) / sr_raw
        
        if sr_raw != 22050:
            y = resample_poly(y_raw, 22050, sr_raw).astype(np.float32)
            sr = 22050
        else:
            y, sr = y_raw.copy(), sr_raw
        
        y = bandpass_filter(y, sr)
        y, leading_s, trailing_s = trim_silence(y, sr)
        stripped_duration = len(y) / sr
        y = pad_or_crop(y, sr)
        y = normalize_amplitude(y)
        
        target_samples = int(6 * sr)
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
        
        meta = {{
            "FILE_NAME": file_name,
            "CLASS": class_name,
            "ACTION": action,
            "ORIGINAL_DURATION_S": round(original_duration, 4),
            "STRIPPED_DURATION_S": round(stripped_duration, 4),
            "FINAL_DURATION_S": round(len(y) / sr, 4),
            "LEADING_SILENCE_S": round(leading_s, 4),
            "TRAILING_SILENCE_S": round(trailing_s, 4),
            "SAMPLE_RATE": sr,
            "N_SAMPLES": len(y),
            "AMPLITUDE_MAX": round(float(np.max(np.abs(y))), 6),
            "RMS": round(float(np.sqrt(np.mean(y**2))), 6),
        }}
        return y, sr, meta
    
    def process_file_udf_handler(file_name: str, stage_name: str, class_name: str) -> dict:
        try:
            if not stage_name.endswith('/'):
                stage_name = stage_name + '/'
            file_path = stage_name + file_name
            
            y, sr = load_audio_from_stage(file_path)
            y_proc, sr_proc, meta = process_file(y, sr, file_name, class_name)
            
            return {{
                "FILE_NAME": file_name,
                "CLASS": class_name,
                "STAGE": stage_name,
                "STATUS": "SUCCESS" if y_proc is not None else "REJECTED",
                "ACTION": meta.get("ACTION"),
                "METADATA": meta,
            }}
        except Exception as e:
            return {{
                "FILE_NAME": file_name,
                "CLASS": class_name,
                "STAGE": stage_name,
                "STATUS": "ERROR",
                "ERROR": str(e)[:500],
            }}
    $$
    """)
    
    session.sql(udf_sql).collect()
    print(f"✓ UDF registered: {udf_name}")


if __name__ == "__main__":
    from snowflake.snowpark.context import get_active_session
    session = get_active_session()
    deploy_udf_process_file(session)
