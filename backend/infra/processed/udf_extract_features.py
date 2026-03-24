"""
Snowflake UDF for audio feature extraction
===========================================

This module provides a Snowflake-deployable UDF for audio feature extraction:
1. Takes processed audio signal (y, sr)
2. Extracts 6 audio features (mel, mfcc, chroma, centroid, bandwidth, zcr)
3. Returns features as VARIANT or saves to stage

Features extracted:
  - Mel-spectrogram (128, 9)
  - MFCC (13, 9)
  - Chroma-STFT (12, 9)
  - Spectral centroid (1, 9)
  - Spectral bandwidth (1, 9)
  - Zero-crossing rate (1, 9)

Usage (in deployment script):
    from snowflake.snowpark.context import get_active_session
    from udf_extract_features import deploy_udf_extract_features
    session = get_active_session()
    deploy_udf_extract_features(session)
"""

import sys
import os
import io
import zipfile
import json
import base64
import textwrap

# Import Snowpark types
from snowflake.snowpark.types import StringType, VariantType, BooleanType

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
import warnings

warnings.filterwarnings('ignore')


# ── Configuration ─────────────────────────────────────────────────────────────
CONFIG = {
    "TARGET_SR": 22050,
    "N_MELS": 128,
    "N_FFT": 2048,
    "HOP_LENGTH": 512,
    "N_MFCC": 13,
}


# ── Feature Extraction ─────────────────────────────────────────────────────────
def extract_features(y, sr, file_name=None, class_name=None):
    """
    Extract 6 audio features from processed signal.
    
    Args:
        y: Audio signal (numpy array)
        sr: Sample rate
        file_name: Original file name (optional, for logging)
        class_name: Class label (optional, for logging)
        
    Returns:
        dict with 6 feature types and their data
    """
    n_mels = CONFIG["N_MELS"]
    n_fft = CONFIG["N_FFT"]
    hop_length = CONFIG["HOP_LENGTH"]
    n_mfcc = CONFIG["N_MFCC"]
    
    features = {
        "mel": librosa.power_to_db(
            librosa.feature.melspectrogram(
                y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length),
            ref=np.max),
        "mfcc": librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc),
        "chroma": librosa.feature.chroma_stft(y=y, sr=sr),
        "centroid": librosa.feature.spectral_centroid(y=y, sr=sr),
        "bandwidth": librosa.feature.spectral_bandwidth(y=y, sr=sr),
        "zcr": librosa.feature.zero_crossing_rate(y),
    }
    
    return features


def features_to_variant(features, file_name=None, class_name=None):
    """
    Convert features dict to VARIANT-compatible format.
    
    Returns dict with feature metadata and serialized arrays.
    """
    variant = {
        "FILE_NAME": file_name,
        "CLASS": class_name,
        "FEATURES": {}
    }
    
    for feat_type, feat_data in features.items():
        variant["FEATURES"][feat_type] = {
            "shape": list(feat_data.shape),
            "dtype": str(feat_data.dtype),
            "data": feat_data.tolist(),
        }
    
    return variant


# ── Snowflake UDF ─────────────────────────────────────────────────────────────
def extract_features_udf(
    processed_audio_path: str,
    stage_name: str,
    file_name: str,
    class_name: str,
    save_to_stage: bool = False,
    output_stage: str = "@M2_ISD_EQUIPE_1_DB.TEST.STG_RESPIRATORY_FEATURES"
) -> dict:
    """
    Snowflake UDF to extract audio features.
    
    Args:
        processed_audio_path: Path to processed audio file in stage
        stage_name: Stage name/path prefix
        file_name: Audio file name
        class_name: Class label
        save_to_stage: Whether to save features as .npy files
        output_stage: Stage path to save features (if save_to_stage=True)
        
    Returns:
        dict with feature data and metadata
    """
    try:
        # Construct full path
        if not stage_name.endswith('/'):
            stage_name = stage_name + '/'
        file_path = stage_name + processed_audio_path
        
        # Load processed audio
        y, sr = librosa.load(file_path, sr=None)
        
        # Extract features
        features = extract_features(y, sr, file_name, class_name)
        
        # Convert to VARIANT format
        result = {
            "FILE_NAME": file_name,
            "CLASS": class_name,
            "STAGE": stage_name,
            "STATUS": "SUCCESS",
            "FEATURES": {}
        }
        
        # Build feature metadata
        for feat_type, feat_data in features.items():
            result["FEATURES"][feat_type] = {
                "shape": list(feat_data.shape),
                "dtype": str(feat_data.dtype),
                "n_frames": feat_data.shape[1] if len(feat_data.shape) > 1 else 1,
                "n_coefficients": feat_data.shape[0] if len(feat_data.shape) > 0 else 1,
            }
        
        # Optional: Save to stage as .npy files
        if save_to_stage and output_stage:
            import tempfile
            with tempfile.TemporaryDirectory() as tmp:
                saved_files = {}
                for feat_type, feat_data in features.items():
                    file_stem = os.path.splitext(file_name)[0]
                    npy_name = f"{file_stem}_{feat_type}.npy"
                    npy_path = os.path.join(tmp, npy_name)
                    
                    # Save locally
                    np.save(npy_path, feat_data)
                    saved_files[feat_type] = npy_name
                
                # Upload to stage
                if not output_stage.endswith('/'):
                    output_stage = output_stage + '/'
                
                for feat_type, npy_name in saved_files.items():
                    # This would require session object, handled in deployment
                    result["FEATURES"][feat_type]["NPY_FILENAME"] = npy_name
        
        return result
        
    except Exception as e:
        return {
            "FILE_NAME": file_name,
            "CLASS": class_name,
            "STATUS": "ERROR",
            "ERROR": str(e)[:500],
        }

# ── Deployment ────────────────────────────────────────────────────────────────
def deploy_udf_extract_features(session, udf_name: str = "EXTRACT_FEATURES_UDF"):
    """
    Register the extract_features UDF to Snowflake.
    
    Args:
        session: Snowflake session
        udf_name: Name for the UDF in Snowflake
    """
    # Register UDF via SQL with inline Python code
    udf_sql = textwrap.dedent(f"""
    CREATE OR REPLACE FUNCTION {udf_name}(
        processed_audio_path VARCHAR,
        stage_name VARCHAR,
        file_name VARCHAR,
        class_name VARCHAR,
        save_to_stage BOOLEAN,
        output_stage VARCHAR
    )
    RETURNS VARIANT
    LANGUAGE PYTHON
    RUNTIME_VERSION = '3.11'
    PACKAGES = ('scipy', 'numpy')
    IMPORTS = ('@M2_ISD_EQUIPE_1_DB.PUBLIC.STG_LIBRARIES/libs/libroza.zip')
    HANDLER = 'extract_features_handler'
    AS $$
    import sys
    import os
    import io
    import zipfile
    import numpy as np
    
    # Setup librosa from imported zip
    def _setup_librosa():
        import sys, os, io, zipfile

        # Récupère le répertoire temporaire que Snowflake monte pour les imports
        import_dir = sys._xoptions.get("snowflake_import_directory")
        final_lib_dir = "/tmp/site-packages"
        os.makedirs(final_lib_dir, exist_ok=True)

        # Chemin du zip (doit correspondre EXACTEMENT à celui déclaré dans IMPORTS)
        zip_path = os.path.join(import_dir, "libroza.zip")

        try:
            # Ouvre le zip importé depuis le stage
            with zipfile.ZipFile(zip_path, 'r') as outer:
                for name in outer.namelist():
                    # Chaque .whl à l'intérieur est un package Python
                    if name.endswith(".whl"):
                        whl_bytes = outer.read(name)
                        try:
                            with zipfile.ZipFile(io.BytesIO(whl_bytes), 'r') as whl:
                                whl.extractall(final_lib_dir)
                        except FileExistsError:
                            # Si un dossier existe déjà (e.g. librosa/core), on ignore
                            pass
        except FileNotFoundError:
            return f"❌ Zip non trouvé"

        # Ajoute /tmp/site-packages dans sys.path si absent
        if final_lib_dir not in sys.path:
            sys.path.insert(0, final_lib_dir)

        # Teste l'import effectif de librosa
        try:
            import librosa
            return "✅ Librosa importée — version: " + librosa.__version__
        except Exception as e:
            return f"❌ Erreur import librosa"
    
    _setup_librosa()
    import librosa
    
    def extract_features(y, sr):
        n_mels = 128
        n_fft = 2048
        hop_length = 512
        n_mfcc = 13
        
        features = {{
            "mel": librosa.power_to_db(
                librosa.feature.melspectrogram(
                    y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length),
                ref=np.max),
            "mfcc": librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc),
            "chroma": librosa.feature.chroma_stft(y=y, sr=sr),
            "centroid": librosa.feature.spectral_centroid(y=y, sr=sr),
            "bandwidth": librosa.feature.spectral_bandwidth(y=y, sr=sr),
            "zcr": librosa.feature.zero_crossing_rate(y),
        }}
        
        return features
    
    def extract_features_handler(
        processed_audio_path: str,
        stage_name: str,
        file_name: str,
        class_name: str,
        save_to_stage: bool,
        output_stage: str = "@M2_ISD_EQUIPE_1_DB.TEST.STG_RESPIRATORY_FEATURES"
    ) -> dict:
        try:
            if not stage_name.endswith('/'):
                stage_name = stage_name + '/'
            file_path = stage_name + processed_audio_path
            
            y, sr = librosa.load(file_path, sr=None)
            features = extract_features(y, sr)
            
            result = {{
                "FILE_NAME": file_name,
                "CLASS": class_name,
                "STAGE": stage_name,
                "STATUS": "SUCCESS",
                "FEATURES": {{}}
            }}
            
            for feat_type, feat_data in features.items():
                result["FEATURES"][feat_type] = {{
                    "shape": list(feat_data.shape),
                    "dtype": str(feat_data.dtype),
                    "n_frames": feat_data.shape[1] if len(feat_data.shape) > 1 else 1,
                    "n_coefficients": feat_data.shape[0] if len(feat_data.shape) > 0 else 1,
                }}
            
            return result
            
        except Exception as e:
            return {{
                "FILE_NAME": file_name,
                "CLASS": class_name,
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
    deploy_udf_extract_features(session)
