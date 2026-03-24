import tempfile
import librosa

def load_audio(uploaded_file, target_sr=None):
    """
    Robust audio loader for Streamlit uploads.
    Works with WAV, MP3, FLAC (Snowflake / prod safe).
    """

    if uploaded_file is None:
        return None, None

    suffix = uploaded_file.name.split(".")[-1].lower()

    with tempfile.NamedTemporaryFile(suffix=f".{suffix}") as tmp:
        tmp.write(uploaded_file.read())
        tmp.flush()

        audio, sr = librosa.load(tmp.name, sr=target_sr)

    return audio, sr


def preprocess_audio(audio, sr, target_sr=22050):
    """
    Standardize audio for ML model.
    """

    if audio is None:
        return None, None

    # Resample
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    # Normalize
    audio = librosa.util.normalize(audio)

    return audio, sr