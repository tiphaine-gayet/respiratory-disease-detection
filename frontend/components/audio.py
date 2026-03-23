import streamlit as st

def upload_audio():
    """Audio file uploader widget."""
    return st.file_uploader(
        "Upload respiratory sound",
        type=["wav", "mp3", "flac"],
        label_visibility="collapsed",
    )
