import streamlit as st
import librosa
import numpy as np

def render_patient():
    st.markdown(_PATIENT_CSS, unsafe_allow_html=True)

    # Logo + tagline
    st.markdown(
        """
        <div class="p-logo-wrap">
            <div class="p-logo">TESS<span>AN</span></div>
            <div class="p-tagline">CABINET MÉDICAL CONNECTÉ · ANALYSE RESPIRATOIRE</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Centered card
    spacer_l, card_col, spacer_r = st.columns([1, 2, 1])

    with card_col:
        st.markdown(
            """
            <div class="p-card-header">
                <div class="p-title">Analyse de vos sons respiratoires</div>
                <div class="p-subtitle">
                    Déposez ou enregistrez votre respiration. Notre IA analyse en temps réel
                    et transmet un pré-diagnostic à votre médecin.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "Déposer un fichier audio",
            type=["wav", "mp3", "flac"],
            label_visibility="collapsed",
        )

        st.markdown('<div class="p-or">ou</div>', unsafe_allow_html=True)

        if st.button("🎙  Commencer l'enregistrement", use_container_width=True):
            st.session_state["recorded"] = True

        if uploaded_file:
            with st.spinner("Chargement…"):
                audio, sr = librosa.load(uploaded_file, sr=None)

            st.markdown('<div class="p-viz-label">FORME D\'ONDE</div>', unsafe_allow_html=True)
            from components.charts import waveform_chart, mel_spectrogram

            st.pyplot(waveform_chart(audio, sr), use_container_width=True)

            st.markdown('<div class="p-viz-label">MEL-SPECTROGRAMME</div>', unsafe_allow_html=True)
            st.pyplot(mel_spectrogram(audio, sr), use_container_width=True)

            if st.button("Analyser ma respiration →", use_container_width=True, key="analyze"):
                st.success("Analyse envoyée au médecin ✔️")

    # Step pills
    st.markdown(
        """
        <div class="p-steps-wrap">
            <div class="p-steps">
                <div class="p-step"><div class="p-step-num">01</div><div class="p-step-text">Déposez<br>votre audio</div></div>
                <div class="p-step"><div class="p-step-num">02</div><div class="p-step-text">Analyse<br>IA temps réel</div></div>
                <div class="p-step"><div class="p-step-num">03</div><div class="p-step-text">Médecin<br>reçoit le rapport</div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


_PATIENT_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Space+Mono:wght@400;700&display=swap');

.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main, section.main, section.main > div,
.block-container,
[data-testid="stVerticalBlock"],
[data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stHorizontalBlock"],
[data-testid="column"] {
    background-color: #2D3F5C !important;
}
.block-container { padding: 0 !important; max-width: 100% !important; }
header[data-testid="stHeader"] { display: none !important; }
footer { display: none !important; }
[data-testid="stVerticalBlock"] { gap: 0.75rem !important; }

.p-logo-wrap { text-align: center; padding: 48px 0 32px; background: #2D3F5C; }
.p-logo { font-family: 'Space Mono', monospace; font-size: 22px; font-weight: 700; color: white; letter-spacing: 0.15em; margin-bottom: 4px; }
.p-logo span { color: #E8714A; }
.p-tagline { font-size: 11px; color: rgba(255,255,255,0.4); letter-spacing: 0.12em; }

[data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(2) > div {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 20px !important;
    padding: 28px 32px 24px !important;
    max-width: 560px !important;
    margin: 0 auto !important;
}

.p-card-header { margin-bottom: 16px; }
.p-title { font-family: 'DM Sans', sans-serif; font-size: 18px; font-weight: 500; color: white; margin-bottom: 8px; }
.p-subtitle { font-family: 'DM Sans', sans-serif; font-size: 13px; color: rgba(255,255,255,0.45); line-height: 1.55; }

[data-testid="stFileUploader"] > section {
    border: 1.5px dashed rgba(255,255,255,0.2) !important;
    border-radius: 14px !important;
    background: rgba(0,0,0,0.15) !important;
    padding: 28px 20px !important;
}
[data-testid="stFileUploader"] > section:hover { border-color: #E8714A !important; background: rgba(232,113,74,0.06) !important; }
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] div { color: rgba(255,255,255,0.55) !important; }
[data-testid="stFileUploader"] button { color: rgba(255,255,255,0.7) !important; border-color: rgba(255,255,255,0.25) !important; background: transparent !important; }
[data-testid="stFileUploader"] button:hover { border-color: #E8714A !important; color: #E8714A !important; }
[data-testid="stFileUploader"] svg { color: rgba(255,255,255,0.4) !important; fill: rgba(255,255,255,0.4) !important; }

.p-or { display: flex; align-items: center; gap: 12px; color: rgba(255,255,255,0.25); font-size: 12px; margin: 8px 0; font-family: 'DM Sans', sans-serif; }
.p-or::before, .p-or::after { content: ''; flex: 1; height: 1px; background: rgba(255,255,255,0.12); }

.stButton > button {
    width: 100% !important; padding: 14px 24px !important;
    background: #E8714A !important; border: none !important;
    border-radius: 10px !important; color: white !important;
    font-family: 'DM Sans', sans-serif !important; font-size: 14px !important; font-weight: 500 !important;
}
.stButton > button:hover { background: #d45e37 !important; color: white !important; }
.stButton > button:active, .stButton > button:focus { background: #c0522e !important; color: white !important; border: none !important; box-shadow: none !important; }

.p-viz-label { font-size: 11px; color: rgba(255,255,255,0.4); letter-spacing: 0.1em; margin: 12px 0 6px; font-family: 'Space Mono', monospace; }

.p-steps-wrap { display: flex; justify-content: center; padding: 24px 32px 48px; background: #2D3F5C; }
.p-steps { display: flex; gap: 8px; max-width: 560px; width: 100%; }
.p-step { flex: 1; text-align: center; padding: 12px 8px; border-radius: 10px; background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.08); }
.p-step-num { font-size: 18px; font-weight: 600; color: #E8714A; font-family: 'Space Mono', monospace; }
.p-step-text { font-size: 10px; color: rgba(255,255,255,0.4); margin-top: 3px; line-height: 1.4; font-family: 'DM Sans', sans-serif; }
</style>
"""
