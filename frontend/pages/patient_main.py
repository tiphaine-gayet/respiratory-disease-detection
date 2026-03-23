import streamlit as st
import numpy as np
from components.audio import load_audio, preprocess_audio
from components.charts import waveform_chart, mel_spectrogram


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

    # Wider centered card
    _, card_col, _ = st.columns([1, 3, 1])

    with card_col:
        st.markdown(
            """
            <div class="p-card-header">
                <div class="p-icon-row">
                    <div class="p-header-icon">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z"/><path d="M12 8v4l3 3"/></svg>
                    </div>
                    <span class="p-header-badge">Analyse IA</span>
                </div>
                <div class="p-title">Analyse respiratoire</div>
                <div class="p-subtitle">
                    Déposez ou enregistrez votre respiration. Notre IA analyse en temps réel
                    et transmet un pré-diagnostic à votre médecin.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── File upload / replace logic ──
        if "uploaded_audio" not in st.session_state:
            st.session_state["uploaded_audio"] = None
            st.session_state["audio_data"] = None
            st.session_state["audio_sr"] = None

        if st.session_state["uploaded_audio"] is None:
            # Show uploader
            uploaded_file = st.file_uploader(
                "Déposer un fichier audio",
                type=["wav", "mp3", "flac"],
                label_visibility="collapsed",
            )

            st.markdown('<div class="p-or">ou</div>', unsafe_allow_html=True)

            if st.button("🎙  Commencer l'enregistrement", use_container_width=True):
                st.session_state["recorded"] = True

            if uploaded_file:
                if not uploaded_file.name.endswith((".wav", ".mp3", ".flac")):
                    st.error("Format non supporté")
                    return
                uploaded_file.seek(0)
                with st.spinner("Chargement…"):
                    audio, sr = load_audio(uploaded_file)
                    audio, sr = preprocess_audio(audio, sr)
                st.session_state["uploaded_audio"] = uploaded_file
                st.session_state["audio_data"] = audio
                st.session_state["audio_sr"] = sr
                st.rerun()
        else:
            
            # ── Audio player + replace button ──
            st.session_state["uploaded_audio"].seek(0)
            st.audio(st.session_state["uploaded_audio"])
            st.markdown('<div class="replace-btn-wrap">', unsafe_allow_html=True)
            if st.button("Remplacer le fichier", key="replace_audio"):
                st.session_state["uploaded_audio"] = None
                st.session_state["audio_data"] = None
                st.session_state["audio_sr"] = None
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

            audio = st.session_state["audio_data"]
            sr = st.session_state["audio_sr"]

            # ── Forme d'onde ──
            st.markdown(
                '<div class="p-section-label">FORME D\'ONDE</div>',
                unsafe_allow_html=True,
            )
            st.pyplot(waveform_chart(audio, sr), use_container_width=True)

            # ── Mel-Spectrogramme ──
            st.markdown(
                '<div class="p-section-label">MEL-SPECTROGRAMME</div>',
                unsafe_allow_html=True,
            )
            st.pyplot(mel_spectrogram(audio, sr), use_container_width=True)

            # TODO: Replace fake data with model predictions
            # e.g. probas = model.predict(audio, sr)
            probas = [
                ("Asthme", 62, "asthma"),
                ("BPCO", 18, "copd"),
                ("Pneumonie", 10, "pneumo"),
                ("Bronchite", 7, "bronchi"),
                ("Sain", 3, "healthy"),
            ]

            rows_html = ""
            for name, pct, cls in probas:
                rows_html += f"""
                <div class="proba-row">
                    <div class="proba-top">
                        <span class="proba-name">{name}</span>
                        <span class="proba-pct proba-{cls}">{pct}%</span>
                    </div>
                    <div class="proba-bar-track">
                        <div class="proba-bar-fill bar-{cls}" style="width:{pct}%"></div>
                    </div>
                </div>
                """

            st.markdown(
                f"""
                <div class="p-result-card">
                    <div class="p-result-title">Probabilités par classe</div>
                    <div class="proba-list">
                        {rows_html}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            
            # ── Recommandation ──
            # TODO: Generate recommendation text dynamically from model output
            st.markdown(
                """
                <div class="rec-card">
                    <div class="rec-header">
                        <div class="rec-icon-wrap">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
                        </div>
                        <div class="rec-title">Recommandation d'action</div>
                    </div>
                    <div class="rec-body">
                        Le modèle détecte avec <strong>62%</strong> de probabilité un <strong>profil asthmatique</strong>.
                        Un suivi médical dans les <strong>48–72h</strong> est recommandé.
                        En l'absence de symptômes aigus, une consultation de routine suffit.
                        Si sibilances ou dyspnée aiguë : consultation urgente.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.button("Envoyer au médecin →", use_container_width=True, key="analyze"):
                st.success("Analyse envoyée au médecin ✔️")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CSS — Tessan brand system
#
# Brand palette (from tessan.io):
#   Deep navy   #1A2B4A  — primary background, trust & authority
#   Coral       #E8714A  — accent / CTA, warmth & approachability
#   Slate       #64748B  — secondary text
#   White       #FFFFFF  — headings on dark, card bg in doctor view
#   Off-white   #F7F8FA  — doctor-side page bg
#
# Typography:
#   Poppins     — headlines & body (matches Tessan's rounded modern sans)
#   JetBrains Mono — data values, labels, badges
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
_PATIENT_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Cormorant+Garamond:wght@400;500;600;700&display=swap');

:root {
    /* ── Tessan-inspired light palette from screenshots ── */
    --bg:           #F6F4EE;
    --bg-soft:      #F1EEE6;
    --card:         #FFFFFF;
    --card-soft:    #F7F5EF;
    --green:        #0C4B43;
    --green-2:      #0F5A50;
    --green-3:      #D8E3DB;
    --green-4:      #E6EEE8;
    --text-main:    #0C4B43;
    --text-soft:    #42675F;
    --text-muted:   #6F8C85;
    --border:       #D7E3DC;
    --yellow:       #E7E56A;
    --yellow-dark:  #D8D54F;
    --shadow:       0 8px 24px rgba(12, 75, 67, 0.05);

    /* disease colors kept for probability chart */
    --c-asthma:  #D95C4F;
    --c-copd:    #D8A63D;
    --c-pneumo:  #5B8DEF;
    --c-bronchi: #58A889;
    --c-healthy: #9AA8A5;

    --font-body: 'Inter', sans-serif;
    --font-title: 'Cormorant Garamond', serif;
}

/* ━━ Global Streamlit reset ━━ */
html, body, .stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main, section.main, section.main > div,
.block-container,
[data-testid="stVerticalBlock"],
[data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stHorizontalBlock"],
[data-testid="column"] {
    background: var(--bg) !important;
}

.block-container {
    padding: 0 0 40px 0 !important;
    max-width: 100% !important;
}

header[data-testid="stHeader"] { display: none !important; }
footer { display: none !important; }

[data-testid="stVerticalBlock"] {
    gap: 0.75rem !important;
}

/* ━━ Logo bar ━━ */
.p-logo-wrap {
    text-align: center;
    padding: 42px 0 28px;
    background: var(--bg);
}

.p-logo {
    font-family: var(--font-body);
    font-size: 24px;
    font-weight: 700;
    color: var(--green);
    letter-spacing: 0.18em;
    margin-bottom: 8px;
}

.p-logo span {
    color: var(--green);
}

.p-tagline {
    font-size: 10px;
    color: var(--text-muted);
    letter-spacing: 0.14em;
    font-family: var(--font-body);
    font-weight: 500;
    text-transform: uppercase;
}

/* ━━ Main card wrapper ━━ */
[data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(2) > div {
    background: var(--card-soft) !important;
    border: 1px solid var(--border) !important;
    border-radius: 28px !important;
    padding: 28px 30px 26px !important;
    max-width: 980px !important;
    margin: 0 auto !important;
    box-shadow: var(--shadow);
}

/* ━━ Header block ━━ */
.p-card-header {
    margin-bottom: 22px;
    padding: 26px 28px;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 22px;
}

.p-icon-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 14px;
}

.p-header-icon {
    width: 30px;
    height: 30px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--green);
}

.p-header-badge {
    font-family: var(--font-body);
    font-size: 11px;
    font-weight: 600;
    color: var(--green);
    background: #EEF4EF;
    padding: 5px 10px;
    border-radius: 999px;
    letter-spacing: 0.02em;
    border: 1px solid var(--border);
}

.p-title {
    font-family: var(--font-title);
    font-size: 44px;
    line-height: 0.95;
    font-weight: 500;
    color: var(--text-main);
    margin-bottom: 12px;
    letter-spacing: -0.02em;
}

.p-subtitle {
    font-family: var(--font-body);
    font-size: 15px;
    font-weight: 400;
    color: var(--text-soft);
    line-height: 1.7;
    max-width: 760px;
}

/* ━━ File uploader ━━ */
[data-testid="stFileUploader"] section div {
    visibility: hidden;
}

[data-testid="stFileUploader"] section::before {
    content: "Glissez-déposez votre fichier ici";
    display: block;
    text-align: center;
    font-size: 15px;
    color: var(--text-main);
    margin-bottom: 8px;
    font-family: var(--font-body);
    font-weight: 500;
}

[data-testid="stFileUploader"] section::after {
    content: "Formats acceptés : WAV, MP3, FLAC";
    display: block;
    text-align: center;
    font-size: 12px;
    color: var(--text-muted);
    font-family: var(--font-body);
}

[data-testid="stFileUploader"] > section {
    border: 1.5px dashed var(--border) !important;
    border-radius: 20px !important;
    background: var(--card) !important;
    padding: 30px 22px !important;
    transition: all 0.2s ease;
}

[data-testid="stFileUploader"] > section:hover {
    border-color: var(--green-2) !important;
    background: #FAFBF8 !important;
}

[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] div {
    color: var(--text-soft) !important;
}

[data-testid="stFileUploader"] button {
    color: var(--green) !important;
    border: 1px solid var(--border) !important;
    background: var(--card) !important;
    font-family: var(--font-body) !important;
    border-radius: 12px !important;
    font-weight: 500 !important;
}

[data-testid="stFileUploader"] button:hover {
    border-color: var(--green) !important;
    color: var(--green) !important;
    background: #F8FAF7 !important;
}

[data-testid="stFileUploader"] svg {
    color: var(--green) !important;
    fill: var(--green) !important;
}

/* ━━ Divider ━━ */
.p-or {
    display: flex;
    align-items: center;
    gap: 12px;
    color: var(--text-muted);
    font-size: 12px;
    margin: 10px 0;
    font-family: var(--font-body);
    font-weight: 500;
}

.p-or::before,
.p-or::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ━━ Primary button: Tessan-like pale yellow CTA ━━ */
.stButton > button {
    width: 100% !important;
    padding: 14px 24px !important;
    background: var(--yellow) !important;
    border: 1px solid transparent !important;
    border-radius: 10px !important;
    color: var(--green) !important;
    font-family: var(--font-body) !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    letter-spacing: 0 !important;
    box-shadow: none !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover {
    background: var(--yellow-dark) !important;
    color: var(--green) !important;
    border-color: transparent !important;
}

.stButton > button:active,
.stButton > button:focus {
    background: var(--yellow-dark) !important;
    color: var(--green) !important;
    border-color: transparent !important;
    box-shadow: none !important;
    outline: none !important;
}

/* ━━ Replace button ━━ */
/* ━━ Replace button as small text link under audio ━━ */
.replace-btn-wrap {
    margin-top: 6px;
    margin-bottom: 10px;
}

.replace-btn-wrap .stButton > button {
    width: auto !important;
    min-width: unset !important;
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    color: var(--green) !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    line-height: 1.2 !important;
    box-shadow: none !important;
    border-radius: 0 !important;
    text-decoration: underline;
    display: inline-block !important;
}

.replace-btn-wrap .stButton > button:hover,
.replace-btn-wrap .stButton > button:focus,
.replace-btn-wrap .stButton > button:active {
    background: transparent !important;
    border: none !important;
    color: var(--green-2) !important;
    box-shadow: none !important;
    transform: none !important;
    outline: none !important;
}

/* ━━ Audio player container feel cleaner in light mode ━━ */
[data-testid="stAudio"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
    padding: 6px !important;
}

/* ━━ Section labels ━━ */
.p-section-label {
    font-size: 11px;
    color: var(--text-muted);
    letter-spacing: 0.14em;
    margin: 22px 0 8px;
    font-family: var(--font-body);
    font-weight: 700;
}

/* ━━ Result cards ━━ */
.p-result-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 18px 20px;
    box-shadow: 0 4px 14px rgba(12, 75, 67, 0.03);
}

.p-result-title {
    font-size: 11px;
    font-weight: 700;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.10em;
    margin-bottom: 16px;
    font-family: var(--font-body);
}

/* ━━ Probability bars ━━ */
.proba-list {
    display: flex;
    flex-direction: column;
    gap: 14px;
}

.proba-top {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 6px;
}

.proba-name {
    font-size: 14px;
    font-weight: 500;
    color: var(--text-main);
    font-family: var(--font-body);
}

.proba-pct {
    font-size: 13px;
    font-weight: 600;
    font-family: var(--font-body);
}

.proba-bar-track {
    height: 7px;
    background: #E8EFEB;
    border-radius: 999px;
    overflow: hidden;
}

.proba-bar-fill {
    height: 7px;
    border-radius: 999px;
    transition: width 0.6s ease;
}

.proba-asthma  { color: var(--c-asthma);  } .bar-asthma  { background: var(--c-asthma);  }
.proba-copd    { color: var(--c-copd);    } .bar-copd    { background: var(--c-copd);    }
.proba-pneumo  { color: var(--c-pneumo);  } .bar-pneumo  { background: var(--c-pneumo);  }
.proba-bronchi { color: var(--c-bronchi); } .bar-bronchi { background: var(--c-bronchi); }
.proba-healthy { color: var(--c-healthy); } .bar-healthy { background: var(--c-healthy); }

/* ━━ Recommendation card ━━ */
.rec-card {
    background: #F3F7F4;
    border-radius: 20px;
    border: 1px solid var(--border);
    padding: 18px 20px;
    margin-top: 6px;
    margin-bottom: 16px;
}

.rec-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 12px;
}

.rec-icon-wrap {
    width: 34px;
    height: 34px;
    background: #E8F0EB;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--green);
}

.rec-title {
    font-size: 12px;
    font-weight: 700;
    color: var(--green);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-family: var(--font-body);
}

.rec-body {
    font-size: 14px;
    color: var(--text-soft);
    line-height: 1.7;
    font-family: var(--font-body);
    font-weight: 400;
}

.rec-body strong {
    color: var(--text-main);
    font-weight: 600;
}

/* ━━ Success alert ━━ */
[data-testid="stAlert"] {
    background: #EEF6F1 !important;
    border: 1px solid #CFE0D6 !important;
    border-radius: 14px !important;
    color: var(--green) !important;
    font-family: var(--font-body) !important;
}

/* ━━ Make matplotlib/chart blocks feel embedded in light UI ━━ */
[data-testid="stImage"],
[data-testid="stPlotlyChart"],
[data-testid="stMarkdownContainer"] canvas {
    border-radius: 18px !important;
}

/* ━━ Responsive tune ━━ */
@media (max-width: 900px) {
    .p-title {
        font-size: 34px;
    }

    [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-child(2) > div {
        max-width: 100% !important;
        padding: 20px !important;
        border-radius: 22px !important;
    }

    .p-card-header {
        padding: 20px;
    }
}
</style>
"""