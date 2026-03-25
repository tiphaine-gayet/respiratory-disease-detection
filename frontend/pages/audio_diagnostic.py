import streamlit as st
import os
from components.audio import load_audio, preprocess_audio
from components.charts import waveform_chart, mel_spectrogram
from backend.router.predictions import load_pharmacies_for_select
from backend.router.ingestion import upload_patient_audio_with_metadata

# ── Path to reference audio files ──
REF_AUDIO_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "assets", "ref_audio")

# ── Reference data for inline comparison ──
REF_AUDIOS = {
    "Asthme": {
        "label": "Asthme modéré — Homme 52 ans",
        "meta": "Réf. RESP-A-042 · 5.1s · Sibilances bilatérales",
        "probas": [("Asthme", 78, "asthma"), ("BPCO", 12, "copd"), ("Pneumonie", 5, "pneumo"), ("Bronchite", 3, "bronchi"), ("Sain", 2, "healthy")],
        "similarity": 74,
        "audio_file": "asthma.wav",
    },
    "BPCO": {
        "label": "BPCO stade II — Femme 64 ans",
        "meta": "Réf. RESP-B-018 · 4.4s · Expiration prolongée",
        "probas": [("Asthme", 15, "asthma"), ("BPCO", 71, "copd"), ("Pneumonie", 8, "pneumo"), ("Bronchite", 4, "bronchi"), ("Sain", 2, "healthy")],
        "similarity": 41,
        "audio_file": "copd.wav",
    },
    "Pneumonie": {
        "label": "Pneumonie lobaire — Homme 38 ans",
        "meta": "Réf. RESP-P-009 · 4.9s · Crépitants fins",
        "probas": [("Asthme", 8, "asthma"), ("BPCO", 11, "copd"), ("Pneumonie", 68, "pneumo"), ("Bronchite", 9, "bronchi"), ("Sain", 4, "healthy")],
        "similarity": 29,
        "audio_file": "pneumonie.wav",
    },
    "Bronchite": {
        "label": "Bronchite aiguë — Femme 45 ans",
        "meta": "Réf. RESP-BR-007 · 4.6s · Ronchi diffus",
        "probas": [("Asthme", 10, "asthma"), ("BPCO", 9, "copd"), ("Pneumonie", 6, "pneumo"), ("Bronchite", 65, "bronchi"), ("Sain", 10, "healthy")],
        "similarity": 22,
        "audio_file": "bronchite.wav",
    },
    "Sain": {
        "label": "Respiration normale — Femme 29 ans",
        "meta": "Réf. RESP-N-001 · 5.0s · Contrôle",
        "probas": [("Asthme", 5, "asthma"), ("BPCO", 4, "copd"), ("Pneumonie", 3, "pneumo"), ("Bronchite", 6, "bronchi"), ("Sain", 82, "healthy")],
        "similarity": 18,
        "audio_file": "sain.wav",
    },
}

# Map class keys used in probas to REF_AUDIOS keys
_CLS_TO_REF = {
    "asthma": "Asthme",
    "copd": "BPCO",
    "pneumo": "Pneumonie",
    "bronchi": "Bronchite",
    "healthy": "Sain",
}

# Disease-class color map
_CLS_COLORS = {
    "asthma": "#D95C4F",
    "copd": "#D8A63D",
    "pneumo": "#5B8DEF",
    "bronchi": "#58A889",
    "healthy": "#9AA8A5",
}


@st.cache_data
def _load_ref_audio(audio_file):
    """Load and preprocess a reference audio file (cached)."""
    import librosa
    path = os.path.join(REF_AUDIO_DIR, audio_file)
    audio, sr_ = librosa.load(path, sr=22050)
    audio = librosa.util.normalize(audio)
    return audio, sr_


@st.cache_data
def _load_ref_audio_bytes(audio_file):
    """Return raw bytes of a reference audio file (cached)."""
    path = os.path.join(REF_AUDIO_DIR, audio_file)
    with open(path, "rb") as f:
        return f.read()


@st.cache_data(ttl=3600, show_spinner=False)
def _load_pharmacies_options():
    return load_pharmacies_for_select()


def render_diagnostic(is_doctor=False):
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

    # ── Pharmacy selection (used to attach pharmacy_id to the test context) ──
    pharmacies_df = _load_pharmacies_options()
    pharmacy_options = [""]
    pharmacy_labels = {"": "Sélectionner votre pharmacie"}
    if not pharmacies_df.empty:
        for _, row in pharmacies_df.iterrows():
            pid = row["pharmacie_id"]
            pharmacy_options.append(pid)
            pharmacy_labels[pid] = row["label"]

    if "selected_pharmacy_id" not in st.session_state:
        st.session_state["selected_pharmacy_id"] = ""

    st.selectbox(
        "Pharmacie du test",
        options=pharmacy_options,
        format_func=lambda pid: pharmacy_labels.get(pid, pid),
        key="selected_pharmacy_id",
        help="Sélectionnez la pharmacie où le test a été effectué.",
    )
    
    # DEBUG: Print pharmacy ID
    print(f"🔍 Pharmacy ID selected: {st.session_state.get('selected_pharmacy_id', 'NONE')}")

    st.markdown('<div class="p-content-wrap">', unsafe_allow_html=True)

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
        st.session_state["analysis_sent"] = False

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
                st.markdown("</div>", unsafe_allow_html=True)
                return
            uploaded_file.seek(0)
            with st.spinner("Chargement…"):
                audio, sr = load_audio(uploaded_file)
                audio, sr = preprocess_audio(audio, sr)
            st.session_state["uploaded_audio"] = uploaded_file
            st.session_state["audio_data"] = audio
            st.session_state["audio_sr"] = sr
            st.session_state["analysis_sent"] = False
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
            st.session_state["analysis_sent"] = False
            st.session_state["compare_class"] = None
            st.session_state["inference_result"] = None
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        audio = st.session_state["audio_data"]
        sr = st.session_state["audio_sr"]

        if not st.session_state.get("analysis_sent", False):
            if st.button("Envoyer →", use_container_width=True, key="send_analysis"):
                if not st.session_state.get("selected_pharmacy_id"):
                    st.warning("Veuillez sélectionner une pharmacie avant l'envoi.")
                    st.stop()

                patient_id = st.session_state.get("user_id", "").strip()
                if not patient_id:
                    st.error("Identifiant patient introuvable. Veuillez vous reconnecter.")
                    st.stop()

                uploaded_audio = st.session_state.get("uploaded_audio")
                if uploaded_audio is None:
                    st.error("Aucun fichier audio à envoyer.")
                    st.stop()

                uploaded_audio.seek(0)
                payload = uploaded_audio.read()
                if not payload:
                    st.error("Le fichier audio est vide.")
                    st.stop()

                stage_file_name = ""
                audio_metadata = {}
                try:
                    with st.spinner("Envoi de l'audio vers la plateforme..."):
                        stage_file_name, audio_metadata, inference_result = upload_patient_audio_with_metadata(
                            audio_bytes=payload,
                            audio=audio,
                            sr=sr,
                            patient_id=patient_id,
                            pharmacie_id=st.session_state.get("selected_pharmacy_id"),
                            original_filename=getattr(uploaded_audio, "name", None),
                        )
                except Exception as exc:
                    st.error(f"Impossible d'envoyer l'audio: {exc}")
                    st.stop()

                if not stage_file_name:
                    st.error("Le fichier audio n'a pas pu etre enregistre.")
                    st.stop()

                # Keep selected pharmacy id available for downstream persistence.
                st.session_state["analysis_pharmacie_id"] = st.session_state["selected_pharmacy_id"]
                st.session_state["analysis_audio_file_name"] = stage_file_name
                st.session_state["analysis_audio_metadata"] = audio_metadata
                st.session_state["inference_result"] = inference_result
                st.session_state["analysis_sent"] = True
                st.rerun()
        else:
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

            # Build probas from model predictions stored in session state
            _ir = st.session_state.get("inference_result") or {}
            probas = sorted([
                ("Asthme",    round(_ir.get("pct_asthma",    0)), "asthma"),
                ("BPCO",      round(_ir.get("pct_copd",      0)), "copd"),
                ("Pneumonie", round(_ir.get("pct_pneumonia", 0)), "pneumo"),
                ("Bronchite", round(_ir.get("pct_bronchial", 0)), "bronchi"),
                ("Sain",      round(_ir.get("pct_healthy",   0)), "healthy"),
            ], key=lambda x: x[1], reverse=True)

            # ── Probability card ──
            if "compare_class" not in st.session_state:
                st.session_state["compare_class"] = None

            rows_html = ""
            for i, (name, pct, cls) in enumerate(probas):
                is_last = i == len(probas) - 1
                border_bottom = "none" if is_last else "1px solid #D7E3DC"

                rows_html += f"""
                <div style="display:flex;align-items:center;gap:12px;padding:14px 20px;
                            border-bottom:{border_bottom};background:#fff;">
                    <div style="flex:1;">
                        <div style="display:flex;justify-content:space-between;margin-bottom:8px;">
                            <span style="font-size:14px;font-weight:500;color:#0C4B43;
                                         font-family:Inter,sans-serif;">{name}</span>
                            <span style="font-size:13px;font-weight:600;color:{_CLS_COLORS[cls]};
                                         font-family:Inter,sans-serif;">{pct}%</span>
                        </div>
                        <div style="height:7px;background:#E8EFEB;border-radius:999px;overflow:hidden;">
                            <div style="height:7px;width:{pct}%;background:{_CLS_COLORS[cls]};
                                        border-radius:999px;"></div>
                        </div>
                    </div>
                </div>
                """

            st.markdown(
                f"""
                <div class="proba-card">
                    <div class="proba-card-header">
                        <span class="proba-card-title">Probabilités par classe</span>
                    </div>
                    {rows_html}
                </div>
                """,
                unsafe_allow_html=True,
            )         

            # ── Compare dropdown ──
            compare_options = ["—"] + [name for name, _, _ in probas]
            cls_keys = [None] + [cls for _, _, cls in probas]
            name_to_cls = {name: cls for name, _, cls in probas}

            current_cls = st.session_state.get("compare_class")
            current_name = "—"
            for name, _, cls in probas:
                if cls == current_cls:
                    current_name = name
                    break
                
            current_idx = compare_options.index(current_name)

            def _on_compare_change():
                sel = st.session_state.get("cmp_select", "—")
                st.session_state["compare_class"] = name_to_cls.get(sel, None)

            st.markdown(
                '<div class="cmp-section-title">Comparer à un audio de référence</div>',
                unsafe_allow_html=True,
            )
            st.markdown('<div class="cmp-select-wrap">', unsafe_allow_html=True)
            st.selectbox(
                "Comparer avec une référence",
                options=compare_options,
                index=current_idx,
                key="cmp_select",
                label_visibility="collapsed",
                placeholder="⇄  Sélectionner une classe…",
                on_change=_on_compare_change,
            )
            st.markdown('</div>', unsafe_allow_html=True)

            # ── Inline comparison panel ──
            if st.session_state["compare_class"] is not None:
                active_cls = st.session_state["compare_class"]
                ref_key = _CLS_TO_REF[active_cls]
                ref = REF_AUDIOS[ref_key]
                cls_color = _CLS_COLORS[active_cls]

                st.markdown(
                    f"""
                    <div class="compare-panel" style="border-color: {cls_color};">
                        <div class="compare-panel-header" style="background: {cls_color}12;">
                            <div>
                                <span class="compare-panel-tag" style="background:{cls_color}; color:#fff;">
                                    Comparaison — {ref_key}
                                </span>
                                <span class="compare-panel-meta">{ref['meta']}</span>
                            </div>
                            <span class="compare-panel-sim">
                                Similarité spectrale
                                <strong>{ref['similarity']}%</strong>
                            </span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                ref_audio, ref_sr = _load_ref_audio(ref["audio_file"])

                # ── Side-by-side audio players ──
                st.markdown(
                    '<div class="compare-section-label">ÉCOUTE</div>',
                    unsafe_allow_html=True,
                )
                ca_p, ca_r = st.columns(2)
                with ca_p:
                    st.markdown(
                        '<div class="compare-col-tag tag-patient">Patient</div>',
                        unsafe_allow_html=True,
                    )
                    st.session_state["uploaded_audio"].seek(0)
                    st.audio(st.session_state["uploaded_audio"])
                with ca_r:
                    st.markdown(
                        '<div class="compare-col-tag tag-ref">Référence</div>',
                        unsafe_allow_html=True,
                    )
                    st.audio(_load_ref_audio_bytes(ref["audio_file"]))

                # ── Side-by-side waveforms ──
                st.markdown(
                    '<div class="compare-section-label">FORME D\'ONDE</div>',
                    unsafe_allow_html=True,
                )
                cw_p, cw_r = st.columns(2)
                with cw_p:
                    st.markdown(
                        '<div class="compare-col-tag tag-patient">Patient</div>',
                        unsafe_allow_html=True,
                    )
                    st.pyplot(waveform_chart(audio, sr), use_container_width=True)
                with cw_r:
                    st.markdown(
                        '<div class="compare-col-tag tag-ref">Référence</div>',
                        unsafe_allow_html=True,
                    )
                    st.pyplot(waveform_chart(ref_audio, ref_sr), use_container_width=True)

                # ── Side-by-side mel spectrograms ──
                st.markdown(
                    '<div class="compare-section-label">MEL-SPECTROGRAMME</div>',
                    unsafe_allow_html=True,
                )
                cm_p, cm_r = st.columns(2)
                with cm_p:
                    st.markdown(
                        '<div class="compare-col-tag tag-patient">Patient</div>',
                        unsafe_allow_html=True,
                    )
                    st.pyplot(mel_spectrogram(audio, sr), use_container_width=True)
                with cm_r:
                    st.markdown(
                        '<div class="compare-col-tag tag-ref">Référence</div>',
                        unsafe_allow_html=True,
                    )
                    st.pyplot(mel_spectrogram(ref_audio, ref_sr), use_container_width=True)

                # Close comparison
                st.markdown('<div class="compare-close-wrap">', unsafe_allow_html=True)
                if st.button("✕  Fermer la comparaison", key="close_cmp", use_container_width=True):
                    st.session_state["compare_class"] = None
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)

            _rec_text = (_ir.get("detailed_action") or "Résultat en attente.").replace("\n", "<br>")
            st.markdown(
                f"""
                <div class="rec-card">
                    <div class="rec-header">
                        <div class="rec-icon-wrap">
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
                        </div>
                        <div class="rec-title">Recommandation d'action</div>
                    </div>
                    <div class="rec-body">{_rec_text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)
    


_PATIENT_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Cormorant+Garamond:wght@400;500;600;700&display=swap');

:root {
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

    --c-asthma:  #D95C4F;
    --c-copd:    #D8A63D;
    --c-pneumo:  #5B8DEF;
    --c-bronchi: #58A889;
    --c-healthy: #9AA8A5;

    --font-body: 'Inter', sans-serif;
    --font-title: 'Cormorant Garamond', serif;
}

/* ━━ Global reset ━━ */
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
    padding: 40px 20px !important;
    max-width: 800px !important;
    margin: 0 auto !important;
}

header[data-testid="stHeader"] { display: none !important; }
footer { display: none !important; }

[data-testid="stVerticalBlock"] {
    gap: 0.75rem !important;
}

/* ━━ Logo ━━ */
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

/* ━━ Main content wrapper ━━ */
.p-content-wrap {
    width: 100%;
    margin: 0 auto;
    display: flex;
    flex-direction: column;
    gap: 16px;
}

/* ━━ Header card ━━ */
.p-card-header {
    margin-bottom: 22px;
    padding: 26px 28px;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 22px;
    box-shadow: 0 4px 14px rgba(12, 75, 67, 0.03);
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

/* ━━ Default buttons ━━ */
.stButton > button {
    width: 100% !important;
    padding: 14px 24px !important;
    background: white !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--green) !important;
    font-family: var(--font-body) !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    box-shadow: none !important;
    transition: all 0.2s ease !important;
}

.stButton > button:hover,
.stButton > button:focus,
.stButton > button:active {
    background: var(--green) !important;
    color: white !important;
    border-color: var(--green) !important;
    box-shadow: none !important;
    outline: none !important;
}

/* ━━ Replace button ━━ */
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

/* ━━ Audio player ━━ */
[data-testid="stAudio"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
    padding: 6px !important;
}

/* ━━ Section labels ━━ */
.p-section-label,
.compare-section-label {
    font-size: 11px;
    color: var(--text-muted);
    letter-spacing: 0.14em;
    margin: 22px 0 8px;
    font-family: var(--font-body);
    font-weight: 700;
}

/* ━━━━ Probability card (single markdown block) ━━━━ */
.proba-card {
    background: #fff;
    border: 1px solid var(--border);
    border-radius: 20px 20px 20px 20px;
    overflow: hidden;
    box-shadow: 0 4px 14px rgba(12, 75, 67, 0.03);
}

.proba-card-header {
    padding: 16px 20px 12px;
    border-bottom: 1px solid var(--border);
}

.proba-card-title {
    font-size: 11px;
    font-weight: 700;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.10em;
    font-family: var(--font-body);
}

/* ━━ Compare section title ━━ */
.cmp-section-title {
    font-family: var(--font-body);
    font-size: 14px;
    font-weight: 600;
    color: var(--text-main);
    margin-top: 20px;
    margin-bottom: 6px;
}

/* ━━ Compare selectbox ━━ */
.cmp-select-wrap {
    margin-top: -4px;
    margin-bottom: 10px;
}

.cmp-select-wrap [data-testid="stSelectbox"] > div > div {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    font-family: var(--font-body) !important;
    font-size: 13px !important;
    color: var(--text-main) !important;
    padding: 2px 4px !important;
    cursor: pointer !important;
    transition: border-color 0.2s ease !important;
}

.cmp-select-wrap [data-testid="stSelectbox"] > div > div:hover {
    border-color: var(--green) !important;
}

.cmp-select-wrap [data-testid="stSelectbox"] svg {
    color: var(--green) !important;
}

/* ━━ Comparison panel ━━ */
.compare-panel {
    border: 2px solid var(--border);
    border-radius: 18px;
    overflow: hidden;
    margin-top: 8px;
    margin-bottom: 8px;
    background: var(--card);
}

.compare-panel-header {
    padding: 14px 18px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
}

.compare-panel-tag {
    font-size: 11px;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 8px;
    font-family: var(--font-body);
    letter-spacing: 0.02em;
}

.compare-panel-meta {
    font-size: 11px;
    color: var(--text-muted);
    margin-left: 10px;
    font-family: var(--font-body);
}

.compare-panel-sim {
    font-size: 12px;
    color: var(--text-soft);
    font-family: var(--font-body);
    font-weight: 400;
}

.compare-panel-sim strong {
    font-size: 18px;
    font-weight: 600;
    color: var(--text-main);
    margin-left: 6px;
    font-family: var(--font-body);
}

/* ━━ Compare labels ━━ */
.compare-col-tag {
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 3px 10px;
    border-radius: 6px;
    display: inline-block;
    margin-bottom: 6px;
    font-family: var(--font-body);
}

.tag-patient {
    background: #FDF0EC;
    color: #D95C4F;
}

.tag-ref {
    background: #EAF1FB;
    color: #3B6DC2;
}

/* ━━ Compare probability cards ━━ */
.cmp-proba-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 12px 14px;
    display: flex;
    flex-direction: column;
    gap: 10px;
}

.cmp-proba-row {
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    gap: 6px;
}

.cmp-proba-name {
    font-size: 12px;
    font-weight: 500;
    color: var(--text-main);
    font-family: var(--font-body);
    flex: 1;
}

.cmp-proba-pct {
    font-size: 12px;
    font-weight: 600;
    font-family: var(--font-body);
}

.cmp-proba-row .proba-bar-track {
    width: 100%;
    height: 5px;
}

.cmp-proba-row .proba-bar-fill {
    height: 5px;
}

/* Color classes for proba bars */
.proba-asthma, .bar-asthma { color: var(--c-asthma); background: var(--c-asthma); }
.proba-copd, .bar-copd { color: var(--c-copd); background: var(--c-copd); }
.proba-pneumo, .bar-pneumo { color: var(--c-pneumo); background: var(--c-pneumo); }
.proba-bronchi, .bar-bronchi { color: var(--c-bronchi); background: var(--c-bronchi); }
.proba-healthy, .bar-healthy { color: var(--c-healthy); background: var(--c-healthy); }

/* ━━ Close comparison button ━━ */
.compare-close-wrap {
    margin-top: 4px;
    margin-bottom: 8px;
}

.compare-close-wrap .stButton > button {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-muted) !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    padding: 8px 16px !important;
    border-radius: 10px !important;
}

.compare-close-wrap .stButton > button:hover,
.compare-close-wrap .stButton > button:focus,
.compare-close-wrap .stButton > button:active {
    background: #FDF0EC !important;
    border-color: var(--c-asthma) !important;
    color: var(--c-asthma) !important;
    box-shadow: none !important;
    outline: none !important;
}

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

/* ━━ Charts ━━ */
[data-testid="stImage"],
[data-testid="stPlotlyChart"],
[data-testid="stMarkdownContainer"] canvas {
    border-radius: 18px !important;
}

/* ━━ Responsive ━━ */
@media (max-width: 900px) {
    .p-title {
        font-size: 34px;
    }

    .p-content-wrap {
        padding: 0 16px 30px;
    }

    .p-card-header {
        padding: 20px;
    }
}
</style>
"""