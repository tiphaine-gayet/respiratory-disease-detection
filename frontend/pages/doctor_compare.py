import streamlit as st
from components.doctor_styles import inject_doctor_css, doctor_header
from components.charts import waveform_chart_doc, mel_spectrogram_doc

# ── Reference data ──
REF_AUDIOS = {
    "Asthme": {
        "badge_cls": "ref-asthma",
        "label": "Asthme modéré — Homme 52 ans",
        "meta": "Réf. RESP-A-042 · 5.1s · Sibilances bilatérales",
        "probas": [("Asthme", 78, "asthma"), ("BPCO", 12, "copd"), ("Pneumonie", 5, "pneumo"), ("Bronchite", 3, "bronchi"), ("Sain", 2, "sain")],
        "similarity": 74,
    },
    "BPCO": {
        "badge_cls": "ref-copd",
        "label": "BPCO stade II — Femme 64 ans",
        "meta": "Réf. RESP-B-018 · 4.4s · Expiration prolongée",
        "probas": [("Asthme", 15, "asthma"), ("BPCO", 71, "copd"), ("Pneumonie", 8, "pneumo"), ("Bronchite", 4, "bronchi"), ("Sain", 2, "sain")],
        "similarity": 41,
    },
    "Pneumonie": {
        "badge_cls": "ref-pneumo",
        "label": "Pneumonie lobaire — Homme 38 ans",
        "meta": "Réf. RESP-P-009 · 4.9s · Crépitants fins",
        "probas": [("Asthme", 8, "asthma"), ("BPCO", 11, "copd"), ("Pneumonie", 68, "pneumo"), ("Bronchite", 9, "bronchi"), ("Sain", 4, "sain")],
        "similarity": 29,
    },
    "Sain": {
        "badge_cls": "ref-sain",
        "label": "Respiration normale — Femme 29 ans",
        "meta": "Réf. RESP-N-001 · 5.0s · Contrôle",
        "probas": [("Asthme", 5, "asthma"), ("BPCO", 4, "copd"), ("Pneumonie", 3, "pneumo"), ("Bronchite", 6, "bronchi"), ("Sain", 82, "sain")],
        "similarity": 18,
    },
}

PATIENT_PROBAS = [("Asthme", 62, "asthma"), ("BPCO", 18, "copd"), ("Pneumonie", 10, "pneumo"), ("Bronchite", 7, "bronchi"), ("Sain", 3, "sain")]

def _proba_html(probas, heading_color="#E8714A", heading_text="Patient"):
    rows = ""
    for name, pct, cls in probas:
        rows += f"""
        <div class="proba-row">
            <div class="proba-top">
                <span class="proba-name" style="font-size:12px;">{name}</span>
                <span class="proba-pct color-{cls}" style="font-size:12px;">{pct}%</span>
            </div>
            <div class="proba-bar-bg"><div class="proba-bar-fill fill-{cls}" style="width:{pct}%"></div></div>
        </div>"""
    return f"""
    <div class="compare-audio-box" style="padding:14px;">
        <div style="font-size:10px;font-weight:600;color:{heading_color};
                    text-transform:uppercase;letter-spacing:0.08em;margin-bottom:10px;">
            {heading_text}</div>
        <div class="proba-list" style="gap:8px;">{rows}</div>
    </div>"""

def render_compare():
    inject_doctor_css()
    doctor_header()

    st.markdown('<div class="doc-content">', unsafe_allow_html=True)

    st.markdown(
        '<p class="compare-intro">Comparez l\'audio du patient avec des enregistrements '
        'de référence clinique. Sélectionnez un audio de référence puis lancez la comparaison '
        "pour visualiser les différences spectrales et les probabilités côte à côte.</p>",
        unsafe_allow_html=True,
    )

    # ── Patient audio mini card ──
    st.markdown(
        """
        <div class="compare-audio-box" style="margin-bottom:16px;">
            <div class="compare-audio-tag tag-patient">Audio patient (chargé)</div>
            <div style="font-size:13px;font-weight:500;color:#333;margin-bottom:10px;font-family:var(--font);">
                Martin Rousseau · 4.8s · Reçu 14h23
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_w, col_m = st.columns(2)
    with col_w:
        st.pyplot(waveform_chart_doc(color="#E8714A"), use_container_width=True)
    with col_m:
        st.pyplot(mel_spectrogram_doc(), use_container_width=True)

    # ── Reference audio selection ──
    st.markdown(
        '<div style="font-size:13px;font-weight:500;color:#888;text-transform:uppercase;'
        'letter-spacing:0.08em;margin:20px 0 10px;font-family:var(--font);">'
        "Audios de référence clinique</div>",
        unsafe_allow_html=True,
    )

    # Build reference list as HTML + radio for selection
    for ref_name, ref_data in REF_AUDIOS.items():
        st.markdown(
            f"""
            <div class="ref-audio-item">
                <span class="ref-badge {ref_data['badge_cls']}">{ref_name}</span>
                <div style="flex:1;">
                    <div class="ref-name">{ref_data['label']}</div>
                    <div class="ref-meta">{ref_data['meta']}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    selected = st.selectbox(
        "Sélectionner une référence",
        list(REF_AUDIOS.keys()),
        index=0,
        label_visibility="collapsed",
    )

    # ── Compare button ──
    if st.button("🔀  Comparer les audios", use_container_width=True):
        st.session_state["compare_result"] = selected

    # ── Results ──
    if "compare_result" in st.session_state:
        ref = REF_AUDIOS[st.session_state["compare_result"]]

        st.markdown(
            f"""
            <div class="similarity-banner">
                Similarité spectrale avec la référence sélectionnée :
                <strong style="font-size:20px;font-family:var(--mono);">{ref['similarity']}%</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div style="font-size:13px;font-weight:500;color:#888;text-transform:uppercase;'
            'letter-spacing:0.08em;margin-bottom:10px;font-family:var(--font);">'
            "Visualisations comparées</div>",
            unsafe_allow_html=True,
        )

        # Side-by-side waveforms
        col_p, col_r = st.columns(2)
        with col_p:
            st.markdown(
                '<div class="compare-audio-box">'
                '<div class="compare-audio-tag tag-patient">Patient</div>'
                '<div style="font-size:12px;font-weight:500;color:#333;margin-bottom:8px;">Martin Rousseau</div>'
                "</div>",
                unsafe_allow_html=True,
            )
            st.pyplot(waveform_chart_doc(color="#E8714A"), use_container_width=True)
            st.pyplot(mel_spectrogram_doc(), use_container_width=True)

        with col_r:
            st.markdown(
                f'<div class="compare-audio-box">'
                f'<div class="compare-audio-tag tag-ref">Référence</div>'
                f'<div style="font-size:12px;font-weight:500;color:#333;margin-bottom:8px;">{ref["label"]}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )
            st.pyplot(waveform_chart_doc(color="#378ADD", seed=42), use_container_width=True)
            st.pyplot(mel_spectrogram_doc(seed=42), use_container_width=True)

        # Side-by-side probabilities
        st.markdown(
            '<div style="font-size:13px;font-weight:500;color:#888;text-transform:uppercase;'
            'letter-spacing:0.08em;margin:16px 0 10px;font-family:var(--font);">'
            "Probabilités comparées</div>",
            unsafe_allow_html=True,
        )

        col_pp, col_rp = st.columns(2)
        with col_pp:
            st.markdown(_proba_html(PATIENT_PROBAS, "#E8714A", "Patient"), unsafe_allow_html=True)
        with col_rp:
            st.markdown(_proba_html(ref["probas"], "#185FA5", "Référence"), unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
