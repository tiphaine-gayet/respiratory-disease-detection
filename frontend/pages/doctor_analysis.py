import streamlit as st
import numpy as np
from components.doctor_styles import inject_doctor_css, doctor_header
from components.charts import radar_chart, waveform_chart_doc, mel_spectrogram_doc


def render_analysis():
    inject_doctor_css()
    doctor_header()

    st.markdown('<div class="doc-content">', unsafe_allow_html=True)

    # ── Patient Audio Card ──
    st.markdown(
        """
        <div class="patient-audio-card">
            <div class="doc-card-title">Patient · Audio reçu</div>
            <div class="patient-row">
                <div class="patient-avatar-doc">MR</div>
                <div class="patient-info-doc">
                    <div class="patient-name-doc">Martin Rousseau</div>
                    <div class="patient-meta-doc">48 ans · Reçu à 14h23 · Durée : 4.8s</div>
                </div>
                <span class="badge badge-warn">Priorité modérée</span>
            </div>
            <div class="audio-controls">
                <button class="play-btn">
                    <svg viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg>
                </button>
                <div class="audio-track"><div class="audio-progress"></div></div>
                <span class="audio-time">1:42 / 4:48</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Waveform + Mel (matplotlib, light theme for doctor)
    col_w, col_m = st.columns(2)
    with col_w:
        st.markdown(
            '<p style="font-size:10px;color:#888;font-family:var(--mono);margin-bottom:4px;">Forme d\'onde</p>',
            unsafe_allow_html=True,
        )
        st.pyplot(waveform_chart_doc(), use_container_width=True)
    with col_m:
        st.markdown(
            '<p style="font-size:10px;color:#888;font-family:var(--mono);margin-bottom:4px;">Mel-Spectrogramme</p>',
            unsafe_allow_html=True,
        )
        st.pyplot(mel_spectrogram_doc(), use_container_width=True)

    # ── Analysis grid: Probabilities + Confidence ──
    st.markdown('<div class="analysis-grid">', unsafe_allow_html=True)

    col_prob, col_conf = st.columns(2)

    with col_prob:
        probas = [
            ("Asthme", 62, "asthma"),
            ("BPCO", 18, "copd"),
            ("Pneumonie", 10, "pneumo"),
            ("Bronchite", 7, "bronchi"),
            ("Sain", 3, "sain"),
        ]
        rows_html = ""
        for name, pct, cls in probas:
            rows_html += f"""
            <div class="proba-row">
                <div class="proba-top">
                    <span class="proba-name">{name}</span>
                    <span class="proba-pct color-{cls}">{pct}%</span>
                </div>
                <div class="proba-bar-bg">
                    <div class="proba-bar-fill fill-{cls}" style="width:{pct}%"></div>
                </div>
            </div>"""

        st.markdown(
            f"""
            <div class="doc-card">
                <div class="doc-card-title">Probabilités par classe</div>
                <div class="proba-list">{rows_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_conf:
        st.markdown(
            """
            <div class="doc-card">
                <div class="doc-card-title">Niveau de confiance</div>
                <div class="confidence-display">
                    <div class="confidence-big">82%</div>
                    <div class="confidence-label">Score de confiance du modèle IA</div>
                    <div class="confidence-bar-wrap">
                        <div class="confidence-bar-bg"><div class="confidence-bar-fill"></div></div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        # Radar chart
        radar_data = {
            "Asthme": 62,
            "BPCO": 18,
            "Pneumonie": 10,
            "Bronchite": 7,
            "Sain": 3,
        }
        st.pyplot(radar_chart(radar_data), use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)  # close analysis-grid

    # ── Recommendation ──
    st.markdown(
        """
        <div class="recommendation-card">
            <div class="rec-header">
                <div class="rec-icon">⚠️</div>
                <div class="rec-title">Recommandation d'action</div>
            </div>
            <div class="rec-text">
                Le modèle détecte avec 62% de probabilité un <strong>profil asthmatique</strong>.
                Un suivi médical dans les <strong>48–72h</strong> est recommandé.
                En l'absence de symptômes aigus, une consultation de routine suffit.
                Si sibilances ou dyspnée aiguë : consultation urgente.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Dashboard metrics row ──
    st.markdown(
        """
        <div class="dashboard-row">
            <div class="dash-metric">
                <div class="dash-metric-label">Patients aujourd'hui</div>
                <div class="dash-metric-value">14</div>
                <div class="dash-metric-sub">+3 vs hier</div>
            </div>
            <div class="dash-metric">
                <div class="dash-metric-label">Cas urgents</div>
                <div class="dash-metric-value" style="color:#E24B4A;">2</div>
                <div class="dash-metric-sub">Consultation immédiate</div>
            </div>
            <div class="dash-metric">
                <div class="dash-metric-label">Confiance moy.</div>
                <div class="dash-metric-value">79%</div>
                <div class="dash-metric-sub">Toutes analyses</div>
            </div>
            <div class="dash-metric">
                <div class="dash-metric-label">Région — Île-de-France</div>
                <div class="dash-metric-value">34%</div>
                <div class="dash-metric-sub">Asthme ce mois</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)  # close doc-content
