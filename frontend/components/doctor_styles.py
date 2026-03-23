import streamlit as st

DOCTOR_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Space+Mono:wght@400;700&display=swap');

:root {
    --tessan-navy: #2D3F5C;
    --tessan-coral: #E8714A;
    --font: 'DM Sans', sans-serif;
    --mono: 'Space Mono', monospace;
}

/* ── Page background ── */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main, section.main, section.main > div,
.block-container {
    background-color: #f8f8f8 !important;
}
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}
header[data-testid="stHeader"] { display: none !important; }
footer { display: none !important; }

/* ── Doctor header bar ── */
.doc-header {
    background: var(--tessan-navy);
    padding: 14px 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-radius: 0;
    margin-bottom: 0;
}
.doc-logo {
    font-family: var(--mono);
    font-size: 16px;
    font-weight: 700;
    color: white;
    letter-spacing: 0.1em;
}
.doc-logo span { color: var(--tessan-coral); }
.doc-user-info { display: flex; align-items: center; gap: 8px; }
.doc-avatar {
    width: 30px; height: 30px;
    background: var(--tessan-coral);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    color: white; font-size: 11px; font-weight: 600;
}
.doc-name { font-size: 12px; color: rgba(255,255,255,0.7); }

/* ── Content area ── */
.doc-content { padding: 20px 24px; }

/* ── Cards ── */
.doc-card {
    background: white;
    border-radius: 12px;
    border: 1px solid rgba(0,0,0,0.08);
    padding: 16px 18px;
    margin-bottom: 16px;
}
.doc-card-title {
    font-size: 11px;
    font-weight: 500;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 14px;
    font-family: var(--font);
}

/* ── Patient audio card ── */
.patient-audio-card {
    background: white;
    border-radius: 12px;
    border: 1px solid rgba(0,0,0,0.08);
    padding: 16px 18px;
    margin-bottom: 16px;
}
.patient-row { display: flex; align-items: center; gap: 14px; margin-bottom: 16px; }
.patient-avatar-doc {
    width: 40px; height: 40px;
    background: #e8f0fe; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-weight: 600; font-size: 13px; color: #185FA5;
    flex-shrink: 0;
}
.patient-info-doc { flex: 1; }
.patient-name-doc { font-size: 14px; font-weight: 500; color: #333; font-family: var(--font); }
.patient-meta-doc { font-size: 12px; color: #888; font-family: var(--font); }

/* ── Badges ── */
.badge { font-size: 11px; font-weight: 500; padding: 4px 10px; border-radius: 20px; }
.badge-warn { background: #FEF3C7; color: #92400E; }
.badge-ok   { background: #D1FAE5; color: #065F46; }
.badge-danger { background: #FEE2E2; color: #991B1B; }

/* ── Audio controls ── */
.audio-controls {
    display: flex; align-items: center; gap: 10px;
    background: #f8f8f8; border-radius: 8px;
    padding: 8px 12px; margin-bottom: 12px;
}
.play-btn {
    width: 30px; height: 30px;
    background: var(--tessan-navy);
    border-radius: 50%; border: none; cursor: pointer;
    display: flex; align-items: center; justify-content: center;
}
.play-btn svg { width: 12px; height: 12px; fill: white; }
.audio-track { flex: 1; height: 4px; background: rgba(0,0,0,0.1); border-radius: 2px; }
.audio-progress { height: 100%; width: 35%; background: var(--tessan-coral); border-radius: 2px; }
.audio-time { font-size: 11px; color: #888; font-family: var(--mono); }

/* ── Proba bars ── */
.proba-list { display: flex; flex-direction: column; gap: 10px; }
.proba-top { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 4px; }
.proba-name { font-size: 13px; font-weight: 500; color: #333; font-family: var(--font); }
.proba-pct  { font-size: 13px; font-weight: 500; font-family: var(--mono); }
.proba-bar-bg  { height: 6px; background: rgba(0,0,0,0.06); border-radius: 3px; }
.proba-bar-fill { height: 6px; border-radius: 3px; }

.color-asthma { color: #E24B4A; } .fill-asthma { background: #E24B4A; }
.color-copd   { color: #EF9F27; } .fill-copd   { background: #EF9F27; }
.color-pneumo { color: #378ADD; } .fill-pneumo { background: #378ADD; }
.color-bronchi{ color: #1D9E75; } .fill-bronchi{ background: #1D9E75; }
.color-sain   { color: #888780; } .fill-sain   { background: #888780; }

/* ── Confidence ── */
.confidence-display { text-align: center; padding: 10px 0; }
.confidence-big { font-size: 42px; font-weight: 300; color: var(--tessan-coral); font-family: var(--mono); line-height: 1; }
.confidence-label { font-size: 12px; color: #888; margin-top: 6px; font-family: var(--font); }
.confidence-bar-wrap { margin-top: 16px; }
.confidence-bar-bg  { height: 8px; background: rgba(0,0,0,0.06); border-radius: 4px; overflow: hidden; }
.confidence-bar-fill { height: 8px; width: 82%; background: var(--tessan-coral); border-radius: 4px; }

/* ── Recommendation ── */
.recommendation-card {
    background: white;
    border-radius: 12px;
    border: 2px solid var(--tessan-coral);
    padding: 16px 18px;
    margin-bottom: 16px;
}
.rec-header { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
.rec-icon {
    width: 32px; height: 32px;
    background: rgba(232,113,74,0.12); border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px;
}
.rec-title { font-size: 13px; font-weight: 600; color: var(--tessan-coral); text-transform: uppercase; letter-spacing: 0.06em; font-family: var(--font); }
.rec-text  { font-size: 13px; color: #333; line-height: 1.6; font-family: var(--font); }

/* ── Dashboard metrics ── */
.dashboard-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 18px; }
.dash-metric {
    background: white; border-radius: 10px;
    border: 1px solid rgba(0,0,0,0.08);
    padding: 14px 16px;
}
.dash-metric-label { font-size: 11px; color: #888; text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 6px; font-family: var(--font); }
.dash-metric-value { font-size: 22px; font-weight: 500; font-family: var(--mono); color: #333; }
.dash-metric-sub   { font-size: 11px; color: #888; margin-top: 2px; font-family: var(--font); }

/* ── Compare ── */
.compare-intro { font-size: 13px; color: #888; margin-bottom: 16px; line-height: 1.5; font-family: var(--font); }
.ref-audio-item {
    background: white; border-radius: 10px;
    border: 1px solid rgba(0,0,0,0.08);
    padding: 12px 14px;
    display: flex; align-items: center; gap: 12px;
    margin-bottom: 8px;
}
.ref-badge { padding: 4px 10px; border-radius: 20px; font-size: 11px; font-weight: 600; flex-shrink: 0; }
.ref-asthma { background: #FEE2E2; color: #991B1B; }
.ref-copd   { background: #FEF3C7; color: #92400E; }
.ref-pneumo { background: #DBEAFE; color: #1e40af; }
.ref-sain   { background: #D1FAE5; color: #065F46; }
.ref-name { flex: 1; font-size: 13px; font-weight: 500; color: #333; font-family: var(--font); }
.ref-meta { font-size: 11px; color: #888; font-family: var(--font); }

.similarity-banner {
    background: rgba(232,113,74,0.08);
    border: 1px solid rgba(232,113,74,0.3);
    border-radius: 10px;
    padding: 12px 16px;
    text-align: center;
    margin-bottom: 16px;
    font-size: 13px;
    color: var(--tessan-coral);
    font-weight: 500;
    font-family: var(--font);
}

.compare-audio-box {
    background: white; border-radius: 10px;
    border: 1px solid rgba(0,0,0,0.08);
    padding: 12px;
}
.compare-audio-tag { font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px; }
.tag-patient { color: var(--tessan-coral); }
.tag-ref     { color: #185FA5; }

/* ── Analysis grid ── */
.analysis-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }

/* ── Table ── */
.doc-table {
    width: 100%; font-size: 13px; border-collapse: collapse;
    font-family: var(--font);
}
.doc-table th {
    text-align: left; padding: 8px 0;
    color: #888; font-weight: 500; font-size: 11px;
    border-bottom: 1px solid rgba(0,0,0,0.08);
}
.doc-table td {
    padding: 9px 0;
    border-bottom: 1px solid rgba(0,0,0,0.04);
}
.doc-table td.name-col { font-weight: 500; }
.doc-table td.mono-col { font-family: var(--mono); font-size: 12px; }
.doc-table td.sub-col  { font-size: 12px; color: #888; }
.doc-table td.urgent-col { font-size: 12px; color: #E24B4A; font-weight: 500; }

/* ── Fix Streamlit buttons in doctor mode ── */
.stButton > button {
    font-family: var(--font) !important;
}
</style>
"""


def inject_doctor_css():
    st.markdown(DOCTOR_CSS, unsafe_allow_html=True)


def doctor_header():
    st.markdown(
        """
        <div class="doc-header">
            <div class="doc-logo">TESS<span>AN</span>
                <span style="font-size:11px;opacity:0.5;font-family:var(--font);font-weight:400;margin-left:6px;">— Espace Médecin</span>
            </div>
            <div class="doc-user-info">
                <div class="doc-avatar">DL</div>
                <div class="doc-name">Dr. Leclerc</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
