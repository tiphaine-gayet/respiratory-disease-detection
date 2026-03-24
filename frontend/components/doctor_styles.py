import streamlit as st

DOCTOR_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Cormorant+Garamond:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    /* ── Shared app.py / patient tokens ── */
    --bg:           #F6F4EE;
    --bg-soft:      #F1EEE6;
    --card:         #FFFFFF;
    --green:        #0C4B43;
    --green-2:      #0F5A50;
    --text-main:    #0C4B43;
    --text-soft:    #42675F;
    --text-muted:   #6F8C85;
    --border:       #D7E3DC;
    --yellow:       #E7E56A;
    --shadow:       0 8px 24px rgba(12, 75, 67, 0.05);

    /* ── Legacy aliases to keep existing component classes working ── */
    --navy:       var(--green);
    --navy-light: var(--green-2);
    --coral:      #D95C4F;
    --coral-dark: #B84D42;
    --coral-glow: rgba(217, 92, 79, 0.12);
    --white:      #FFFFFF;
    --off-white:  var(--bg);
    --slate:      var(--text-muted);

    --font:       'Inter', sans-serif;
    --font-title: 'Cormorant Garamond', serif;
    --mono:       'JetBrains Mono', monospace;

    /* ── Disease color scale ── */
    --c-asthma:  #EF4444;
    --c-copd:    #F59E0B;
    --c-pneumo:  #3B82F6;
    --c-bronchi: #10B981;
    --c-healthy: #94A3B8;
}

/* ── Page background (doctor = light) ── */
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main, section.main, section.main > div,
.block-container {
    background-color: var(--bg) !important;
}
.block-container {
    padding: 24px 20px 28px !important;
    max-width: 1320px !important;
}

/* ── Sidebar alignment with app.py ── */
[data-testid="stSidebar"] {
    background-color: var(--bg) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .stRadio label p {
    color: var(--text-main) !important;
    font-family: var(--font) !important;
    font-weight: 500 !important;
}
[data-testid="stSidebar"] button,
[data-testid="collapsedControl"] button {
    color: var(--text-main) !important;
}
header[data-testid="stHeader"] { display: none !important; }
footer { display: none !important; }

/* ── Doctor header bar ── */
.doc-header {
    background: linear-gradient(180deg, #FFFFFF 0%, #FBFAF6 100%);
    border: 1px solid var(--border);
    box-shadow: var(--shadow);
    padding: 14px 18px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-radius: 16px;
    margin-bottom: 16px;
}
.doc-logo {
    font-family: var(--font-title);
    font-size: 16px;
    font-weight: 600;
    color: var(--text-main);
    letter-spacing: 0.08em;
}
.doc-logo span { color: #D95C4F; }
.doc-user-info { display: flex; align-items: center; gap: 8px; }
.doc-avatar {
    width: 30px; height: 30px;
    background: var(--green);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    color: white; font-size: 11px; font-weight: 600;
    font-family: var(--font);
}
.doc-name { font-size: 12px; color: var(--text-soft); font-family: var(--font); }

/* ── Content area ── */
.doc-content { padding: 0; }

/* ── Cards ── */
.doc-card {
    background: var(--card);
    border-radius: 20px;
    border: 1px solid var(--border);
    padding: 18px 20px;
    margin-bottom: 16px;
    box-shadow: var(--shadow);
}
.doc-card-title {
    font-size: 14px;
    font-weight: 600;
    color: var(--text-main);
    letter-spacing: 0.02em;
    margin-bottom: 14px;
    font-family: var(--font);
}

/* ── Patient audio card ── */
.patient-audio-card {
    background: white;
    border-radius: 14px;
    border: 1px solid rgba(0,0,0,0.06);
    padding: 16px 18px;
    margin-bottom: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.patient-row { display: flex; align-items: center; gap: 14px; margin-bottom: 16px; }
.patient-avatar-doc {
    width: 40px; height: 40px;
    background: rgba(26,43,74,0.08); border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-weight: 600; font-size: 13px; color: var(--navy);
    flex-shrink: 0;
    font-family: var(--font);
}
.patient-info-doc { flex: 1; }
.patient-name-doc { font-size: 14px; font-weight: 500; color: #1e293b; font-family: var(--font); }
.patient-meta-doc { font-size: 12px; color: var(--slate); font-family: var(--font); }

/* ── Badges ── */
.badge { font-size: 11px; font-weight: 500; padding: 4px 10px; border-radius: 20px; font-family: var(--font); }
.badge-warn { background: #FEF3C7; color: #92400E; }
.badge-ok   { background: #D1FAE5; color: #065F46; }
.badge-danger { background: #FEE2E2; color: #991B1B; }

/* ── Audio controls ── */
.audio-controls {
    display: flex; align-items: center; gap: 10px;
    background: var(--off-white); border-radius: 10px;
    padding: 8px 12px; margin-bottom: 12px;
}
.play-btn {
    width: 30px; height: 30px;
    background: var(--navy);
    border-radius: 50%; border: none; cursor: pointer;
    display: flex; align-items: center; justify-content: center;
}
.play-btn svg { width: 12px; height: 12px; fill: white; }
.audio-track { flex: 1; height: 4px; background: rgba(0,0,0,0.08); border-radius: 2px; }
.audio-progress { height: 100%; width: 35%; background: var(--coral); border-radius: 2px; }
.audio-time { font-size: 11px; color: var(--slate); font-family: var(--mono); }

/* ── Proba bars ── */
.proba-list { display: flex; flex-direction: column; gap: 10px; }
.proba-top { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 4px; }
.proba-name { font-size: 13px; font-weight: 500; color: #1e293b; font-family: var(--font); }
.proba-pct  { font-size: 13px; font-weight: 500; font-family: var(--mono); }
.proba-bar-bg  { height: 6px; background: rgba(0,0,0,0.05); border-radius: 3px; }
.proba-bar-fill { height: 6px; border-radius: 3px; }

.color-asthma { color: var(--c-asthma); } .fill-asthma { background: var(--c-asthma); }
.color-copd   { color: var(--c-copd);   } .fill-copd   { background: var(--c-copd);   }
.color-pneumo { color: var(--c-pneumo); } .fill-pneumo { background: var(--c-pneumo); }
.color-bronchi{ color: var(--c-bronchi);} .fill-bronchi{ background: var(--c-bronchi);}
.color-sain   { color: var(--c-healthy);} .fill-sain   { background: var(--c-healthy);}

/* ── Confidence ── */
.confidence-display { text-align: center; padding: 10px 0; }
.confidence-big { font-size: 42px; font-weight: 300; color: var(--coral); font-family: var(--mono); line-height: 1; }
.confidence-label { font-size: 12px; color: var(--slate); margin-top: 6px; font-family: var(--font); }
.confidence-bar-wrap { margin-top: 16px; }
.confidence-bar-bg  { height: 8px; background: rgba(0,0,0,0.05); border-radius: 4px; overflow: hidden; }
.confidence-bar-fill { height: 8px; width: 82%; background: var(--coral); border-radius: 4px; }

/* ── Recommendation ── */
.recommendation-card {
    background: white;
    border-radius: 14px;
    border: 2px solid var(--coral);
    padding: 16px 18px;
    margin-bottom: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.rec-header { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
.rec-icon {
    width: 32px; height: 32px;
    background: var(--coral-glow); border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px;
}
.rec-title { font-size: 12px; font-weight: 600; color: var(--coral); text-transform: uppercase; letter-spacing: 0.06em; font-family: var(--font); }
.rec-text  { font-size: 13px; color: #334155; line-height: 1.6; font-family: var(--font); }

/* ── Dashboard metrics ── */
.dashboard-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 18px; }
.dash-metric {
    background: white; border-radius: 12px;
    border: 1px solid rgba(0,0,0,0.06);
    padding: 14px 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.dash-metric-label { font-size: 10px; color: var(--slate); text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px; font-family: var(--font); }
.dash-metric-value { font-size: 22px; font-weight: 500; font-family: var(--mono); color: #1e293b; }
.dash-metric-sub   { font-size: 11px; color: var(--slate); margin-top: 2px; font-family: var(--font); }

/* ── Compare ── */
.compare-intro { font-size: 13px; color: var(--slate); margin-bottom: 16px; line-height: 1.5; font-family: var(--font); }
.ref-audio-item {
    background: white; border-radius: 12px;
    border: 1px solid rgba(0,0,0,0.06);
    padding: 12px 14px;
    display: flex; align-items: center; gap: 12px;
    margin-bottom: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.ref-badge { padding: 4px 10px; border-radius: 20px; font-size: 11px; font-weight: 600; flex-shrink: 0; font-family: var(--font); }
.ref-asthma { background: #FEE2E2; color: #991B1B; }
.ref-copd   { background: #FEF3C7; color: #92400E; }
.ref-pneumo { background: #DBEAFE; color: #1e40af; }
.ref-sain   { background: #D1FAE5; color: #065F46; }
.ref-name { flex: 1; font-size: 13px; font-weight: 500; color: #1e293b; font-family: var(--font); }
.ref-meta { font-size: 11px; color: var(--slate); font-family: var(--font); }

.similarity-banner {
    background: var(--coral-glow);
    border: 1px solid rgba(232,113,74,0.3);
    border-radius: 12px;
    padding: 12px 16px;
    text-align: center;
    margin-bottom: 16px;
    font-size: 13px;
    color: var(--coral);
    font-weight: 500;
    font-family: var(--font);
}

.compare-audio-box {
    background: white; border-radius: 12px;
    border: 1px solid rgba(0,0,0,0.06);
    padding: 12px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.compare-audio-tag { font-size: 10px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px; font-family: var(--font); }
.tag-patient { color: var(--coral); }
.tag-ref     { color: var(--navy); }

/* ── Analysis grid ── */
.analysis-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }

/* ── Table ── */
.doc-table {
    width: 100%; font-size: 13px; border-collapse: collapse;
    font-family: var(--font);
}
.doc-table th {
    text-align: left; padding: 8px 0;
    color: var(--slate); font-weight: 500; font-size: 11px;
    border-bottom: 1px solid rgba(0,0,0,0.06);
}
.doc-table td {
    padding: 9px 0;
    border-bottom: 1px solid rgba(0,0,0,0.03);
}
.doc-table td.name-col { font-weight: 500; }
.doc-table td.mono-col { font-family: var(--mono); font-size: 12px; }
.doc-table td.sub-col  { font-size: 12px; color: var(--slate); }
.doc-table td.urgent-col { font-size: 12px; color: var(--c-asthma); font-weight: 500; }

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