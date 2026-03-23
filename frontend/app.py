import streamlit as st

st.set_page_config(page_title="TESSAN", layout="wide")

# ── State ──
if "mode" not in st.session_state:
    st.session_state.mode = "patient"
if "doc_tab" not in st.session_state:
    st.session_state.doc_tab = "analyse"

# ── Sidebar ──
with st.sidebar:
    st.markdown(
        '<div style="font-family:\'Space Mono\',monospace;font-size:18px;font-weight:700;'
        'color:white;letter-spacing:0.12em;margin-bottom:24px;">'
        'TESS<span style="color:#E8714A;">AN</span></div>',
        unsafe_allow_html=True,
    )
    mode = st.radio(
        "Mode",
        ["Patient", "Médecin"],
        index=0 if st.session_state.mode == "patient" else 1,
        label_visibility="collapsed",
    )
    st.session_state.mode = "patient" if mode == "Patient" else "doctor"

    if st.session_state.mode == "doctor":
        st.markdown("---")
        st.markdown(
            '<p style="color:rgba(255,255,255,0.45);font-size:10px;'
            'text-transform:uppercase;letter-spacing:0.1em;margin-bottom:6px;">'
            'Espace Médecin</p>',
            unsafe_allow_html=True,
        )
        tab = st.radio(
            "Section",
            ["Analyse", "Comparer", "Dashboard"],
            index=["analyse", "comparer", "dashboard"].index(st.session_state.doc_tab),
            label_visibility="collapsed",
        )
        st.session_state.doc_tab = tab.lower()

    # Sidebar styling
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=Space+Mono:wght@400;700&display=swap');
        [data-testid="stSidebar"] {
            background: #232f42 !important;
            min-width: 190px !important;
            max-width: 210px !important;
        }
        [data-testid="stSidebar"] .stRadio label span,
        [data-testid="stSidebar"] .stRadio label p {
            color: rgba(255,255,255,0.7) !important;
            font-family: 'DM Sans', sans-serif !important;
            font-size: 13px !important;
        }
        [data-testid="stSidebar"] hr {
            border-color: rgba(255,255,255,0.1) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# ── Router ──
if st.session_state.mode == "patient":
    from pages.patient_main import render_patient
    render_patient()
elif st.session_state.doc_tab == "analyse":
    from pages.doctor_analysis import render_analysis
    render_analysis()
elif st.session_state.doc_tab == "comparer":
    from pages.doctor_compare import render_compare
    render_compare()
else:
    from pages.doctor_dashboard import render_dashboard
    render_dashboard()
