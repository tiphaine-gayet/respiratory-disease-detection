import streamlit as st

# Force the centered layout for a more professional medical portal look
st.set_page_config(
    page_title="TESSAN", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# ── State Initialization ──
if "mode" not in st.session_state:
    st.session_state.mode = "patient"

# ── Sidebar Navigation ──
with st.sidebar:
    st.markdown(
        '<div style="font-family:\'Space Mono\',monospace;font-size:18px;font-weight:700;'
        'color:white;letter-spacing:0.12em;margin-bottom:24px;">'
        'TESS<span style="color:#E8714A;">AN</span></div>',
        unsafe_allow_html=True,
    )
    
    # Simple Toggle between Patient and Médecin
    mode_selection = st.radio(
        "Mode",
        ["Patient", "Médecin"],
        index=0 if st.session_state.mode == "patient" else 1,
        label_visibility="collapsed",
    )
    st.session_state.mode = "patient" if mode_selection == "Patient" else "doctor"

    # Sidebar styling to match the dark navy Tessan brand
    st.markdown(
    """
    <style>
    /* 1. Background and Text color for the Sidebar */
    [data-testid="stSidebar"] {
        background-color: #F6F4EE !important; /* var(--bg) from patient_main */
        border-right: 1px solid #D7E3DC !important; /* var(--border) */
    }

    /* 2. Style the radio buttons/text in the sidebar to match */
    [data-testid="stSidebar"] .stRadio label p {
        color: #0C4B43 !important; /* var(--text-main) */
        font-family: 'Inter', sans-serif !important;
        font-weight: 500 !important;
    }

    /* 3. Style the sidebar toggle button (the 'x' and the '>' to open) */
    [data-testid="stSidebar"] button, [data-testid="collapsedControl"] button {
        color: #0C4B43 !important;
    }
    
    /* 4. Ensure the main container doesn't leave a gap when sidebar is closed */
    [data-testid="stAppViewContainer"] {
        background-color: #F6F4EE !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
    )

# ── Router in app.py ──
if st.session_state.mode == "patient":
    from pages.audio_diagnostic import render_diagnostic
    render_diagnostic(is_doctor=False) # Normal view

else:
    from pages.audio_diagnostic import render_diagnostic
    render_diagnostic(is_doctor=True)