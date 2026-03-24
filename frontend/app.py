import streamlit as st

# Force the centered layout for a more professional medical portal look
st.set_page_config(
    page_title="TESSAN", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# ── Custom CSS to hide default sidebar ──
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ── Auth State Initialization ──
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "is_doctor" not in st.session_state:
    st.session_state.is_doctor = False
if "user_full_name" not in st.session_state:
    st.session_state.user_full_name = ""
if "user_email" not in st.session_state:
    st.session_state.user_email = ""
if "user_id" not in st.session_state:
    st.session_state.user_id = ""
if "current_page" not in st.session_state:
    st.session_state.current_page = None
if "selected_pharmacy_id" not in st.session_state:
    st.session_state.selected_pharmacy_id = ""

# ── Horizontal Header Navigation ──
header_col1, header_col2, header_col3 = st.columns([1, 2, 1])

with header_col1:
    st.markdown(
        '<div style="font-family:\'Space Mono\',monospace;font-size:20px;font-weight:700;'
        'color:#0C4B43;letter-spacing:0.12em;">'
        'TESS<span style="color:#E8714A;">AN</span></div>',
        unsafe_allow_html=True,
    )

with header_col2:
    if st.session_state.authenticated:
        role_text = "Médecin" if st.session_state.is_doctor else "Patient"
        
        # Display user info and doctor button
        nav_col1, nav_col2 = st.columns([2, 1])
        with nav_col1:
            st.caption(f"**{st.session_state.user_full_name}** ({role_text})")
        with nav_col2:
            if st.session_state.is_doctor:
                if st.button("🎤 Diagnostic audio", key="nav_diagnostic"):
                    st.session_state.current_page = "audio_diagnostic"

with header_col3:
    if st.session_state.authenticated:
        if st.button("🚪 Déconnexion"):
            for key in [
                "authenticated",
                "is_doctor",
                "user_full_name",
                "user_email",
                "user_id",
                "auth_page",
                "register_role",
                "current_page",
            ]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

st.divider()

# ── Header styling ──
st.markdown(
    """
    <style>
    /* Header and main container styling */
    [data-testid="stAppViewContainer"] {
        background-color: #F6F4EE !important;
    }
    
    /* Style all buttons - white background with green text */
    button {
        background-color: white !important;
        color: #0C4B43 !important;
        border: 1px solid #D7E3DC !important;
        font-weight: 500 !important;
        padding: 8px 16px !important;
    }
    
    button:hover {
        background-color: #0C4B43 !important;
        color: white !important;
        border-color: #0C4B43 !important;
    }
    
    button:focus {
        background-color: white !important;
        color: #0C4B43 !important;
    }
    
    button p, button span, button * {
        color: #0C4B43 !important;
    }
    
    button:hover p, button:hover span, button:hover * {
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Auth gate + Router ──
if not st.session_state.authenticated:
    from pages.auth import render_auth_page
    render_auth_page()
    st.stop()

# Determine which page to show
if st.session_state.current_page == "audio_diagnostic":
    from pages.audio_diagnostic import render_diagnostic
    render_diagnostic(is_doctor=False)
elif st.session_state.is_doctor:
    from pages.doctor_dashboard import render_dashboard
    render_dashboard()
else:
    from pages.audio_diagnostic import render_diagnostic
    render_diagnostic(is_doctor=False)