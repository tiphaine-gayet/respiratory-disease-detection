import streamlit as st

# Force the centered layout for a more professional medical portal look
st.set_page_config(
    page_title="TESSAN", 
    layout="wide", 
    initial_sidebar_state="collapsed"
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

# ── Sidebar Navigation ──
with st.sidebar:
    st.markdown(
        '<div style="font-family:\'Space Mono\',monospace;font-size:18px;font-weight:700;'
        'color:#0C4B43;letter-spacing:0.12em;margin-bottom:16px;">'
        'TESS<span style="color:#E8714A;">AN</span></div>',
        unsafe_allow_html=True,
    )

    if st.session_state.authenticated:
        role_text = "Medecin" if st.session_state.is_doctor else "Patient"
        st.caption(f"Connecte: {st.session_state.user_full_name} ({role_text})")
        st.caption(st.session_state.user_email)
        if st.button("Se deconnecter", use_container_width=True):
            for key in [
                "authenticated",
                "is_doctor",
                "user_full_name",
                "user_email",
                "user_id",
                "auth_page",
                "register_role",
            ]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

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

# ── Auth gate + Router ──
if not st.session_state.authenticated:
    from pages.auth import render_auth_page
    render_auth_page()
    st.stop()

if st.session_state.is_doctor:
    from pages.doctor_dashboard import render_dashboard
    render_dashboard()
else:
    from pages.audio_diagnostic import render_diagnostic
    render_diagnostic(is_doctor=False)