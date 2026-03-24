"""
frontend/pages/auth.py
Streamlit authentication UI (login/register) for Patient and Medecin users.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backend.router.auth import authenticate_user, create_user


_AUTH_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Cormorant+Garamond:wght@400;500;600;700&display=swap');

:root {
    --bg:         #F6F4EE;
    --card:       #FFFFFF;
    --green:      #0C4B43;
    --green-2:    #0F5A50;
    --green-sel:  #1F6A5F;
    --text-main:  #0C4B43;
    --text-soft:  #42675F;
    --text-muted: #6F8C85;
    --border:     #D7E3DC;
    --accent:     #D95C4F;
    --shadow:     0 8px 24px rgba(12, 75, 67, 0.05);
    --font-body:  'Inter', sans-serif;
    --font-title: 'Cormorant Garamond', serif;
}

[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main,
section.main,
section.main > div,
.block-container {
    background: var(--bg) !important;
}

.block-container {
    max-width: 820px !important;
    padding-top: 28px !important;
    padding-bottom: 24px !important;
}

.auth-wrap {
    max-width: 560px;
    margin: 24px auto 10px;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 20px;
    box-shadow: var(--shadow);
    padding: 24px 22px 16px;
}

.auth-logo {
    font-family: var(--font-title);
    color: var(--text-main);
    font-size: 38px;
    line-height: 1;
    margin-bottom: 6px;
    text-align: center;
}

.auth-logo span {
    color: var(--accent);
}

.auth-sub {
    text-align: center;
    color: var(--text-soft);
    font-family: var(--font-body);
    font-size: 13px;
    margin-bottom: 14px;
}

[data-testid="stForm"] {
    background: linear-gradient(180deg, #FFFFFF 0%, #FBFAF6 100%);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 12px 12px 6px;
}

[data-testid="stTextInput"] label,
[data-testid="stTextInput"] p,
[data-testid="stForm"] label,
[data-testid="stForm"] p,
[data-testid="stMarkdownContainer"] p {
    font-family: var(--font-body) !important;
    color: var(--text-soft) !important;
}

[data-testid="stTextInput"] input {
    border-radius: 10px !important;
    border: 1px solid var(--border) !important;
    color: var(--text-main) !important;
    background: #FFFFFF !important;
    font-family: var(--font-body) !important;
}

.stButton > button,
[data-testid="stFormSubmitButton"] > button {
    background: #FFFFFF !important;
    color: var(--green) !important;
    border-color: var(--green) !important;
    border-radius: 11px !important;
    border: 1px solid var(--green) !important;
    font-family: var(--font-body) !important;
    font-weight: 600 !important;
    transition: background-color 0.18s ease, color 0.18s ease, border-color 0.18s ease;
}

.stButton > button[kind="primary"],
[data-testid="stFormSubmitButton"] > button[kind="primary"] {
    background: var(--green-sel) !important;
    color: #FFFFFF !important;
    border-color: var(--green-sel) !important;
}

.stButton > button[kind="primary"] *,
[data-testid="stFormSubmitButton"] > button[kind="primary"] * {
    color: #FFFFFF !important;
}

.stButton > button[kind="secondary"] {
    background: #FFFFFF !important;
    color: var(--green) !important;
    border-color: var(--green) !important;
}

.stButton > button:hover,
[data-testid="stFormSubmitButton"] > button:hover {
    background: var(--green-sel) !important;
    color: #FFFFFF !important;
    border-color: var(--green-2) !important;
}

.stButton > button:hover *,
[data-testid="stFormSubmitButton"] > button:hover * {
    color: #FFFFFF !important;
}

[data-testid="stFormSubmitButton"] > button[kind="secondary"] {
    background: #FFFFFF !important;
    color: var(--green) !important;
    border-color: var(--green) !important;
}

.auth-note {
    color: var(--text-muted);
    font-size: 12px;
    font-family: var(--font-body);
    margin-top: 8px;
    text-align: center;
}

[data-testid="stAlert"] {
    border-radius: 12px !important;
    border: 1px solid var(--border) !important;
    font-family: var(--font-body) !important;
}
</style>
"""


def _ensure_auth_state() -> None:
    if "auth_page" not in st.session_state:
        st.session_state.auth_page = "login"
    if "register_role" not in st.session_state:
        st.session_state.register_role = "patient"


def _set_authenticated_user(user: dict) -> None:
    st.session_state.authenticated = True
    st.session_state.user_id = user["user_id"]
    st.session_state.user_full_name = user["full_name"]
    st.session_state.user_email = user["email"]
    st.session_state.is_doctor = bool(user["is_doctor"])


def render_auth_page() -> None:
    _ensure_auth_state()
    st.markdown(_AUTH_CSS, unsafe_allow_html=True)

    st.markdown('<div class="auth-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="auth-logo">TESS<span>AN</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="auth-sub">Connexion securisee a votre espace patient/medecin</div>', unsafe_allow_html=True)

    col_login, col_register = st.columns(2)
    with col_login:
        if st.button(
            "Connexion",
            use_container_width=True,
            type="primary" if st.session_state.auth_page == "login" else "secondary",
        ):
            st.session_state.auth_page = "login"
            st.rerun()
    with col_register:
        if st.button(
            "Enregistrement",
            use_container_width=True,
            type="primary" if st.session_state.auth_page == "register" else "secondary",
        ):
            st.session_state.auth_page = "register"
            st.rerun()

    if st.session_state.auth_page == "login":
        with st.form("login_form", clear_on_submit=False):
            email = st.text_input("Email")
            password = st.text_input("Mot de passe", type="password")
            submitted = st.form_submit_button("Se connecter", use_container_width=True, type="secondary")

        if submitted:
            ok, message, user = authenticate_user(email=email, password=password)
            if not ok or user is None:
                st.error(message)
            else:
                _set_authenticated_user(user)
                st.success("Connexion reussie")
                st.rerun()

    else:
        role_user, role_doc = st.columns(2)
        with role_user:
            if st.button("Patient", use_container_width=True, type="primary" if st.session_state.register_role == "patient" else "secondary"):
                st.session_state.register_role = "patient"
                st.rerun()
        with role_doc:
            if st.button("Medecin", use_container_width=True, type="primary" if st.session_state.register_role == "doctor" else "secondary"):
                st.session_state.register_role = "doctor"
                st.rerun()

        with st.form("register_form", clear_on_submit=False):
            full_name = st.text_input("Nom complet")
            email = st.text_input("Email")
            password = st.text_input("Mot de passe", type="password")
            password_confirm = st.text_input("Confirmer mot de passe", type="password")
            submitted = st.form_submit_button("Creer mon compte", use_container_width=True, type="primary")

        if submitted:
            if password != password_confirm:
                st.error("Les mots de passe ne correspondent pas.")
            else:
                is_doctor = st.session_state.register_role == "doctor"
                ok, message, user = create_user(
                    full_name=full_name,
                    email=email,
                    password=password,
                    is_doctor=is_doctor,
                )
                if not ok or user is None:
                    st.error(message)
                else:
                    _set_authenticated_user(user)
                    st.success("Compte cree et connexion etablie")
                    st.rerun()

    st.markdown('<div class="auth-note">Role choisi a l\'inscription: patient ou medecin.</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
