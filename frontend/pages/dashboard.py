import streamlit as st

def render_dashboard():
    st.markdown("<div class='doc-content'>Dashboard</div>", unsafe_allow_html=True)

    st.metric("Patients", "312")
    st.metric("Cas urgents", "18")