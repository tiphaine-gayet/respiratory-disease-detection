import streamlit as st

def render_doctor():
    """Doctor-facing interface — placeholder."""
    st.markdown(
        """
        <div class="doctor-wrap">
            <div class="doc-header">
                <div class="doc-logo">TESS<span>AN</span></div>
                <div class="doc-tabs">
                    <button class="doc-tab active">Analyse</button>
                    <button class="doc-tab">Historique</button>
                    <button class="doc-tab">Comparer</button>
                </div>
                <div class="doc-user-info">
                    <div class="doc-avatar">DR</div>
                    <div class="doc-name">Dr. Martin</div>
                </div>
            </div>
            <div class="doc-content">
                <div class="doc-card">
                    <div class="doc-card-title">En attente d'analyse</div>
                    <p style="color:#888; font-size:14px;">
                        Aucune analyse patient reçue pour le moment.
                    </p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
