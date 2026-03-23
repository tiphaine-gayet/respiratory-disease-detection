import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from components.doctor_styles import inject_doctor_css, doctor_header

def _bar_chart():
    """Distribution des diagnostics — bar chart."""
    labels = ["Asthme", "BPCO", "Pneumonie", "Bronchite", "Sain"]
    values = [34, 22, 18, 14, 12]
    colors = ["#E24B4A", "#EF9F27", "#378ADD", "#1D9E75", "#888780"]

    fig, ax = plt.subplots(figsize=(6, 3))
    bars = ax.bar(labels, values, color=colors, width=0.55, edgecolor="none", zorder=3)
    for bar in bars:
        bar.set_clip_on(False)
        # Rounded top corners via linewidth trick
    ax.set_ylabel("%", fontsize=10, color="#999")
    ax.tick_params(axis="x", labelsize=10, colors="#999")
    ax.tick_params(axis="y", labelsize=10, colors="#999")
    ax.set_ylim(0, 42)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#eee")
    ax.spines["bottom"].set_color("#eee")
    ax.yaxis.grid(True, color="#f0f0f0", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    fig.tight_layout()
    return fig

def _line_chart():
    """Tendance hebdomadaire — line chart."""
    weeks = ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"]
    patients = [32, 38, 29, 44, 41, 52, 48, 58]

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(weeks, patients, color="#2D3F5C", linewidth=2, marker="o", markersize=4, zorder=3)
    ax.fill_between(weeks, patients, alpha=0.06, color="#2D3F5C")
    ax.tick_params(axis="x", labelsize=10, colors="#999")
    ax.tick_params(axis="y", labelsize=10, colors="#999")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#eee")
    ax.spines["bottom"].set_color("#eee")
    ax.yaxis.grid(True, color="#f0f0f0", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    fig.tight_layout()
    return fig

def render_dashboard():
    inject_doctor_css()
    doctor_header()

    st.markdown('<div class="doc-content">', unsafe_allow_html=True)

    # ── Top KPI metrics ──
    st.markdown(
        """
        <div class="dashboard-row">
            <div class="dash-metric">
                <div class="dash-metric-label">Patients total (mois)</div>
                <div class="dash-metric-value">312</div>
                <div class="dash-metric-sub">+14% vs M-1</div>
            </div>
            <div class="dash-metric">
                <div class="dash-metric-label">Cas urgents</div>
                <div class="dash-metric-value" style="color:#E24B4A;">18</div>
                <div class="dash-metric-sub">Ce mois</div>
            </div>
            <div class="dash-metric">
                <div class="dash-metric-label">Confiance IA moy.</div>
                <div class="dash-metric-value">81%</div>
                <div class="dash-metric-sub">Tous patients</div>
            </div>
            <div class="dash-metric">
                <div class="dash-metric-label">Pharmacies actives</div>
                <div class="dash-metric-value">7</div>
                <div class="dash-metric-sub">Île-de-France</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Charts row ──
    col_bar, col_line = st.columns(2)

    with col_bar:
        st.markdown(
            '<div class="doc-card"><div class="doc-card-title">Distribution des diagnostics — Région IDF</div>',
            unsafe_allow_html=True,
        )
        st.pyplot(_bar_chart(), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_line:
        st.markdown(
            '<div class="doc-card"><div class="doc-card-title">Tendance hebdomadaire — Nouveaux patients</div>',
            unsafe_allow_html=True,
        )
        st.pyplot(_line_chart(), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Patients table ──
    st.markdown(
        """
        <div class="doc-card" style="margin-top:16px;">
            <div class="doc-card-title">Patients récents</div>
            <table class="doc-table">
                <tr>
                    <th>Patient</th>
                    <th>Pré-diagnostic</th>
                    <th>Confiance</th>
                    <th>Action</th>
                    <th>Pharmacie</th>
                </tr>
                <tr>
                    <td class="name-col">Martin Rousseau</td>
                    <td><span class="badge badge-warn">Asthme 62%</span></td>
                    <td class="mono-col">82%</td>
                    <td class="sub-col">Suivi 48h</td>
                    <td style="font-size:12px;">Pharm. Opéra</td>
                </tr>
                <tr>
                    <td class="name-col">Sophie Menard</td>
                    <td><span class="badge badge-danger">Pneumonie 58%</span></td>
                    <td class="mono-col">76%</td>
                    <td class="urgent-col">Urgence</td>
                    <td style="font-size:12px;">Pharm. Bastille</td>
                </tr>
                <tr>
                    <td class="name-col">Paul Girard</td>
                    <td><span class="badge badge-ok">Sain 71%</span></td>
                    <td class="mono-col">91%</td>
                    <td class="sub-col">RAS</td>
                    <td style="font-size:12px;">Pharm. Nation</td>
                </tr>
                <tr>
                    <td class="name-col">Camille Dubois</td>
                    <td><span class="badge badge-warn">BPCO 44%</span></td>
                    <td class="mono-col">68%</td>
                    <td class="sub-col">Suivi 1 sem.</td>
                    <td style="font-size:12px;">Pharm. Opéra</td>
                </tr>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)
