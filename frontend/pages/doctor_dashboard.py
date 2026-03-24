"""
frontend/pages/doctor_dashboard.py
──────────────────────────────────────────────────────────────────────────────
Tableau de bord médecin — pré-diagnostics respiratoires.

• Carte pydeck (France) — ronds proportionnels par pharmacie / commune / dép.
• KPIs agrégés sur la période
• Tableau filtrable des pré-diagnostics patients
"""

from __future__ import annotations

import pydeck as pdk
import pandas as pd
import streamlit as st
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # Ensure project root is on sys.path
from backend.router.predictions import load_predictions
from components.doctor_styles import inject_doctor_css, doctor_header

# ── Métadonnées maladies ───────────────────────────────────────────────────────

DISEASES: dict[str, dict] = {
    "asthma":    {"label": "Asthme",    "col": "pct_asthma",    "rgb": [217, 92,  79]},
    "copd":      {"label": "BPCO",      "col": "pct_copd",      "rgb": [216, 166, 61]},
    "bronchial": {"label": "Bronchite", "col": "pct_bronchial", "rgb": [91,  141, 239]},
    "pneumonia": {"label": "Pneumonie", "col": "pct_pneumonia", "rgb": [139, 92,  246]},
}

_DISEASE_COLS = [d["col"] for d in DISEASES.values()]

_ACTION_LABELS = {
    "RAS":       "RAS",
    "SUIVI_48H": "Suivi 48 h",
    "URGENCE":   "Urgence",
}


def _dominant(row) -> str:
    """Renvoie la clé de la maladie dominante, ou 'healthy'."""
    best_k = max(DISEASES, key=lambda k: row[DISEASES[k]["col"]])
    if row["pct_healthy"] >= row[DISEASES[best_k]["col"]]:
        return "healthy"
    return best_k


# ── Chargement des données ────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner="Chargement des pré-diagnostics…")
def _fetch(date_from: date, date_to: date) -> pd.DataFrame:
    return load_predictions(date_from, date_to)


# ── Agrégation pour la carte ──────────────────────────────────────────────────

def _agg_for_map(df: pd.DataFrame, disease_key: str, granularity: str) -> pd.DataFrame:
    """
    Aggrège les pré-diagnostics pour un ScatterplotLayer pydeck.
    Colonnes : lat, lon, cas, color, radius, tooltip_text, grp_key.
    """
    if df.empty:
        return pd.DataFrame()

    gdf = df.dropna(subset=["loc_lat", "loc_long"]).copy()
    if gdf.empty:
        return pd.DataFrame()

    gdf["dominant"] = gdf.apply(_dominant, axis=1)

    # Filtre par maladie
    if disease_key == "all":
        gdf = gdf[gdf["dominant"] != "healthy"]
    else:
        gdf = gdf[gdf["dominant"] == disease_key]

    if gdf.empty:
        return pd.DataFrame()

    # Couleur par ligne (stockée en colonnes entières pour groupby)
    if disease_key == "all":
        gdf["_r"] = gdf["dominant"].map({k: v["rgb"][0] for k, v in DISEASES.items()})
        gdf["_g"] = gdf["dominant"].map({k: v["rgb"][1] for k, v in DISEASES.items()})
        gdf["_b"] = gdf["dominant"].map({k: v["rgb"][2] for k, v in DISEASES.items()})
    else:
        r, g, b = DISEASES[disease_key]["rgb"]
        gdf["_r"], gdf["_g"], gdf["_b"] = r, g, b

    # Granularité
    if granularity == "Pharmacies":
        gdf["grp_key"]   = gdf["pharmacie_id"].fillna("?")
        gdf["grp_label"] = gdf.apply(
            lambda row: (row.get("pharmacie_nom") or row["pharmacie_id"] or "?")
            + (f" ({row['commune']})" if pd.notna(row.get("commune")) else ""),
            axis=1,
        )
        gdf["grp_lat"] = gdf["loc_lat"]
        gdf["grp_lon"] = gdf["loc_long"]

    elif granularity == "Communes":
        gdf["grp_key"]   = gdf["commune"].fillna("?")
        gdf["grp_label"] = gdf["commune"].fillna("Inconnue")
        gdf["grp_lat"]   = gdf["commune"].map(gdf.groupby("commune")["loc_lat"].mean())
        gdf["grp_lon"]   = gdf["commune"].map(gdf.groupby("commune")["loc_long"].mean())

    else:  # Départements
        gdf["grp_key"]   = gdf["code_departement"].fillna("??")
        gdf["grp_label"] = "Dép. " + gdf["code_departement"].fillna("??")
        gdf["grp_lat"]   = gdf["code_departement"].map(gdf.groupby("code_departement")["loc_lat"].mean())
        gdf["grp_lon"]   = gdf["code_departement"].map(gdf.groupby("code_departement")["loc_long"].mean())

    agg = (
        gdf.groupby(["grp_key", "grp_label", "grp_lat", "grp_lon"])
        .agg(
            cas=("prediction_id", "count"),
            r=("_r", "first"),
            g=("_g", "first"),
            b=("_b", "first"),
        )
        .reset_index()
    )

    agg.rename(columns={"grp_lat": "lat", "grp_lon": "lon", "grp_label": "label"}, inplace=True)
    agg["color"]  = agg[["r", "g", "b"]].apply(
        lambda x: [int(x.r), int(x.g), int(x.b), 210], axis=1
    )
    max_cas        = agg["cas"].max() or 1
    agg["radius"]  = (4_000 + (agg["cas"] / max_cas) * 42_000).astype(int)
    agg["tooltip_text"] = agg["label"] + " — " + agg["cas"].astype(str) + " pré-diagnostic(s)"

    return agg[["lat", "lon", "cas", "color", "radius", "tooltip_text", "grp_key"]]


# ── Rendu de la carte ─────────────────────────────────────────────────────────

def _render_map(df: pd.DataFrame, disease_key: str, granularity: str, selected_ids: list[str]) -> None:
    map_df = _agg_for_map(df, disease_key, granularity)

    if map_df.empty:
        st.markdown(
            '<div class="doc-empty">Aucune donnée géographique pour la période sélectionnée.</div>',
            unsafe_allow_html=True,
        )
        return

    layers = [
        pdk.Layer(
            "ScatterplotLayer",
            data=map_df,
            get_position=["lon", "lat"],
            get_fill_color="color",
            get_radius="radius",
            radius_min_pixels=5,
            radius_max_pixels=38,
            opacity=0.75,
            stroked=True,
            get_line_color=[255, 255, 255, 80],
            line_width_min_pixels=1,
            pickable=True,
        )
    ]

    # Anneau de surbrillance pour les pharmacies sélectionnées
    if selected_ids and granularity == "Pharmacies":
        hl = map_df[map_df["grp_key"].isin(selected_ids)]
        if not hl.empty:
            layers.append(
                pdk.Layer(
                    "ScatterplotLayer",
                    data=hl,
                    get_position=["lon", "lat"],
                    get_fill_color=[255, 255, 255, 0],
                    get_radius="radius",
                    radius_min_pixels=5,
                    radius_max_pixels=38,
                    opacity=1.0,
                    stroked=True,
                    get_line_color=[231, 229, 106, 255],   # --yellow du thème
                    line_width_min_pixels=3,
                    pickable=False,
                )
            )

    st.pydeck_chart(
        pdk.Deck(
            layers=layers,
            initial_view_state=pdk.ViewState(
                latitude=46.8,
                longitude=2.35,
                zoom=5,
                pitch=0,
            ),
            map_style="https://basemaps.cartocdn.com/gl/voyager-gl-style/style.json",
            tooltip={"text": "{tooltip_text}"},
            height=440,
        ),
        use_container_width=True,
    )


# ── Dashboard principal ───────────────────────────────────────────────────────

def render_dashboard() -> None:
    inject_doctor_css()
    doctor_header()

    st.markdown(_DASHBOARD_CSS, unsafe_allow_html=True)
    st.markdown('<div class="doc-content">', unsafe_allow_html=True)

    today            = date.today()
    first_this_month = today.replace(day=1)
    default_from     = (first_this_month - timedelta(days=1)).replace(day=1)

    # ── Disposition deux colonnes : carte | filtres ───────────────────────
    map_col, ctrl_col = st.columns([3, 2], gap="medium")

    # Filtres rendus en premier pour obtenir leurs valeurs
    with ctrl_col:
        st.markdown('<div class="doc-card">', unsafe_allow_html=True)
        st.markdown('<div class="doc-card-title">Filtres &amp; indicateurs</div>', unsafe_allow_html=True)

        d1, d2 = st.columns(2)
        with d1:
            date_from = st.date_input("Du", value=default_from, key="dash_date_from")
        with d2:
            date_to = st.date_input("Au", value=today, key="dash_date_to")

        if date_from > date_to:
            st.warning("La date de début doit être antérieure à la date de fin.")
            st.markdown("</div></div>", unsafe_allow_html=True)
            return

        granularity = st.radio(
            "Granularité carte",
            options=["Pharmacies", "Communes", "Départements"],
            index=2,
            horizontal=True,
            key="map_granularity",
        )

    # ── Chargement ────────────────────────────────────────────────────────
    df = _fetch(date_from, date_to)

    ph_options: dict[str, str] = {}
    if not df.empty:
        ph_df = (
            df.dropna(subset=["pharmacie_nom"])
            .drop_duplicates("pharmacie_id")[["pharmacie_id", "pharmacie_nom", "commune"]]
        )
        ph_options = {
            row["pharmacie_id"]: f"{row['pharmacie_nom']} ({row['commune']})"
            for _, row in ph_df.iterrows()
        }

    # ── Suite colonne droite : pharmacie + KPIs ───────────────────────────
    with ctrl_col:
        selected_ids: list[str] = st.multiselect(
            "Filtrer par pharmacie",
            options=list(ph_options.keys()),
            format_func=lambda k: ph_options.get(k, k),
            key="ph_filter",
            help="Filtre le tableau ci-dessous. La carte affiche toujours l'ensemble des cas.",
        )

        total     = len(df)
        urgents   = int((df["action"] == "URGENCE").sum())   if not df.empty else 0
        confiance = float(df["pct_confiance"].mean() * 100)  if not df.empty else 0.0
        ph_count  = int(df["pharmacie_id"].nunique())         if not df.empty else 0

        st.markdown(
            f"""
            <div class="kpi-grid">
                <div class="dash-metric">
                    <div class="dash-metric-label">Pré-diagnostics</div>
                    <div class="dash-metric-value">{total}</div>
                    <div class="dash-metric-sub">Période sélectionnée</div>
                </div>
                <div class="dash-metric">
                    <div class="dash-metric-label">Cas urgents</div>
                    <div class="dash-metric-value" style="color:#D95C4F;">{urgents}</div>
                    <div class="dash-metric-sub">Action = Urgence</div>
                </div>
                <div class="dash-metric">
                    <div class="dash-metric-label">Confiance IA</div>
                    <div class="dash-metric-value">{confiance:.0f}%</div>
                    <div class="dash-metric-sub">Moyenne</div>
                </div>
                <div class="dash-metric">
                    <div class="dash-metric-label">Pharmacies actives</div>
                    <div class="dash-metric-value">{ph_count}</div>
                    <div class="dash-metric-sub">Période sélectionnée</div>
                </div>
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Carte (colonne gauche) ─────────────────────────────────────────────
    with map_col:
        st.markdown('<div class="doc-card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="doc-card-title">Carte des pré-diagnostics respiratoires</div>',
            unsafe_allow_html=True,
        )
        tab_all, tab_asthma, tab_copd, tab_bronchial, tab_pneumonia = st.tabs(
            ["🗺 Toutes maladies", "Asthme", "BPCO", "Bronchite", "Pneumonie"]
        )
        with tab_all:
            _render_map(df, "all", granularity, selected_ids)
        with tab_asthma:
            _render_map(df, "asthma", granularity, selected_ids)
        with tab_copd:
            _render_map(df, "copd", granularity, selected_ids)
        with tab_bronchial:
            _render_map(df, "bronchial", granularity, selected_ids)
        with tab_pneumonia:
            _render_map(df, "pneumonia", granularity, selected_ids)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Tableau pré-diagnostics ───────────────────────────────────────────
    df_table = df[df["pharmacie_id"].isin(selected_ids)].copy() if selected_ids else df.copy()

    filter_suffix = ""
    if selected_ids:
        names = [ph_options.get(s, s) for s in selected_ids]
        filter_suffix = f" — {', '.join(names)}"

    st.markdown(
        f'<div class="doc-card" style="margin-top:16px;">'
        f'<div class="doc-card-title">Pré-diagnostics patients{filter_suffix}</div>',
        unsafe_allow_html=True,
    )

    if df_table.empty:
        st.markdown(
            '<div class="doc-empty">Aucun pré-diagnostic pour la période et la sélection.</div>',
            unsafe_allow_html=True,
        )
    else:
        df_table = df_table.copy().reset_index(drop=True)
        df_table["dominant"] = df_table.apply(_dominant, axis=1)
        df_table["Pré-diagnostic"] = df_table.apply(
            lambda r: f"Sain ({r['pct_healthy'] * 100:.0f} %)"
            if r["dominant"] == "healthy"
            else (
                f"{DISEASES[r['dominant']]['label']} "
                f"({r[DISEASES[r['dominant']]['col']] * 100:.0f} %)"
            ),
            axis=1,
        )

        display = pd.DataFrame({
            "Date":           pd.to_datetime(df_table["predicted_at"]).dt.strftime("%d/%m/%Y %H:%M"),
            "Patient (NSS)":  df_table["patient_id"].astype(str).str[:7] + "••••••••",
            "Pharmacie":      df_table["pharmacie_nom"].fillna("—"),
            "Commune":        df_table["commune"].fillna("—"),
            "Code postal":    df_table["code_postal"].fillna("—"),
            "Pré-diagnostic": df_table["Pré-diagnostic"],
            "Confiance IA":   (df_table["pct_confiance"] * 100).round(1).astype(str) + " %",
            "Action":         df_table["action"].map(_ACTION_LABELS).fillna(df_table["action"]),
        })

        _diag_hex = {k: "#{:02X}{:02X}{:02X}".format(*v["rgb"]) for k, v in DISEASES.items()}
        _diag_hex["healthy"] = "#1D9E75"
        _action_colors_map = {
            _ACTION_LABELS["URGENCE"]:   "#D95C4F",
            _ACTION_LABELS["SUIVI_48H"]: "#D8A63D",
            _ACTION_LABELS["RAS"]:       "#9AA8A5",
        }

        def _badge(text: str, color: str) -> str:
            return (
                f'<span style="background:{color}22;color:{color};'
                f'border:1px solid {color}55;border-radius:6px;'
                f'padding:2px 8px;font-size:12px;font-weight:600;'
                f'white-space:nowrap;">{text}</span>'
            )

        rows = []
        for i in display.index:
            dom         = df_table.at[i, "dominant"]
            diag_color  = _diag_hex.get(dom, "#9AA8A5")
            action_val  = display.at[i, "Action"]
            action_color = _action_colors_map.get(action_val, "#9AA8A5")
            row_bg      = "background:rgba(217,92,79,0.06);" if action_val == _ACTION_LABELS["URGENCE"] else ""
            diag_cell   = display.at[i, "Pré-diagnostic"] if dom == "healthy" else _badge(display.at[i, "Pré-diagnostic"], diag_color)
            rows.append(
                f'<tr style="{row_bg}">'
                f'<td style="color:var(--text-soft);font-size:12px;">{display.at[i,"Date"]}</td>'
                f'<td style="font-family:monospace;font-size:12px;">{display.at[i,"Patient (NSS)"]}</td>'
                f'<td>{display.at[i,"Pharmacie"]}</td>'
                f'<td style="color:var(--text-muted);">{display.at[i,"Commune"]}</td>'
                f'<td>{diag_cell}</td>'
                f'<td style="text-align:right;">{display.at[i,"Confiance IA"]}</td>'
                f'<td>{_badge(action_val, action_color)}</td>'
                f'</tr>'
            )

        _headers = ["Date", "Patient (NSS)", "Pharmacie", "Commune", "Pré-diagnostic", "Confiance IA", "Action"]
        thead = "".join(f"<th>{h}</th>" for h in _headers)
        st.markdown(
            f'<div style="overflow-x:auto;">'
            f'<table class="pred-table"><thead><tr>{thead}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>',
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ── CSS spécifique au dashboard (complète doctor_styles) ─────────────────────

_DASHBOARD_CSS = """
<style>
/* ━━ KPI 2×2 grid (colonne filtres) ━━ */
.kpi-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin-top: 16px;
}

.dash-metric {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 14px 16px;
    box-shadow: var(--shadow);
}

.dash-metric-label {
    font-size: 10px;
    font-weight: 700;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.10em;
    font-family: var(--font-body);
    margin-bottom: 6px;
}

.dash-metric-value {
    font-size: 26px;
    font-weight: 700;
    color: var(--text-main);
    font-family: var(--font-title);
    line-height: 1;
    margin-bottom: 4px;
}

.dash-metric-sub {
    font-size: 11px;
    color: var(--text-muted);
    font-family: var(--font-body);
}

/* ━━ Cards ━━ */
.doc-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 20px 22px;
    box-shadow: var(--shadow);
}

.doc-card-title {
    font-size: 15px;
    font-weight: 600;
    color: var(--text-main);
    font-family: var(--font-body);
    margin-bottom: 16px;
}

/* ━━ Empty state ━━ */
.doc-empty {
    color: var(--text-muted);
    font-size: 13px;
    font-family: var(--font-body);
    padding: 24px 0 8px;
    text-align: center;
}

/* ━━ Tabs — align with Tessan palette ━━ */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    gap: 4px;
    border-bottom: 1px solid var(--border) !important;
    background: transparent !important;
}

[data-testid="stTabs"] [data-baseweb="tab"] {
    font-family: var(--font-body) !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    color: var(--text-muted) !important;
    padding: 8px 14px !important;
    border-radius: 8px 8px 0 0 !important;
    background: transparent !important;
}

[data-testid="stTabs"] [aria-selected="true"] {
    color: var(--green) !important;
    font-weight: 600 !important;
    border-bottom: 2px solid var(--green) !important;
}

/* ━━ Radio (granularity) ━━ */
[data-testid="stRadio"] label {
    font-family: var(--font-body) !important;
    font-size: 13px !important;
    color: var(--text-soft) !important;
}

/* ━━ Multiselect ━━ */
[data-testid="stMultiSelect"] * {
    font-family: var(--font-body) !important;
    font-size: 13px !important;
}

/* ━━ Dataframe ━━ */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    overflow: hidden;
}

/* ━━ Date inputs ━━ */
[data-testid="stDateInput"] input {
    font-family: var(--font-body) !important;
    font-size: 13px !important;
    border-radius: 10px !important;
    border-color: var(--border) !important;
    color: var(--text-main) !important;
}

/* ━━ Map container breathing room ━━ */
[data-testid="stDeckGlJsonChart"] {
    border-radius: 14px !important;
    overflow: hidden;
    margin-top: 4px;
}

/* ━━ Predictions table ━━ */
.pred-table {
    width: 100%;
    border-collapse: collapse;
    font-family: var(--font-body);
    font-size: 13px;
    color: var(--text-main);
}

.pred-table thead tr {
    border-bottom: 1px solid var(--border);
}

.pred-table th {
    text-align: left;
    padding: 8px 12px;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted);
}

.pred-table td {
    padding: 9px 12px;
    border-bottom: 1px solid var(--border);
    vertical-align: middle;
}

.pred-table tbody tr:last-child td {
    border-bottom: none;
}

.pred-table tbody tr:hover td {
    background: rgba(0,0,0,0.02);
}
</style>
"""
