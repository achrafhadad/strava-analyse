"""
Streamlit Dashboard voor Strava Run Analyse.
Start met: streamlit run dashboard.py
"""
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analyze import (
    load_runs, summary_stats, weekly_summary, monthly_summary,
    predict_pace, predict_weekly_volume, personal_records,
    training_load, format_pace,
)
from qa import answer_question

# --- Page config & theme ---
st.set_page_config(
    page_title="Strava Run Analyse",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS ---
st.markdown("""
<style>
/* Import Google Font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* Global */
.stApp {
    font-family: 'Inter', sans-serif;
}

/* Hide default header */
header[data-testid="stHeader"] {
    background: transparent;
}

/* Hero header */
.hero {
    background: linear-gradient(135deg, #FC4C02 0%, #FF6B35 50%, #FF8C42 100%);
    padding: 2.5rem 2rem;
    border-radius: 20px;
    margin-bottom: 2rem;
    color: white;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 400px;
    height: 400px;
    background: rgba(255,255,255,0.08);
    border-radius: 50%;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -30%;
    left: 10%;
    width: 200px;
    height: 200px;
    background: rgba(255,255,255,0.05);
    border-radius: 50%;
}
.hero h1 {
    font-size: 2.4rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: -0.5px;
}
.hero p {
    font-size: 1.1rem;
    opacity: 0.9;
    margin-top: 0.5rem;
    font-weight: 300;
}

/* KPI cards */
.kpi-card {
    background: linear-gradient(145deg, #1a1a2e, #16213e);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}
.kpi-card:hover {
    transform: translateY(-4px);
    border-color: rgba(252, 76, 2, 0.3);
    box-shadow: 0 8px 32px rgba(252, 76, 2, 0.15);
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, #FC4C02, #FF8C42);
    opacity: 0;
    transition: opacity 0.3s;
}
.kpi-card:hover::before {
    opacity: 1;
}
.kpi-icon {
    font-size: 2rem;
    margin-bottom: 0.5rem;
    display: block;
}
.kpi-value {
    font-size: 1.8rem;
    font-weight: 800;
    color: #FC4C02;
    margin: 0.3rem 0;
    letter-spacing: -0.5px;
}
.kpi-label {
    font-size: 0.8rem;
    color: #8892b0;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 600;
}

/* Section headers */
.section-header {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 2.5rem 0 1.2rem 0;
    padding-bottom: 0.8rem;
    border-bottom: 2px solid rgba(252, 76, 2, 0.15);
}
.section-header .icon {
    font-size: 1.6rem;
    background: linear-gradient(135deg, #FC4C02, #FF8C42);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.section-header h2 {
    font-size: 1.5rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.3px;
}

/* Record cards */
.record-card {
    background: linear-gradient(145deg, #1a1a2e, #16213e);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 1.5rem;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    height: 100%;
}
.record-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 32px rgba(252, 76, 2, 0.12);
    border-color: rgba(252, 76, 2, 0.25);
}
.record-card .trophy {
    font-size: 2.2rem;
    margin-bottom: 0.5rem;
    display: block;
}
.record-card .record-value {
    font-size: 1.6rem;
    font-weight: 800;
    color: #FC4C02;
}
.record-card .record-label {
    font-size: 0.75rem;
    color: #8892b0;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
}
.record-card .record-detail {
    font-size: 0.85rem;
    color: #a8b2d1;
    margin-top: 0.6rem;
    line-height: 1.5;
}

/* Prediction cards */
.pred-card {
    background: linear-gradient(145deg, #1a1a2e, #16213e);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 1.5rem;
}
.pred-card .trend-up { color: #00C48C; }
.pred-card .trend-down { color: #FC4C02; }
.pred-week {
    display: flex;
    justify-content: space-between;
    padding: 0.4rem 0;
    border-bottom: 1px solid rgba(255,255,255,0.04);
    font-size: 0.9rem;
}
.pred-week:last-child { border: none; }

/* Training load gauge */
.load-card {
    background: linear-gradient(145deg, #1a1a2e, #16213e);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
}
.load-ratio {
    font-size: 3rem;
    font-weight: 900;
    letter-spacing: -1px;
}
.load-good { color: #00C48C; }
.load-warning { color: #FFB020; }
.load-danger { color: #FF4757; }

/* Chat section */
.chat-header {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin-bottom: 1rem;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0a1a 0%, #1a1a2e 100%);
}
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #FC4C02;
}

/* Plotly charts dark background */
.js-plotly-plot .plotly .main-svg {
    border-radius: 12px;
}

/* Smooth animations */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}
.animate-in {
    animation: fadeInUp 0.5s ease-out forwards;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #FC4C02; border-radius: 3px; }

/* Expander styling */
.streamlit-expanderHeader {
    font-weight: 600;
    font-size: 0.95rem;
}
</style>
""", unsafe_allow_html=True)

# --- Plotly template ---
PLOTLY_TEMPLATE = dict(
    layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(26,26,46,0.5)",
        font=dict(family="Inter", color="#ccd6f6"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.05)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.05)"),
        colorway=["#FC4C02", "#FF6B35", "#FF8C42", "#00C48C", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7"],
        hoverlabel=dict(bgcolor="#1a1a2e", font_size=13, font_family="Inter"),
        margin=dict(l=40, r=40, t=40, b=40),
    )
)

# --- Data laden ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RUNS_CSV = os.path.join(DATA_DIR, "runs.csv")

if not os.path.exists(RUNS_CSV):
    st.error("Geen data gevonden. Draai eerst `python fetch_data.py` om je Strava data op te halen.")
    st.stop()

df = load_runs()

# --- Sidebar ---
with st.sidebar:
    st.markdown("### 🎛️ Filters")
    st.markdown("---")

    years = sorted(df["year"].unique())
    selected_years = st.multiselect("📅 Jaar", years, default=years)

    run_types = df["sport_type"].unique() if "sport_type" in df.columns else ["Run"]
    selected_types = st.multiselect("🏃 Type", run_types, default=run_types)

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#8892b0; font-size:0.75rem;'>"
        "Powered by Strava API<br>Built with Streamlit & Plotly"
        "</div>",
        unsafe_allow_html=True,
    )

filtered = df[df["year"].isin(selected_years)]
if "sport_type" in filtered.columns:
    filtered = filtered[filtered["sport_type"].isin(selected_types)]

stats = summary_stats(filtered)

# === HERO HEADER ===
st.markdown(f"""
<div class="hero animate-in">
    <h1>🏃 Strava Run Analyse</h1>
    <p>{stats['totaal_runs']} runs · {stats['totaal_km']} km · Gem. pace {stats['gem_pace']}/km</p>
</div>
""", unsafe_allow_html=True)

# === KPI CARDS ===
kpi_data = [
    ("🏃", stats["totaal_runs"], "Totaal Runs"),
    ("📏", f"{stats['totaal_km']} km", "Totaal Afstand"),
    ("⏱️", stats["gem_pace"], "Gem. Pace"),
    ("⚡", stats["snelste_pace"], "Snelste Pace"),
    ("🕐", f"{stats['totaal_uren']} uur", "Totaal Tijd"),
    ("📐", f"{stats['gem_afstand_km']} km", "Gem. Afstand"),
    ("⛰️", f"{stats['totaal_hoogtemeters']}m", "Hoogtemeters"),
    ("❤️", stats["gem_hartslag"], "Gem. Hartslag"),
]

cols = st.columns(4)
for i, (icon, value, label) in enumerate(kpi_data[:4]):
    with cols[i]:
        st.markdown(f"""
        <div class="kpi-card animate-in" style="animation-delay: {i*0.1}s">
            <span class="kpi-icon">{icon}</span>
            <div class="kpi-value">{value}</div>
            <div class="kpi-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

cols2 = st.columns(4)
for i, (icon, value, label) in enumerate(kpi_data[4:]):
    with cols2[i]:
        st.markdown(f"""
        <div class="kpi-card animate-in" style="animation-delay: {(i+4)*0.1}s">
            <span class="kpi-icon">{icon}</span>
            <div class="kpi-value">{value}</div>
            <div class="kpi-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# === PACE ONTWIKKELING ===
st.markdown("""
<div class="section-header">
    <span class="icon">⚡</span>
    <h2>Pace Ontwikkeling</h2>
</div>
""", unsafe_allow_html=True)

fig_pace = px.scatter(
    filtered, x="start_date_local", y="pace_min_per_km",
    color="distance_km", size="distance_km",
    labels={"pace_min_per_km": "Pace (min/km)", "start_date_local": "Datum", "distance_km": "Afstand (km)"},
    color_continuous_scale=[[0, "#4ECDC4"], [0.5, "#FC4C02"], [1, "#FF4757"]],
)
filtered_clean = filtered.dropna(subset=["pace_min_per_km"])
if len(filtered_clean) > 2:
    z = np.polyfit(filtered_clean["start_date_local"].astype(np.int64) // 10**9, filtered_clean["pace_min_per_km"], 1)
    p = np.poly1d(z)
    x_trend = filtered_clean["start_date_local"].sort_values()
    y_trend = p(x_trend.astype(np.int64) // 10**9)
    fig_pace.add_trace(go.Scatter(
        x=x_trend, y=y_trend, mode="lines", name="Trend",
        line=dict(color="#00C48C", dash="dash", width=3),
    ))
fig_pace.update_yaxes(autorange="reversed")
fig_pace.update_layout(PLOTLY_TEMPLATE["layout"])
st.plotly_chart(fig_pace, use_container_width=True)

# === WEKELIJKS VOLUME ===
st.markdown("""
<div class="section-header">
    <span class="icon">📊</span>
    <h2>Wekelijks Volume</h2>
</div>
""", unsafe_allow_html=True)

weekly = weekly_summary(filtered).reset_index()
weekly["year_week_str"] = weekly["year_week"].astype(str)

fig_weekly = go.Figure()
fig_weekly.add_trace(go.Bar(
    x=weekly["year_week_str"], y=weekly["km"], name="Kilometers",
    marker=dict(color=weekly["km"], colorscale=[[0, "#FF8C42"], [1, "#FC4C02"]], cornerradius=6),
))
fig_weekly.add_trace(go.Scatter(
    x=weekly["year_week_str"], y=weekly["runs"], name="Runs",
    yaxis="y2", line=dict(color="#00C48C", width=3), mode="lines+markers",
    marker=dict(size=8, symbol="circle"),
))
fig_weekly.update_layout(
    PLOTLY_TEMPLATE["layout"],
    yaxis=dict(title="Kilometers", gridcolor="rgba(255,255,255,0.05)"),
    yaxis2=dict(title="Aantal runs", overlaying="y", side="right", gridcolor="rgba(255,255,255,0.05)"),
    xaxis=dict(title="Week"),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig_weekly, use_container_width=True)

# === MAANDELIJKS OVERZICHT ===
st.markdown("""
<div class="section-header">
    <span class="icon">📅</span>
    <h2>Maandelijks Overzicht</h2>
</div>
""", unsafe_allow_html=True)

monthly = monthly_summary(filtered).reset_index()
monthly["month_str"] = monthly["month"].astype(str)

fig_monthly = go.Figure()
fig_monthly.add_trace(go.Bar(
    x=monthly["month_str"], y=monthly["km"],
    text=monthly["km"].round(1).astype(str) + " km",
    textposition="outside", textfont=dict(size=12, color="#FC4C02", family="Inter"),
    marker=dict(
        color=monthly["km"],
        colorscale=[[0, "#FF8C42"], [0.5, "#FC4C02"], [1, "#e0440e"]],
        cornerradius=8,
    ),
))
fig_monthly.update_layout(
    PLOTLY_TEMPLATE["layout"],
    xaxis=dict(title="Maand"),
    yaxis=dict(title="Kilometers"),
)
st.plotly_chart(fig_monthly, use_container_width=True)

# === HARTSLAG ANALYSE ===
if "average_heartrate" in filtered.columns and filtered["average_heartrate"].notna().any():
    st.markdown("""
    <div class="section-header">
        <span class="icon">❤️</span>
        <h2>Hartslag Analyse</h2>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    hr_data = filtered.dropna(subset=["average_heartrate"])

    with col1:
        fig_hr = px.scatter(
            hr_data, x="pace_min_per_km", y="average_heartrate",
            color="distance_km", size="distance_km",
            color_continuous_scale=[[0, "#4ECDC4"], [0.5, "#FC4C02"], [1, "#FF4757"]],
            labels={"pace_min_per_km": "Pace (min/km)", "average_heartrate": "Gem. Hartslag", "distance_km": "Afstand"},
        )
        fig_hr.update_xaxes(autorange="reversed")
        fig_hr.update_layout(PLOTLY_TEMPLATE["layout"], title="Pace vs Hartslag")
        st.plotly_chart(fig_hr, use_container_width=True)

    with col2:
        fig_hr_time = px.scatter(
            hr_data, x="start_date_local", y="average_heartrate",
            trendline="lowess",
            labels={"start_date_local": "Datum", "average_heartrate": "Gem. Hartslag"},
            color_discrete_sequence=["#FC4C02"],
            trendline_color_override="#00C48C",
        )
        fig_hr_time.update_layout(PLOTLY_TEMPLATE["layout"], title="Hartslag over Tijd")
        st.plotly_chart(fig_hr_time, use_container_width=True)

# === APPLE HEALTH: RUSTHARTSLAG ===
RESTING_HR_CSV = os.path.join(DATA_DIR, "resting_heartrate.csv")
HR_DETAIL_CSV = os.path.join(DATA_DIR, "heartrate_detail.csv")
DAILY_STEPS_CSV = os.path.join(DATA_DIR, "daily_steps.csv")
RUN_HR_DIR = os.path.join(DATA_DIR, "run_heartrates")

has_apple_data = os.path.exists(RESTING_HR_CSV)

if has_apple_data:
    st.markdown("""
    <div class="section-header">
        <span class="icon">⌚</span>
        <h2>Apple Watch Data</h2>
    </div>
    """, unsafe_allow_html=True)

    # --- Rusthartslag ---
    resting_df = pd.read_csv(RESTING_HR_CSV, parse_dates=["date"])

    col1, col2 = st.columns(2)
    with col1:
        fig_resting = go.Figure()
        fig_resting.add_trace(go.Scatter(
            x=resting_df["date"], y=resting_df["resting_hr"],
            mode="lines", name="Rusthartslag",
            line=dict(color="#FF4757", width=1.5),
            fill="tozeroy", fillcolor="rgba(255,71,87,0.1)",
        ))
        # Rolling average
        resting_df["rolling_rhr"] = resting_df["resting_hr"].rolling(14, min_periods=3).mean()
        fig_resting.add_trace(go.Scatter(
            x=resting_df["date"], y=resting_df["rolling_rhr"],
            mode="lines", name="14-daags gem.",
            line=dict(color="#00C48C", width=3),
        ))
        fig_resting.update_layout(
            PLOTLY_TEMPLATE["layout"],
            title="Rusthartslag over Tijd",
            yaxis_title="BPM",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_resting, use_container_width=True)

    with col2:
        # Rusthartslag KPI's
        recent_rhr = resting_df.tail(7)["resting_hr"].mean()
        overall_rhr = resting_df["resting_hr"].mean()
        lowest_rhr = resting_df["resting_hr"].min()
        trend_rhr = resting_df.tail(30)["resting_hr"].mean() - resting_df.tail(90).head(60)["resting_hr"].mean()

        st.markdown(f"""
        <div class="pred-card" style="margin-bottom:1rem">
            <div style="font-size:1.1rem;font-weight:700;margin-bottom:1rem">❤️ Rusthartslag</div>
            <div class="pred-week"><span>Laatste 7 dagen</span><span style="font-weight:700;color:#FC4C02">{recent_rhr:.0f} bpm</span></div>
            <div class="pred-week"><span>Algeheel gemiddeld</span><span style="font-weight:700;color:#FC4C02">{overall_rhr:.0f} bpm</span></div>
            <div class="pred-week"><span>Laagste ooit</span><span style="font-weight:700;color:#00C48C">{lowest_rhr:.0f} bpm</span></div>
            <div class="pred-week"><span>Trend (30d vs 90d)</span><span style="font-weight:700;color:{'#00C48C' if trend_rhr < 0 else '#FF4757'}">{trend_rhr:+.1f} bpm</span></div>
        </div>
        """, unsafe_allow_html=True)

        # Dagelijkse stappen
        if os.path.exists(DAILY_STEPS_CSV):
            steps_df = pd.read_csv(DAILY_STEPS_CSV, parse_dates=["date"])
            recent_steps = steps_df.tail(7)["steps"].mean()
            overall_steps = steps_df["steps"].mean()
            max_steps = steps_df["steps"].max()
            st.markdown(f"""
            <div class="pred-card">
                <div style="font-size:1.1rem;font-weight:700;margin-bottom:1rem">👟 Dagelijkse Stappen</div>
                <div class="pred-week"><span>Laatste 7 dagen</span><span style="font-weight:700;color:#FC4C02">{recent_steps:,.0f}</span></div>
                <div class="pred-week"><span>Gemiddeld</span><span style="font-weight:700;color:#FC4C02">{overall_steps:,.0f}</span></div>
                <div class="pred-week"><span>Record</span><span style="font-weight:700;color:#00C48C">{max_steps:,.0f}</span></div>
            </div>
            """, unsafe_allow_html=True)

    # --- Hartslag zones per run ---
    if os.path.exists(HR_DETAIL_CSV):
        hr_detail = pd.read_csv(HR_DETAIL_CSV)
        if not hr_detail.empty and "zone_1_pct" in hr_detail.columns:
            st.markdown("""
            <div class="section-header">
                <span class="icon">📊</span>
                <h2>Hartslag Zones per Run</h2>
            </div>
            """, unsafe_allow_html=True)

            # Sorteer op datum
            hr_detail = hr_detail.sort_values("date")
            zone_cols = ["zone_1_pct", "zone_2_pct", "zone_3_pct", "zone_4_pct", "zone_5_pct"]
            zone_names = ["Z1 Recovery", "Z2 Easy", "Z3 Tempo", "Z4 Threshold", "Z5 Max"]
            zone_colors = ["#4ECDC4", "#00C48C", "#FFEAA7", "#FF8C42", "#FF4757"]

            fig_zones = go.Figure()
            for col, name, color in zip(zone_cols, zone_names, zone_colors):
                if col in hr_detail.columns:
                    fig_zones.add_trace(go.Bar(
                        x=hr_detail["date"], y=hr_detail[col],
                        name=name, marker_color=color,
                    ))
            fig_zones.update_layout(
                PLOTLY_TEMPLATE["layout"],
                barmode="stack",
                yaxis_title="% van run",
                xaxis_title="Datum",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_zones, use_container_width=True)

            # Gedetailleerde hartslag voor individuele run
            st.markdown("**📈 Hartslag per Run (selecteer een run):**")
            run_options = hr_detail[["date", "name", "distance_km", "hr_avg", "run_id"]].copy()
            run_options["label"] = run_options.apply(
                lambda r: f"{r['date']} - {r['name']} ({r['distance_km']:.1f}km, {r['hr_avg']:.0f}bpm)", axis=1
            )
            selected_run = st.selectbox("Kies een run:", run_options["label"].tolist()[::-1], key="run_hr_select")

            if selected_run:
                selected_row = run_options[run_options["label"] == selected_run].iloc[0]
                run_hr_file = os.path.join(RUN_HR_DIR, f"run_{int(selected_row['run_id'])}.csv")
                if os.path.exists(run_hr_file):
                    run_hr_data = pd.read_csv(run_hr_file)
                    fig_run_hr = go.Figure()
                    fig_run_hr.add_trace(go.Scatter(
                        x=run_hr_data["minutes_elapsed"], y=run_hr_data["value"],
                        mode="lines", name="Hartslag",
                        line=dict(color="#FF4757", width=2),
                        fill="tozeroy", fillcolor="rgba(255,71,87,0.15)",
                    ))
                    # Rolling average
                    if len(run_hr_data) > 10:
                        run_hr_data["smooth"] = run_hr_data["value"].rolling(10, min_periods=1).mean()
                        fig_run_hr.add_trace(go.Scatter(
                            x=run_hr_data["minutes_elapsed"], y=run_hr_data["smooth"],
                            mode="lines", name="Smoothed",
                            line=dict(color="#FC4C02", width=3),
                        ))
                    fig_run_hr.update_layout(
                        PLOTLY_TEMPLATE["layout"],
                        title=f"Hartslag: {selected_row['name']}",
                        xaxis_title="Minuten",
                        yaxis_title="BPM",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    )
                    st.plotly_chart(fig_run_hr, use_container_width=True, key="individual_run_hr")

    # --- Kilometer Splits ---
    APPLE_SPLITS_CSV = os.path.join(DATA_DIR, "apple_splits.csv")
    APPLE_RUNS_CSV = os.path.join(DATA_DIR, "apple_runs.csv")

    if os.path.exists(APPLE_SPLITS_CSV) and os.path.exists(APPLE_RUNS_CSV):
        splits_df = pd.read_csv(APPLE_SPLITS_CSV)
        apple_runs = pd.read_csv(APPLE_RUNS_CSV, parse_dates=["start_time"])

        if not splits_df.empty:
            st.markdown("""
            <div class="section-header">
                <span class="icon">🔀</span>
                <h2>Kilometer Splits</h2>
            </div>
            """, unsafe_allow_html=True)

            # Run selectie voor splits
            apple_runs_sorted = apple_runs.sort_values("start_time", ascending=False)
            apple_runs_sorted["label"] = apple_runs_sorted.apply(
                lambda r: f"{r['date']} - {r['distance_km']:.1f}km in {r['pace_min_per_km']:.2f}/km", axis=1
            )
            selected_split_run = st.selectbox(
                "Kies een run voor splits:", apple_runs_sorted["label"].tolist(), key="split_select"
            )

            if selected_split_run:
                sel_row = apple_runs_sorted[apple_runs_sorted["label"] == selected_split_run].iloc[0]
                run_splits = splits_df[splits_df["date"] == sel_row["date"]].copy()

                if not run_splits.empty:
                    run_splits["pace_formatted"] = run_splits["split_pace_min"].apply(
                        lambda x: f"{int(x)}:{int((x - int(x)) * 60):02d}"
                    )
                    avg_pace = run_splits["split_pace_min"].mean()

                    # Kleur splits: sneller dan gem = groen, langzamer = rood
                    colors = ["#00C48C" if p < avg_pace else "#FF4757" for p in run_splits["split_pace_min"]]

                    fig_splits = go.Figure(go.Bar(
                        x=run_splits["km"],
                        y=run_splits["split_pace_min"],
                        text=run_splits["pace_formatted"],
                        textposition="outside",
                        textfont=dict(size=12, family="Inter"),
                        marker=dict(color=colors, cornerradius=6),
                    ))
                    # Gemiddelde lijn
                    fig_splits.add_hline(
                        y=avg_pace, line_dash="dash", line_color="#FC4C02", line_width=2,
                        annotation_text=f"Gem: {int(avg_pace)}:{int((avg_pace - int(avg_pace)) * 60):02d}/km",
                        annotation_position="top right",
                        annotation_font_color="#FC4C02",
                    )
                    fig_splits.update_layout(
                        PLOTLY_TEMPLATE["layout"],
                        xaxis_title="Kilometer", yaxis_title="Pace (min/km)",
                        xaxis=dict(dtick=1),
                    )
                    fig_splits.update_yaxes(autorange="reversed")
                    st.plotly_chart(fig_splits, use_container_width=True, key="splits_chart")

    st.markdown("<br>", unsafe_allow_html=True)

# === WANNEER LOOP JE ===
st.markdown("""
<div class="section-header">
    <span class="icon">🕐</span>
    <h2>Wanneer loop je?</h2>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_labels = ["Ma", "Di", "Wo", "Do", "Vr", "Za", "Zo"]
    day_counts = filtered["day_of_week"].value_counts().reindex(day_order).fillna(0)

    fig_day = go.Figure(go.Bar(
        x=day_labels, y=day_counts.values,
        marker=dict(
            color=day_counts.values,
            colorscale=[[0, "rgba(252,76,2,0.3)"], [1, "#FC4C02"]],
            cornerradius=8,
        ),
        text=day_counts.values.astype(int),
        textposition="outside",
        textfont=dict(color="#FC4C02", family="Inter", size=13),
    ))
    fig_day.update_layout(
        PLOTLY_TEMPLATE["layout"],
        title="Per dag", xaxis_title="Dag", yaxis_title="Runs",
    )
    st.plotly_chart(fig_day, use_container_width=True)

with col2:
    filtered_h = filtered.copy()
    filtered_h["hour"] = filtered_h["start_date_local"].dt.hour
    hour_counts = filtered_h["hour"].value_counts().sort_index().reindex(range(24), fill_value=0)

    fig_hour = go.Figure(go.Bar(
        x=hour_counts.index, y=hour_counts.values,
        marker=dict(
            color=hour_counts.values,
            colorscale=[[0, "rgba(252,76,2,0.2)"], [1, "#FC4C02"]],
            cornerradius=6,
        ),
        text=[str(v) if v > 0 else "" for v in hour_counts.values],
        textposition="outside",
        textfont=dict(color="#FC4C02", family="Inter", size=11),
    ))
    fig_hour.update_layout(
        PLOTLY_TEMPLATE["layout"],
        title="Per uur", xaxis_title="Uur", yaxis_title="Runs",
        xaxis=dict(dtick=2),
    )
    st.plotly_chart(fig_hour, use_container_width=True)

# === VOORSPELLINGEN ===
st.markdown("""
<div class="section-header">
    <span class="icon">🔮</span>
    <h2>Voorspellingen</h2>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    try:
        pace_pred = predict_pace(filtered)
        trend_icon = "📈" if pace_pred["trend"] == "verbetering" else "📉"
        trend_class = "trend-up" if pace_pred["trend"] == "verbetering" else "trend-down"

        weeks_html = ""
        for p in pace_pred["voorspellingen"]:
            weeks_html += f'<div class="pred-week"><span>Week +{p["week"]}</span><span style="font-weight:700;color:#FC4C02">{p["voorspelde_pace"]}/km</span></div>'

        st.markdown(f"""
        <div class="pred-card">
            <div style="font-size:1.1rem;font-weight:700;margin-bottom:1rem">⚡ Pace Voorspelling</div>
            <div style="font-size:0.85rem;color:#8892b0;margin-bottom:0.3rem">Trend</div>
            <div class="{trend_class}" style="font-size:1.3rem;font-weight:800;margin-bottom:1rem">{trend_icon} {pace_pred['trend'].title()}</div>
            <div style="font-size:0.85rem;color:#8892b0;margin-bottom:0.3rem">Huidige gem. pace (laatste 10)</div>
            <div style="font-size:1.3rem;font-weight:700;color:#FC4C02;margin-bottom:1rem">{pace_pred['huidige_gem_pace']}/km</div>
            <div style="font-size:0.85rem;color:#8892b0;margin-bottom:0.5rem">Komende weken</div>
            {weeks_html}
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.info(f"Niet genoeg data voor pace voorspelling.")

with col2:
    try:
        vol_pred = predict_weekly_volume(filtered)
        trend_icon = "📈" if vol_pred["trend"] == "stijgend" else "📉"
        trend_class = "trend-up" if vol_pred["trend"] == "stijgend" else "trend-down"

        weeks_html = ""
        for p in vol_pred["voorspellingen"]:
            weeks_html += f'<div class="pred-week"><span>Week +{p["week"]}</span><span style="font-weight:700;color:#FC4C02">{p["voorspelde_km"]} km</span></div>'

        st.markdown(f"""
        <div class="pred-card">
            <div style="font-size:1.1rem;font-weight:700;margin-bottom:1rem">📏 Volume Voorspelling</div>
            <div style="font-size:0.85rem;color:#8892b0;margin-bottom:0.3rem">Trend</div>
            <div class="{trend_class}" style="font-size:1.3rem;font-weight:800;margin-bottom:1rem">{trend_icon} {vol_pred['trend'].title()}</div>
            <div style="font-size:0.85rem;color:#8892b0;margin-bottom:0.3rem">Huidig gem. km/week</div>
            <div style="font-size:1.3rem;font-weight:700;color:#FC4C02;margin-bottom:1rem">{vol_pred['huidig_gem_km_per_week']} km</div>
            <div style="font-size:0.85rem;color:#8892b0;margin-bottom:0.5rem">Komende weken</div>
            {weeks_html}
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.info(f"Niet genoeg data voor volume voorspelling.")

st.markdown("<br>", unsafe_allow_html=True)

# === TRAININGSBELASTING ===
st.markdown("""
<div class="section-header">
    <span class="icon">🏋️</span>
    <h2>Trainingsbelasting</h2>
</div>
""", unsafe_allow_html=True)

try:
    tl = training_load(filtered)
    if "status" in tl:
        st.info(tl["status"])
    else:
        ratio = tl["acute_chronic_ratio"]
        if ratio <= 1.3:
            ratio_class = "load-good"
            ratio_emoji = "✅"
        elif ratio <= 1.5:
            ratio_class = "load-warning"
            ratio_emoji = "⚠️"
        else:
            ratio_class = "load-danger"
            ratio_emoji = "🚨"

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="load-card">
                <div style="font-size:0.8rem;color:#8892b0;text-transform:uppercase;letter-spacing:1.5px;font-weight:600">Acute (deze week)</div>
                <div style="font-size:2.5rem;font-weight:800;color:#FC4C02;margin:0.5rem 0">{tl['acute_km']} km</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="load-card">
                <div style="font-size:0.8rem;color:#8892b0;text-transform:uppercase;letter-spacing:1.5px;font-weight:600">Chronic (gemiddeld)</div>
                <div style="font-size:2.5rem;font-weight:800;color:#4ECDC4;margin:0.5rem 0">{tl['chronic_km']} km</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="load-card">
                <div style="font-size:0.8rem;color:#8892b0;text-transform:uppercase;letter-spacing:1.5px;font-weight:600">A/C Ratio</div>
                <div class="load-ratio {ratio_class}" style="margin:0.5rem 0">{ratio_emoji} {ratio}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background:linear-gradient(145deg,#1a1a2e,#16213e);border:1px solid rgba(255,255,255,0.06);border-radius:12px;padding:1rem 1.5rem;margin-top:1rem">
            <span style="font-weight:600">💡 Advies:</span> {tl['advies']}
        </div>
        """, unsafe_allow_html=True)

except Exception as e:
    st.info(f"Kan trainingsbelasting niet berekenen: {e}")

st.markdown("<br>", unsafe_allow_html=True)

# === PERSOONLIJKE RECORDS ===
st.markdown("""
<div class="section-header">
    <span class="icon">🏆</span>
    <h2>Persoonlijke Records</h2>
</div>
""", unsafe_allow_html=True)

try:
    prs = personal_records(filtered)
    trophy_icons = {"snelste_pace": "⚡", "langste_run": "📏", "meeste_hoogtemeters": "⛰️"}
    trophy_titles = {"snelste_pace": "Snelste Pace", "langste_run": "Langste Run", "meeste_hoogtemeters": "Meeste Hoogtemeters"}

    cols = st.columns(3)
    for i, (key, val) in enumerate(prs.items()):
        icon = trophy_icons.get(key, "🏆")
        title = trophy_titles.get(key, key)

        # Bepaal hoofdwaarde
        if key == "snelste_pace":
            main_val = val.get("pace", "")
            main_suffix = "/km"
        elif key == "langste_run":
            main_val = f"{val.get('afstand_km', '')} km"
            main_suffix = ""
        else:
            main_val = f"{val.get('hoogtemeters', '')}m"
            main_suffix = ""

        details = f"📅 {val.get('datum', '')}<br>🏷️ {val.get('naam', '')}"

        with cols[i]:
            st.markdown(f"""
            <div class="record-card">
                <span class="trophy">{icon}</span>
                <div class="record-label">{title}</div>
                <div class="record-value">{main_val}{main_suffix}</div>
                <div class="record-detail">{details}</div>
            </div>
            """, unsafe_allow_html=True)
except Exception as e:
    st.info(f"Kan records niet laden: {e}")

st.markdown("<br>", unsafe_allow_html=True)

# === STEL EEN VRAAG ===
st.markdown("""
<div class="section-header">
    <span class="icon">💬</span>
    <h2>Stel een vraag over je data</h2>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="chat-header">
    <div style="font-size:0.9rem;color:#8892b0;line-height:1.6">
        Probeer: <span style="color:#FC4C02;font-weight:500">"Wat zijn mijn snelste 5km runs?"</span> ·
        <span style="color:#FC4C02;font-weight:500">"Hoe is mijn progressie?"</span> ·
        <span style="color:#FC4C02;font-weight:500">"Vergelijk de laatste twee maanden"</span> ·
        <span style="color:#FC4C02;font-weight:500">"Geef me trainingsadvies"</span>
    </div>
</div>
""", unsafe_allow_html=True)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for i, msg in enumerate(st.session_state.chat_history):
    with st.chat_message(msg["role"], avatar="🏃" if msg["role"] == "user" else "📊"):
        st.markdown(msg["content"])
        if "fig" in msg and msg["fig"] is not None:
            st.plotly_chart(msg["fig"], use_container_width=True, key=f"chat_hist_{i}")

if prompt := st.chat_input("Stel een vraag over je runs..."):
    msg_idx = len(st.session_state.chat_history)
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🏃"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="📊"):
        result = answer_question(prompt, filtered)
        st.markdown(result["text"])
        fig = result.get("fig")
        if fig is not None:
            if hasattr(fig, "update_layout"):
                fig.update_layout(PLOTLY_TEMPLATE["layout"])
            st.plotly_chart(fig, use_container_width=True, key=f"chat_new_{msg_idx}")

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": result["text"],
        "fig": result.get("fig"),
    })

# --- Ruwe data ---
st.markdown("<br>", unsafe_allow_html=True)
with st.expander("📋 Bekijk ruwe data"):
    st.dataframe(filtered.sort_values("start_date_local", ascending=False), use_container_width=True)
