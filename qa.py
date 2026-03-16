"""
AI-gestuurde Vraag & Antwoord module voor Strava run data.
Gebruikt Claude (Anthropic) voor natuurlijke gesprekken over je trainingsdata.
"""
import os
import re
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from anthropic import Anthropic

from analyze import format_pace

ENV_FILE = os.path.join(os.path.dirname(__file__), ".env")


def _load_api_key():
    """Laad Anthropic API key uit Streamlit secrets of .env."""
    # Probeer eerst Streamlit secrets (voor Cloud deployment)
    try:
        key = st.secrets.get("ANTHROPIC_API_KEY")
        if key:
            return key
    except Exception:
        pass

    # Fallback naar .env bestand (lokaal)
    if not os.path.exists(ENV_FILE):
        return None
    with open(ENV_FILE) as f:
        for line in f:
            line = line.strip()
            if line.startswith("ANTHROPIC_API_KEY="):
                return line.split("=", 1)[1].strip()
    return None


def _build_data_summary(df):
    """Bouw een compacte data-samenvatting voor Claude's context."""
    total_runs = len(df)
    if total_runs == 0:
        return "Geen runs gevonden met huidige filters."

    # Algemene stats
    summary = f"""## Run Data Samenvatting
- Totaal: {total_runs} runs, {df['distance_km'].sum():.1f} km, {df['moving_time_min'].sum()/60:.1f} uur
- Periode: {df['start_date_local'].min().strftime('%d-%m-%Y')} tot {df['start_date_local'].max().strftime('%d-%m-%Y')}
- Gem. afstand: {df['distance_km'].mean():.1f} km | Gem. pace: {format_pace(df['pace_min_per_km'].mean())}/km
- Snelste pace: {format_pace(df['pace_min_per_km'].min())}/km | Langzaamste: {format_pace(df['pace_min_per_km'].max())}/km
- Langste run: {df['distance_km'].max():.1f} km | Kortste: {df['distance_km'].min():.1f} km
- Totaal hoogtemeters: {df['total_elevation_gain'].sum():.0f}m
"""

    if df["average_heartrate"].notna().any():
        hr = df.dropna(subset=["average_heartrate"])
        summary += f"- Gem. hartslag: {hr['average_heartrate'].mean():.0f} bpm | Min: {hr['average_heartrate'].min():.0f} | Max: {hr['average_heartrate'].max():.0f}\n"

    # Laatste 10 runs
    summary += "\n## Laatste 10 runs\n"
    for _, row in df.sort_values("start_date_local", ascending=False).head(10).iterrows():
        hr = f" | {int(row['average_heartrate'])}bpm" if pd.notna(row.get("average_heartrate")) else ""
        summary += f"- {row['date']}: {row['distance_km']:.1f}km, {format_pace(row['pace_min_per_km'])}/km{hr} - \"{row['name']}\"\n"

    # Top 5 snelste
    summary += "\n## Top 5 snelste runs (op pace)\n"
    for _, row in df.nsmallest(5, "pace_min_per_km").iterrows():
        summary += f"- {row['date']}: {row['distance_km']:.1f}km, {format_pace(row['pace_min_per_km'])}/km - \"{row['name']}\"\n"

    # Top 5 langste
    summary += "\n## Top 5 langste runs\n"
    for _, row in df.nlargest(5, "distance_km").iterrows():
        summary += f"- {row['date']}: {row['distance_km']:.1f}km, {format_pace(row['pace_min_per_km'])}/km - \"{row['name']}\"\n"

    # Maandelijks overzicht
    df_copy = df.copy()
    df_copy["year_month"] = df_copy["start_date_local"].dt.to_period("M")
    monthly = df_copy.groupby("year_month").agg(
        runs=("id", "count"),
        km=("distance_km", "sum"),
        gem_pace=("pace_min_per_km", "mean"),
    )
    summary += "\n## Maandelijks overzicht\n"
    for period, row in monthly.iterrows():
        summary += f"- {period}: {int(row['runs'])} runs, {row['km']:.1f}km, gem. {format_pace(row['gem_pace'])}/km\n"

    # Wekelijks (laatste 6 weken)
    df_copy["year_week"] = df_copy["start_date_local"].dt.to_period("W")
    weekly = df_copy.groupby("year_week").agg(
        runs=("id", "count"),
        km=("distance_km", "sum"),
        gem_pace=("pace_min_per_km", "mean"),
    )
    summary += "\n## Laatste 6 weken\n"
    for period, row in weekly.tail(6).iterrows():
        summary += f"- {period}: {int(row['runs'])} runs, {row['km']:.1f}km, gem. {format_pace(row['gem_pace'])}/km\n"

    # Trainingsbelasting
    if len(weekly) >= 2:
        acute = weekly.iloc[-1]["km"]
        chronic = weekly.iloc[:-1].tail(4)["km"].mean()
        ratio = acute / chronic if chronic > 0 else 0
        summary += f"\n## Trainingsbelasting\n"
        summary += f"- Acute (deze week): {acute:.1f} km\n"
        summary += f"- Chronic (gem. voorgaande 4 weken): {chronic:.1f} km\n"
        summary += f"- Acute/Chronic ratio: {ratio:.2f}\n"

    # Afstandsverdeling
    summary += "\n## Afstandsverdeling\n"
    bins = [(0, 5, "0-5km"), (5, 10, "5-10km"), (10, 15, "10-15km"), (15, 21, "15-21km"), (21, 100, "21km+")]
    for low, high, label in bins:
        count = len(df[(df["distance_km"] >= low) & (df["distance_km"] < high)])
        if count > 0:
            subset = df[(df["distance_km"] >= low) & (df["distance_km"] < high)]
            summary += f"- {label}: {count} runs, gem. pace {format_pace(subset['pace_min_per_km'].mean())}/km\n"

    # Dag/uur verdeling
    summary += "\n## Looptijden\n"
    day_counts = df["day_of_week"].value_counts()
    summary += f"- Populairste dag: {day_counts.index[0]} ({day_counts.iloc[0]}x)\n"
    df_h = df.copy()
    df_h["hour"] = df_h["start_date_local"].dt.hour
    hour_counts = df_h["hour"].value_counts()
    summary += f"- Populairste uur: {hour_counts.index[0]}:00 ({hour_counts.iloc[0]}x)\n"

    # Pace progressie
    first_10 = df.head(10)["pace_min_per_km"].mean()
    last_10 = df.tail(10)["pace_min_per_km"].mean()
    improvement = (first_10 - last_10) * 60
    summary += f"\n## Progressie\n"
    summary += f"- Eerste 10 runs gem. pace: {format_pace(first_10)}/km\n"
    summary += f"- Laatste 10 runs gem. pace: {format_pace(last_10)}/km\n"
    summary += f"- Verschil: {improvement:+.0f} seconden/km ({'sneller' if improvement > 0 else 'langzamer'})\n"

    # Apple Health data (als beschikbaar)
    resting_hr_file = os.path.join(os.path.dirname(__file__), "data", "resting_heartrate.csv")
    hr_detail_file = os.path.join(os.path.dirname(__file__), "data", "heartrate_detail.csv")
    steps_file = os.path.join(os.path.dirname(__file__), "data", "daily_steps.csv")

    if os.path.exists(resting_hr_file):
        rhr = pd.read_csv(resting_hr_file, parse_dates=["date"])
        summary += f"\n## Apple Watch: Rusthartslag\n"
        summary += f"- Huidige (7d gem.): {rhr.tail(7)['resting_hr'].mean():.0f} bpm\n"
        summary += f"- Algeheel gem.: {rhr['resting_hr'].mean():.0f} bpm\n"
        summary += f"- Laagste: {rhr['resting_hr'].min():.0f} bpm\n"
        trend = rhr.tail(30)['resting_hr'].mean() - rhr.tail(90).head(60)['resting_hr'].mean()
        summary += f"- Trend (30d vs 90d): {trend:+.1f} bpm\n"

    if os.path.exists(hr_detail_file):
        hrd = pd.read_csv(hr_detail_file)
        if not hrd.empty:
            summary += f"\n## Apple Watch: Hartslag per Run (laatste 5)\n"
            for _, row in hrd.tail(5).iterrows():
                zones = ""
                for z in range(1, 6):
                    col = f"zone_{z}_pct"
                    if col in row and pd.notna(row[col]):
                        zones += f"Z{z}:{row[col]:.0f}% "
                summary += f"- {row['date']}: avg {row['hr_avg']:.0f} max {row['hr_max']:.0f} bpm | {zones}- {row['name']}\n"

    if os.path.exists(steps_file):
        steps = pd.read_csv(steps_file, parse_dates=["date"])
        summary += f"\n## Apple Watch: Stappen\n"
        summary += f"- Gem. dagelijks: {steps['steps'].mean():,.0f}\n"
        summary += f"- Laatste 7 dagen: {steps.tail(7)['steps'].mean():,.0f}\n"

    return summary


def _pick_chart(question, df):
    """Kies een relevante grafiek op basis van de vraag."""
    q = question.lower()

    # Pace gerelateerd
    if any(w in q for w in ["pace", "snelste", "fastest", "tempo", "snel", "speed", "pr", "record", "sub"]):
        fig = px.scatter(
            df.dropna(subset=["pace_min_per_km"]),
            x="start_date_local", y="pace_min_per_km",
            color="distance_km", size="distance_km",
            color_continuous_scale=[[0, "#4ECDC4"], [0.5, "#FC4C02"], [1, "#FF4757"]],
            labels={"pace_min_per_km": "Pace (min/km)", "start_date_local": "Datum", "distance_km": "Afstand (km)"},
        )
        fig.update_yaxes(autorange="reversed")
        return fig

    # Hartslag
    if any(w in q for w in ["hartslag", "heart", "hr", "bpm", "cardiac", "hart"]):
        hr_data = df.dropna(subset=["average_heartrate"])
        if not hr_data.empty:
            fig = px.scatter(
                hr_data, x="pace_min_per_km", y="average_heartrate",
                color="distance_km", size="distance_km",
                color_continuous_scale=[[0, "#4ECDC4"], [0.5, "#FC4C02"], [1, "#FF4757"]],
                labels={"pace_min_per_km": "Pace (min/km)", "average_heartrate": "Hartslag (bpm)"},
            )
            fig.update_xaxes(autorange="reversed")
            return fig

    # Volume / km / week / maand
    if any(w in q for w in ["volume", "kilometer", "km", "week", "maand", "month", "afstand", "distance"]):
        df_copy = df.copy()
        df_copy["year_week"] = df_copy["start_date_local"].dt.to_period("W")
        weekly = df_copy.groupby("year_week")["distance_km"].sum().reset_index()
        weekly["year_week_str"] = weekly["year_week"].astype(str)
        fig = go.Figure(go.Bar(
            x=weekly["year_week_str"], y=weekly["distance_km"],
            marker=dict(color=weekly["distance_km"], colorscale=[[0, "#FF8C42"], [1, "#FC4C02"]], cornerradius=6),
        ))
        fig.update_layout(xaxis_title="Week", yaxis_title="Kilometers")
        return fig

    # Progressie / ontwikkeling / verbetering
    if any(w in q for w in ["progressie", "progress", "ontwikkel", "verbeter", "improv", "trend"]):
        df_sorted = df.sort_values("start_date_local").dropna(subset=["pace_min_per_km"]).copy()
        df_sorted["rolling_pace"] = df_sorted["pace_min_per_km"].rolling(10, min_periods=3).mean()
        fig = px.scatter(df_sorted, x="start_date_local", y="pace_min_per_km",
                         labels={"pace_min_per_km": "Pace (min/km)", "start_date_local": "Datum"},
                         color_discrete_sequence=["#FC4C02"])
        fig.add_trace(go.Scatter(x=df_sorted["start_date_local"], y=df_sorted["rolling_pace"],
                                 mode="lines", name="Trend (10-run)", line=dict(color="#00C48C", width=3)))
        fig.update_yaxes(autorange="reversed")
        return fig

    # Hoogtemeters
    if any(w in q for w in ["hoogte", "elevation", "stijg", "heuvel", "hill", "climb"]):
        fig = px.scatter(df, x="total_elevation_gain", y="pace_min_per_km",
                         size="distance_km", hover_name="name",
                         color_discrete_sequence=["#FC4C02"],
                         labels={"total_elevation_gain": "Hoogtemeters (m)", "pace_min_per_km": "Pace (min/km)"})
        fig.update_yaxes(autorange="reversed")
        return fig

    # Wanneer / dag / tijd
    if any(w in q for w in ["wanneer", "when", "dag", "day", "tijd", "time"]):
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day_labels = ["Ma", "Di", "Wo", "Do", "Vr", "Za", "Zo"]
        day_counts = df["day_of_week"].value_counts().reindex(day_order).fillna(0)
        fig = go.Figure(go.Bar(
            x=day_labels, y=day_counts.values,
            marker=dict(color=day_counts.values, colorscale=[[0, "rgba(252,76,2,0.3)"], [1, "#FC4C02"]], cornerradius=8),
        ))
        fig.update_layout(xaxis_title="Dag", yaxis_title="Runs")
        return fig

    # 5km / 10km / specifieke afstand
    if any(w in q for w in ["5km", "5k", "10km", "10k", "15km", "21km", "halve marathon", "marathon"]):
        fig = px.scatter(
            df.dropna(subset=["pace_min_per_km"]),
            x="start_date_local", y="pace_min_per_km",
            color="distance_km", size="distance_km",
            color_continuous_scale=[[0, "#4ECDC4"], [0.5, "#FC4C02"], [1, "#FF4757"]],
            labels={"pace_min_per_km": "Pace (min/km)", "start_date_local": "Datum"},
        )
        fig.update_yaxes(autorange="reversed")
        return fig

    # Default: pace over tijd
    fig = px.scatter(
        df.dropna(subset=["pace_min_per_km"]),
        x="start_date_local", y="pace_min_per_km",
        color="distance_km", size="distance_km",
        color_continuous_scale=[[0, "#4ECDC4"], [0.5, "#FC4C02"], [1, "#FF4757"]],
        labels={"pace_min_per_km": "Pace (min/km)", "start_date_local": "Datum", "distance_km": "Afstand (km)"},
    )
    fig.update_yaxes(autorange="reversed")
    return fig


def answer_question(question, df):
    """
    Beantwoord een vraag met Claude AI + data context.
    Fallback naar keyword-systeem als API niet beschikbaar is.
    """
    api_key = _load_api_key()
    if not api_key:
        return _keyword_fallback(question, df)

    try:
        client = Anthropic(api_key=api_key)
        data_summary = _build_data_summary(df)

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1500,
            system=f"""Je bent een hardloop-coach en data-analist die een loper helpt met hun Strava data.
Je spreekt Nederlands. Je bent enthousiast maar realistisch.
Je geeft concrete, data-gedreven antwoorden. Gebruik getallen uit de data.
Gebruik markdown formatting (bold, lijstjes).
Houd antwoorden beknopt maar informatief (max ~200 woorden).
Geef bij voorspellingen altijd een disclaimer dat het schattingen zijn.
Bij trainingsadvies, baseer je op bewezen principes (80/20 regel, 10% regel, periodisering).
Vandaag is {pd.Timestamp.now().strftime('%d-%m-%Y')}.

{data_summary}""",
            messages=[{"role": "user", "content": question}],
        )

        text = response.content[0].text
        fig = _pick_chart(question, df)
        return {"text": text, "fig": fig}

    except Exception as e:
        return {"text": f"**AI fout:** {e}\n\nFallback naar basis-analyse...\n\n" + _keyword_fallback(question, df)["text"]}


def _keyword_fallback(question, df):
    """Keyword-based fallback wanneer Claude API niet beschikbaar is."""
    q = question.lower().strip()

    if any(w in q for w in ["snelste", "fastest", "beste pace"]):
        top = df.nsmallest(5, "pace_min_per_km")[["name", "date", "distance_km", "pace_min_per_km"]]
        text = "**Top 5 snelste runs:**\n\n"
        for _, row in top.iterrows():
            text += f"- **{format_pace(row['pace_min_per_km'])}**/km - {row['distance_km']:.1f}km - {row['name']} ({row['date']})\n"
        return {"text": text}

    if any(w in q for w in ["langste", "longest", "verste"]):
        top = df.nlargest(5, "distance_km")[["name", "date", "distance_km", "pace_min_per_km"]]
        text = "**Top 5 langste runs:**\n\n"
        for _, row in top.iterrows():
            text += f"- **{row['distance_km']:.1f}km** - {format_pace(row['pace_min_per_km'])}/km - {row['name']} ({row['date']})\n"
        return {"text": text}

    text = f"Je hebt **{len(df)} runs** over **{df['distance_km'].sum():.1f} km**.\n\n"
    text += "*Claude AI is niet beschikbaar. Voeg ANTHROPIC_API_KEY toe aan .env voor slimme antwoorden.*"
    return {"text": text}
