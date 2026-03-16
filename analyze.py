"""
Run-analyse module: statistieken, trends en voorspellingen.
"""
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RUNS_CSV = os.path.join(DATA_DIR, "runs.csv")


def load_runs():
    """Laad runs data."""
    df = pd.read_csv(RUNS_CSV, parse_dates=["start_date_local"])
    df["week"] = df["start_date_local"].dt.isocalendar().week.astype(int)
    df["month"] = df["start_date_local"].dt.to_period("M")
    df["year"] = df["start_date_local"].dt.year
    df["day_of_week"] = df["start_date_local"].dt.day_name()
    df["date"] = df["start_date_local"].dt.date
    return df


def summary_stats(df):
    """Algemene statistieken."""
    return {
        "totaal_runs": len(df),
        "totaal_km": round(df["distance_km"].sum(), 1),
        "totaal_uren": round(df["moving_time_min"].sum() / 60, 1),
        "gem_afstand_km": round(df["distance_km"].mean(), 1),
        "gem_pace": format_pace(df["pace_min_per_km"].mean()),
        "snelste_pace": format_pace(df["pace_min_per_km"].min()),
        "langste_run_km": round(df["distance_km"].max(), 1),
        "totaal_hoogtemeters": round(df["total_elevation_gain"].sum(), 0),
        "gem_hartslag": round(df["average_heartrate"].mean(), 0) if "average_heartrate" in df and df["average_heartrate"].notna().any() else "N/A",
    }


def format_pace(pace_float):
    """Converteer pace float naar mm:ss formaat."""
    if pd.isna(pace_float):
        return "N/A"
    minutes = int(pace_float)
    seconds = int((pace_float - minutes) * 60)
    return f"{minutes}:{seconds:02d}"


def weekly_summary(df):
    """Wekelijkse samenvatting."""
    df = df.copy()
    df["year_week"] = df["start_date_local"].dt.to_period("W")
    weekly = df.groupby("year_week").agg(
        runs=("id", "count"),
        km=("distance_km", "sum"),
        tijd_min=("moving_time_min", "sum"),
        gem_pace=("pace_min_per_km", "mean"),
        hoogtemeters=("total_elevation_gain", "sum"),
    ).round(1)
    return weekly


def monthly_summary(df):
    """Maandelijkse samenvatting."""
    monthly = df.groupby("month").agg(
        runs=("id", "count"),
        km=("distance_km", "sum"),
        tijd_min=("moving_time_min", "sum"),
        gem_pace=("pace_min_per_km", "mean"),
        hoogtemeters=("total_elevation_gain", "sum"),
    ).round(1)
    return monthly


def predict_pace(df, weeks_ahead=4):
    """Voorspel pace ontwikkeling met lineaire regressie."""
    df = df.copy()
    df = df.dropna(subset=["pace_min_per_km"])
    df["days_since_start"] = (df["start_date_local"] - df["start_date_local"].min()).dt.days

    X = df[["days_since_start"]].values
    y = df["pace_min_per_km"].values

    model = LinearRegression()
    model.fit(X, y)

    last_day = df["days_since_start"].max()
    future_days = np.array([[last_day + 7 * i] for i in range(1, weeks_ahead + 1)])
    predictions = model.predict(future_days)

    r_squared = model.score(X, y)

    return {
        "trend": "verbetering" if model.coef_[0] < 0 else "verslechtering",
        "pace_change_per_week": round(model.coef_[0] * 7, 3),
        "r_squared": round(r_squared, 3),
        "voorspellingen": [
            {"week": i + 1, "voorspelde_pace": format_pace(p)}
            for i, p in enumerate(predictions)
        ],
        "huidige_gem_pace": format_pace(df["pace_min_per_km"].tail(10).mean()),
    }


def predict_weekly_volume(df, weeks_ahead=4):
    """Voorspel wekelijks volume (km)."""
    weekly = weekly_summary(df).reset_index()
    weekly["week_num"] = range(len(weekly))

    X = weekly[["week_num"]].values
    y = weekly["km"].values

    model = LinearRegression()
    model.fit(X, y)

    last_week = weekly["week_num"].max()
    future_weeks = np.array([[last_week + i] for i in range(1, weeks_ahead + 1)])
    predictions = model.predict(future_weeks)

    return {
        "trend": "stijgend" if model.coef_[0] > 0 else "dalend",
        "km_change_per_week": round(model.coef_[0], 2),
        "voorspellingen": [
            {"week": i + 1, "voorspelde_km": round(max(p, 0), 1)}
            for i, p in enumerate(predictions)
        ],
        "huidig_gem_km_per_week": round(weekly["km"].tail(4).mean(), 1),
    }


def personal_records(df):
    """Vind persoonlijke records."""
    records = {}

    # Snelste pace
    fastest = df.loc[df["pace_min_per_km"].idxmin()]
    records["snelste_pace"] = {
        "pace": format_pace(fastest["pace_min_per_km"]),
        "datum": str(fastest["date"]),
        "afstand_km": round(fastest["distance_km"], 1),
        "naam": fastest["name"],
    }

    # Langste run
    longest = df.loc[df["distance_km"].idxmax()]
    records["langste_run"] = {
        "afstand_km": round(longest["distance_km"], 1),
        "datum": str(longest["date"]),
        "pace": format_pace(longest["pace_min_per_km"]),
        "naam": longest["name"],
    }

    # Meeste hoogtemeters
    most_elev = df.loc[df["total_elevation_gain"].idxmax()]
    records["meeste_hoogtemeters"] = {
        "hoogtemeters": round(most_elev["total_elevation_gain"], 0),
        "datum": str(most_elev["date"]),
        "afstand_km": round(most_elev["distance_km"], 1),
        "naam": most_elev["name"],
    }

    return records


def training_load(df, weeks=6):
    """Analyseer trainingsbelasting (laatste N weken)."""
    recent = df[df["start_date_local"] >= df["start_date_local"].max() - pd.Timedelta(weeks=weeks)].copy()
    recent["year_week"] = recent["start_date_local"].dt.to_period("W")

    weekly = recent.groupby("year_week").agg(
        km=("distance_km", "sum"),
        runs=("id", "count"),
        tijd_min=("moving_time_min", "sum"),
    ).reset_index()

    if len(weekly) < 2:
        return {"status": "Niet genoeg data voor trainingsbelasting analyse."}

    # Acute (laatste week) vs chronic (gemiddelde voorgaande weken)
    acute = weekly.iloc[-1]["km"]
    chronic = weekly.iloc[:-1]["km"].mean()
    ratio = acute / chronic if chronic > 0 else 0

    if ratio < 0.8:
        advies = "Trainingsvolume is laag. Je kunt veilig opbouwen."
    elif ratio <= 1.3:
        advies = "Goed trainingsvolume. Blijf op dit niveau."
    elif ratio <= 1.5:
        advies = "Let op: verhoogd blessurerisico door snelle opbouw."
    else:
        advies = "Waarschuwing: te snelle opbouw! Verminder volume."

    return {
        "acute_km": round(acute, 1),
        "chronic_km": round(chronic, 1),
        "acute_chronic_ratio": round(ratio, 2),
        "advies": advies,
        "wekelijks_overzicht": weekly.to_dict("records"),
    }


if __name__ == "__main__":
    runs = load_runs()
    print("\n=== SAMENVATTING ===")
    for k, v in summary_stats(runs).items():
        print(f"  {k}: {v}")

    print("\n=== PERSOONLIJKE RECORDS ===")
    for k, v in personal_records(runs).items():
        print(f"  {k}: {v}")

    print("\n=== PACE VOORSPELLING ===")
    pred = predict_pace(runs)
    for k, v in pred.items():
        print(f"  {k}: {v}")

    print("\n=== TRAININGSBELASTING ===")
    load = training_load(runs)
    for k, v in load.items():
        print(f"  {k}: {v}")
