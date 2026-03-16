"""
Haal alle activiteiten op van Strava en sla ze op als JSON + CSV.
"""
import json
import os
import time

import pandas as pd
import requests

from auth import get_access_token

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
ACTIVITIES_JSON = os.path.join(DATA_DIR, "activities.json")
RUNS_CSV = os.path.join(DATA_DIR, "runs.csv")
ALL_CSV = os.path.join(DATA_DIR, "all_activities.csv")

API_BASE = "https://www.strava.com/api/v3"


def fetch_all_activities(access_token):
    """Haal alle activiteiten op (paginated)."""
    all_activities = []
    page = 1
    per_page = 200

    while True:
        print(f"Ophalen pagina {page}...")
        response = requests.get(
            f"{API_BASE}/athlete/activities",
            headers={"Authorization": f"Bearer {access_token}"},
            params={"page": page, "per_page": per_page},
        )
        response.raise_for_status()
        activities = response.json()

        if not activities:
            break

        all_activities.extend(activities)
        page += 1
        time.sleep(1)  # Rate limiting respecteren

    return all_activities


def process_activities(activities):
    """Verwerk ruwe activiteiten naar een schoon DataFrame."""
    df = pd.DataFrame(activities)

    # Relevante kolommen selecteren
    columns = [
        "id", "name", "type", "sport_type", "start_date_local",
        "distance", "moving_time", "elapsed_time", "total_elevation_gain",
        "average_speed", "max_speed", "average_heartrate", "max_heartrate",
        "average_cadence", "suffer_score", "kudos_count",
        "start_latlng", "end_latlng",
    ]
    # Alleen kolommen die bestaan
    columns = [c for c in columns if c in df.columns]
    df = df[columns].copy()

    # Conversies
    df["start_date_local"] = pd.to_datetime(df["start_date_local"])
    df["distance_km"] = df["distance"] / 1000
    df["moving_time_min"] = df["moving_time"] / 60
    df["elapsed_time_min"] = df["elapsed_time"] / 60

    # Pace berekenen (min/km) voor runs
    df["pace_min_per_km"] = df.apply(
        lambda row: row["moving_time_min"] / row["distance_km"]
        if row["distance_km"] > 0 else None,
        axis=1,
    )

    # Speed in km/h
    df["speed_kmh"] = df["average_speed"] * 3.6

    df = df.sort_values("start_date_local").reset_index(drop=True)
    return df


def save_data(activities, df):
    """Sla data op in verschillende formaten."""
    os.makedirs(DATA_DIR, exist_ok=True)

    # Ruwe JSON
    with open(ACTIVITIES_JSON, "w") as f:
        json.dump(activities, f, indent=2, default=str)

    # Alle activiteiten CSV
    df.to_csv(ALL_CSV, index=False)

    # Alleen runs CSV
    runs = df[df["type"] == "Run"].copy()
    runs.to_csv(RUNS_CSV, index=False)

    print(f"\nData opgeslagen:")
    print(f"  Totaal activiteiten: {len(df)}")
    print(f"  Runs: {len(runs)}")
    print(f"  Bestanden: {DATA_DIR}/")


def main():
    token = get_access_token()
    activities = fetch_all_activities(token)

    if not activities:
        print("Geen activiteiten gevonden.")
        return

    df = process_activities(activities)
    save_data(activities, df)
    return df


if __name__ == "__main__":
    main()
