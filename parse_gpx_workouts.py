"""
Parse Apple Health GPX workout routes en bouw workout data.
Koppelt GPS-data (afstand, pace, splits) aan hartslag uit Records.
"""
import os
import re
import math
from datetime import datetime, timedelta
from glob import glob

import pandas as pd
import numpy as np
from lxml import etree

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
GPX_DIR = os.path.join(os.path.dirname(__file__), "apple_health_export", "workout-routes")
HR_DETAIL_CSV = os.path.join(DATA_DIR, "heartrate_detail.csv")

# Output
APPLE_WORKOUTS_CSV = os.path.join(DATA_DIR, "apple_workouts.csv")
APPLE_SPLITS_CSV = os.path.join(DATA_DIR, "apple_splits.csv")


def haversine(lat1, lon1, lat2, lon2):
    """Bereken afstand in meters tussen twee GPS punten."""
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def parse_gpx(filepath):
    """Parse een GPX bestand naar trackpoints."""
    ns = {"gpx": "http://www.topografix.com/GPX/1/1"}
    tree = etree.parse(filepath)

    points = []
    for trkpt in tree.findall(".//gpx:trkpt", ns):
        lat = float(trkpt.get("lat"))
        lon = float(trkpt.get("lon"))

        ele_elem = trkpt.find("gpx:ele", ns)
        time_elem = trkpt.find("gpx:time", ns)

        ele = float(ele_elem.text) if ele_elem is not None else 0
        time_str = time_elem.text if time_elem is not None else None

        # Speed uit extensions
        speed = None
        extensions = trkpt.find("gpx:extensions", ns)
        if extensions is not None:
            speed_elem = extensions.find("gpx:speed", ns)
            if speed_elem is not None:
                speed = float(speed_elem.text)

        if time_str:
            dt = datetime.strptime(time_str[:19], "%Y-%m-%dT%H:%M:%S")
            points.append({
                "lat": lat, "lon": lon, "ele": ele,
                "datetime": dt, "speed_ms": speed,
            })

    return points


def build_workout_from_points(points, filename):
    """Bereken workout stats uit GPS trackpoints."""
    if len(points) < 10:
        return None, None

    # Tijden
    start_time = points[0]["datetime"]
    end_time = points[-1]["datetime"]
    duration_min = (end_time - start_time).total_seconds() / 60

    if duration_min < 3:  # Skip zeer korte workouts
        return None, None

    # Afstand berekenen
    total_distance = 0
    cumulative_distances = [0]
    for i in range(1, len(points)):
        d = haversine(
            points[i-1]["lat"], points[i-1]["lon"],
            points[i]["lat"], points[i]["lon"]
        )
        total_distance += d
        cumulative_distances.append(total_distance)

    distance_km = total_distance / 1000

    if distance_km < 0.5:  # Skip te korte workouts
        return None, None

    # Pace
    pace = duration_min / distance_km if distance_km > 0 else 0

    # Hoogtemeters
    elevation_gain = 0
    for i in range(1, len(points)):
        diff = points[i]["ele"] - points[i-1]["ele"]
        if diff > 0:
            elevation_gain += diff

    # Gemiddelde snelheid
    speeds = [p["speed_ms"] for p in points if p["speed_ms"] is not None and p["speed_ms"] > 0.5]
    avg_speed = np.mean(speeds) if speeds else 0
    max_speed = np.max(speeds) if speeds else 0

    # Is het een run? (gem snelheid > 2 m/s = ~7:30/km pace, < 7 m/s)
    is_run = 1.5 < avg_speed < 7.0 and pace < 10 and pace > 2.5

    # Bepaal type op basis van snelheid
    if avg_speed < 1.5:
        activity_type = "Walk"
    elif avg_speed < 7:
        activity_type = "Run"
    else:
        activity_type = "Cycle"

    workout = {
        "filename": filename,
        "start_time": start_time,
        "end_time": end_time,
        "duration_min": round(duration_min, 1),
        "distance_km": round(distance_km, 2),
        "pace_min_per_km": round(pace, 2),
        "elevation_gain": round(elevation_gain, 1),
        "avg_speed_ms": round(avg_speed, 2),
        "max_speed_ms": round(max_speed, 2),
        "avg_speed_kmh": round(avg_speed * 3.6, 1),
        "activity_type": activity_type,
        "is_run": is_run,
        "num_points": len(points),
        "date": start_time.date(),
    }

    # Splits per kilometer
    splits = []
    current_km = 1
    split_start_idx = 0
    split_start_time = points[0]["datetime"]

    for i, cum_dist in enumerate(cumulative_distances):
        if cum_dist >= current_km * 1000:
            split_time = (points[i]["datetime"] - split_start_time).total_seconds()
            split_pace = split_time / 60  # minuten voor 1 km

            # Gem snelheid in deze split
            split_speeds = [p["speed_ms"] for p in points[split_start_idx:i+1]
                          if p["speed_ms"] is not None and p["speed_ms"] > 0]
            split_avg_speed = np.mean(split_speeds) if split_speeds else 0

            splits.append({
                "start_time": start_time,
                "km": current_km,
                "split_time_sec": round(split_time, 1),
                "split_pace_min": round(split_pace, 2),
                "avg_speed_ms": round(split_avg_speed, 2),
                "date": start_time.date(),
            })

            current_km += 1
            split_start_idx = i
            split_start_time = points[i]["datetime"]

    return workout, splits


def main():
    gpx_files = sorted(glob(os.path.join(GPX_DIR, "*.gpx")))
    print(f"Gevonden: {len(gpx_files)} GPX bestanden")

    workouts = []
    all_splits = []
    errors = 0

    for i, filepath in enumerate(gpx_files):
        if (i + 1) % 50 == 0:
            print(f"  Verwerkt: {i+1}/{len(gpx_files)}...")

        try:
            filename = os.path.basename(filepath)
            points = parse_gpx(filepath)
            workout, splits = build_workout_from_points(points, filename)

            if workout:
                workouts.append(workout)
            if splits:
                all_splits.extend(splits)
        except Exception as e:
            errors += 1

    print(f"\nResultaat:")
    print(f"  Totaal workouts: {len(workouts)}")
    print(f"  Errors: {errors}")

    # Maak DataFrames
    workouts_df = pd.DataFrame(workouts)
    splits_df = pd.DataFrame(all_splits)

    # Filter alleen runs
    runs = workouts_df[workouts_df["is_run"]].copy()
    other = workouts_df[~workouts_df["is_run"]].copy()

    print(f"  Runs: {len(runs)}")
    print(f"  Andere activiteiten: {len(other)}")

    # Stats
    if len(runs) > 0:
        print(f"\n  Run stats:")
        print(f"    Totaal km: {runs['distance_km'].sum():.1f}")
        print(f"    Gem. afstand: {runs['distance_km'].mean():.1f} km")
        print(f"    Gem. pace: {runs['pace_min_per_km'].mean():.2f} min/km")
        print(f"    Periode: {runs['date'].min()} - {runs['date'].max()}")

    # Koppel hartslag data als beschikbaar
    if os.path.exists(HR_DETAIL_CSV):
        hr_detail = pd.read_csv(HR_DETAIL_CSV, parse_dates=["date"])
        # Match op datum
        runs["date"] = pd.to_datetime(runs["date"])
        hr_detail["date"] = pd.to_datetime(hr_detail["date"])
        runs = runs.merge(
            hr_detail[["date", "hr_avg", "hr_max", "hr_min",
                       "zone_1_pct", "zone_2_pct", "zone_3_pct", "zone_4_pct", "zone_5_pct"]],
            on="date", how="left", suffixes=("", "_apple")
        )
        matched = runs["hr_avg"].notna().sum()
        print(f"  Met hartslagdata: {matched}")

    # Opslaan
    workouts_df.to_csv(APPLE_WORKOUTS_CSV, index=False)
    if len(splits_df) > 0:
        splits_df.to_csv(APPLE_SPLITS_CSV, index=False)

    # Sla runs apart op
    runs_only = workouts_df[workouts_df["is_run"]].copy()
    runs_only.to_csv(os.path.join(DATA_DIR, "apple_runs.csv"), index=False)

    print(f"\nOpgeslagen:")
    print(f"  {APPLE_WORKOUTS_CSV}")
    print(f"  {APPLE_SPLITS_CSV}")
    print(f"  {os.path.join(DATA_DIR, 'apple_runs.csv')}")


if __name__ == "__main__":
    main()
