"""
Parse Apple Health export XML en koppel data aan Strava runs.
Extraheert hartslag, stappen, afstand en rusthart data.
"""
import os
import json
from datetime import datetime, timedelta

import pandas as pd
from lxml import etree

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
HEALTH_XML = os.path.join(os.path.dirname(__file__), "apple_health_export", "export.xml")
RUNS_CSV = os.path.join(DATA_DIR, "runs.csv")

# Output bestanden
HR_DETAIL_CSV = os.path.join(DATA_DIR, "heartrate_detail.csv")
RESTING_HR_CSV = os.path.join(DATA_DIR, "resting_heartrate.csv")
DAILY_STEPS_CSV = os.path.join(DATA_DIR, "daily_steps.csv")
RUN_HR_DIR = os.path.join(DATA_DIR, "run_heartrates")


def parse_datetime(dt_str):
    """Parse Apple Health datetime string."""
    # Format: '2024-09-18 21:14:56 +0100'
    try:
        return datetime.strptime(dt_str[:19], "%Y-%m-%d %H:%M:%S")
    except:
        return None


def extract_health_data():
    """Stream-parse de Apple Health XML en extraheer relevante data."""
    print("Apple Health XML parsen (dit kan even duren bij grote bestanden)...")

    heartrates = []
    steps = []
    distances = []
    count = 0

    for event, elem in etree.iterparse(HEALTH_XML, events=('end',), recover=True):
        if elem.tag != 'Record':
            elem.clear()
            continue

        record_type = elem.get('type', '')
        count += 1

        if count % 500000 == 0:
            print(f"  Verwerkt: {count} records...")

        if record_type == 'HKQuantityTypeIdentifierHeartRate':
            dt = parse_datetime(elem.get('startDate', ''))
            val = elem.get('value')
            if dt and val:
                heartrates.append({
                    'datetime': dt,
                    'value': float(val),
                    'source': elem.get('sourceName', ''),
                })

        elif record_type == 'HKQuantityTypeIdentifierStepCount':
            dt_start = parse_datetime(elem.get('startDate', ''))
            dt_end = parse_datetime(elem.get('endDate', ''))
            val = elem.get('value')
            if dt_start and val:
                steps.append({
                    'start': dt_start,
                    'end': dt_end,
                    'value': float(val),
                })

        elif record_type == 'HKQuantityTypeIdentifierDistanceWalkingRunning':
            dt_start = parse_datetime(elem.get('startDate', ''))
            val = elem.get('value')
            if dt_start and val:
                distances.append({
                    'datetime': dt_start,
                    'value_km': float(val),
                })

        elem.clear()

    print(f"  Totaal verwerkt: {count} records")
    print(f"  Hartslag: {len(heartrates)} metingen")
    print(f"  Stappen: {len(steps)} records")
    print(f"  Afstand: {len(distances)} records")

    return heartrates, steps, distances


def build_resting_heartrate(heartrates):
    """Bereken dagelijkse rusthartslag (laagste 10% van de dag, excl. nacht)."""
    print("Rusthartslag berekenen...")
    df = pd.DataFrame(heartrates)
    df['date'] = df['datetime'].dt.date
    df['hour'] = df['datetime'].dt.hour

    # Filter: 6-23 uur, alleen rustige metingen (niet tijdens workout)
    # Rusthartslag = laagste 10% van de dag
    daily_resting = []
    for date, group in df.groupby('date'):
        values = group['value'].values
        if len(values) >= 10:
            threshold = pd.Series(values).quantile(0.10)
            resting = values[values <= threshold].mean()
            daily_resting.append({
                'date': date,
                'resting_hr': round(resting, 1),
                'min_hr': round(values.min(), 1),
                'max_hr': round(values.max(), 1),
                'measurements': len(values),
            })

    resting_df = pd.DataFrame(daily_resting)
    resting_df.to_csv(RESTING_HR_CSV, index=False)
    print(f"  Rusthartslag: {len(resting_df)} dagen opgeslagen")
    return resting_df


def build_daily_steps(steps):
    """Bereken dagelijkse stappen."""
    print("Dagelijkse stappen berekenen...")
    df = pd.DataFrame(steps)
    df['date'] = df['start'].dt.date
    daily = df.groupby('date')['value'].sum().reset_index()
    daily.columns = ['date', 'steps']
    daily['steps'] = daily['steps'].astype(int)
    daily.to_csv(DAILY_STEPS_CSV, index=False)
    print(f"  Stappen: {len(daily)} dagen opgeslagen")
    return daily


def match_heartrate_to_runs(heartrates, runs_df):
    """Koppel Apple Watch hartslag data aan Strava runs."""
    print("Hartslag koppelen aan Strava runs...")
    os.makedirs(RUN_HR_DIR, exist_ok=True)

    hr_df = pd.DataFrame(heartrates)
    hr_df = hr_df.sort_values('datetime')

    matched_runs = []

    for _, run in runs_df.iterrows():
        run_start = pd.to_datetime(run['start_date_local']).tz_localize(None)
        run_duration_min = run['moving_time_min']
        run_end = run_start + timedelta(minutes=run_duration_min + 5)  # +5 min buffer
        run_start_buf = run_start - timedelta(minutes=2)  # -2 min warmup

        # Filter hartslag data voor deze run
        mask = (hr_df['datetime'] >= run_start_buf) & (hr_df['datetime'] <= run_end)
        run_hr = hr_df[mask].copy()

        if len(run_hr) < 3:
            continue

        # Bereken stats
        hr_values = run_hr['value'].values
        run_info = {
            'run_id': run['id'],
            'date': run['date'],
            'name': run['name'],
            'distance_km': run['distance_km'],
            'hr_avg': round(hr_values.mean(), 1),
            'hr_max': round(hr_values.max(), 1),
            'hr_min': round(hr_values.min(), 1),
            'hr_measurements': len(hr_values),
        }

        # Hartslag zones (op basis van max HR geschat als 220-leeftijd of max gevonden)
        max_hr_estimate = max(hr_values.max(), 190)  # Schat conservatief
        zone_boundaries = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        zone_names = ['Z1 (Recovery)', 'Z2 (Easy)', 'Z3 (Tempo)', 'Z4 (Threshold)', 'Z5 (Max)']
        for i, name in enumerate(zone_names):
            lower = max_hr_estimate * zone_boundaries[i]
            upper = max_hr_estimate * zone_boundaries[i + 1]
            zone_time = len(hr_values[(hr_values >= lower) & (hr_values < upper)])
            zone_pct = round(zone_time / len(hr_values) * 100, 1)
            run_info[f'zone_{i+1}_pct'] = zone_pct

        matched_runs.append(run_info)

        # Sla gedetailleerde hartslag per run op
        run_hr['minutes_elapsed'] = (run_hr['datetime'] - run_start).dt.total_seconds() / 60
        run_hr[['minutes_elapsed', 'value']].to_csv(
            os.path.join(RUN_HR_DIR, f"run_{run['id']}.csv"), index=False
        )

    result_df = pd.DataFrame(matched_runs)
    result_df.to_csv(HR_DETAIL_CSV, index=False)
    print(f"  Gekoppeld: {len(matched_runs)} runs met hartslagdata")
    return result_df


def main():
    if not os.path.exists(HEALTH_XML):
        print("Apple Health export niet gevonden. Plaats export.xml in apple_health_export/")
        return

    if not os.path.exists(RUNS_CSV):
        print("Strava runs niet gevonden. Draai eerst fetch_data.py")
        return

    # Parse Apple Health data
    heartrates, steps, distances = extract_health_data()

    # Bouw afgeleide datasets
    if heartrates:
        resting = build_resting_heartrate(heartrates)
        runs_df = pd.read_csv(RUNS_CSV, parse_dates=['start_date_local'])
        runs_df['date'] = runs_df['start_date_local'].dt.date
        matched = match_heartrate_to_runs(heartrates, runs_df)

    if steps:
        daily_steps = build_daily_steps(steps)

    print("\nKlaar! Data opgeslagen in:", DATA_DIR)


if __name__ == "__main__":
    main()
