import requests
import pandas as pd
import time

BASE_URL = "https://api.openf1.org/v1"

def fetch(endpoint, params={}):
    try:
        r = requests.get(f"{BASE_URL}/{endpoint}", params=params, timeout=30)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        print(f"  Error fetching {endpoint}: {e}")
    return []

def to_dt(series):
    return pd.to_datetime(series, format="ISO8601", utc=True)

# Pull race sessions from 2023 and 2024
sessions = []
for year in [2023, 2024]:
    s = fetch("sessions", {"session_type": "Race", "year": year})
    sessions.extend(s)

print(f"Found {len(sessions)} race sessions")

all_rows = []

for session in sessions:
    session_key = session["session_key"]
    circuit_key = session.get("circuit_key")
    circuit_short_name = session.get("circuit_short_name")
    year = session.get("year")

    print(f"Processing session {session_key} — {circuit_short_name} {year}...")

    # Team radio messages
    radio = fetch("team_radio", {"session_key": session_key})
    if not radio:
        print("  No radio data, skipping")
        continue

    # Laps
    laps = fetch("laps", {"session_key": session_key})
    laps_df = pd.DataFrame(laps) if laps else pd.DataFrame()
    if not laps_df.empty and "date_start" in laps_df.columns:
        laps_df["date_start"] = to_dt(laps_df["date_start"])

    # Drivers
    drivers = fetch("drivers", {"session_key": session_key})
    drivers_df = pd.DataFrame(drivers) if drivers else pd.DataFrame()

    # Stints
    stints = fetch("stints", {"session_key": session_key})
    stints_df = pd.DataFrame(stints) if stints else pd.DataFrame()

    # Position
    position = fetch("position", {"session_key": session_key})
    position_df = pd.DataFrame(position) if position else pd.DataFrame()
    if not position_df.empty and "date" in position_df.columns:
        position_df["date"] = to_dt(position_df["date"])

    # Intervals
    intervals = fetch("intervals", {"session_key": session_key})
    intervals_df = pd.DataFrame(intervals) if intervals else pd.DataFrame()
    if not intervals_df.empty and "date" in intervals_df.columns:
        intervals_df["date"] = to_dt(intervals_df["date"])

    # Weather
    weather = fetch("weather", {"session_key": session_key})
    weather_df = pd.DataFrame(weather) if weather else pd.DataFrame()
    if not weather_df.empty and "date" in weather_df.columns:
        weather_df["date"] = to_dt(weather_df["date"])

    print(f"  {len(radio)} radio messages")

    for msg in radio:
        driver_number = msg.get("driver_number")
        radio_date = to_dt(pd.Series([msg.get("date")]))[0]

        row = {
            "session_key": session_key,
            "circuit_key": circuit_key,
            "circuit_short_name": circuit_short_name,
            "year": year,
            "driver_number": driver_number,
            "date": msg.get("date"),
            "recording_url": msg.get("recording_url"),
            "meeting_key": msg.get("meeting_key"),
        }

        # Driver info
        if not drivers_df.empty and "driver_number" in drivers_df.columns:
            d = drivers_df[drivers_df["driver_number"] == driver_number]
            if not d.empty:
                row["first_name"] = d.iloc[0].get("first_name")
                row["last_name"] = d.iloc[0].get("last_name")
                row["name_acronym"] = d.iloc[0].get("name_acronym")
                row["team_name"] = d.iloc[0].get("team_name")

        # Lap context
        lap_number = None
        if not laps_df.empty and "driver_number" in laps_df.columns:
            driver_laps = laps_df[
                (laps_df["driver_number"] == driver_number) &
                laps_df["lap_duration"].notna() &
                laps_df["date_start"].notna()
            ].sort_values("lap_number").copy()

            current_lap = None
            for _, lap in driver_laps.iterrows():
                if lap["date_start"] <= radio_date:
                    current_lap = lap

            if current_lap is not None:
                lap_number = int(current_lap["lap_number"])
                row["lap_number"] = lap_number
                row["lap_duration"] = current_lap["lap_duration"]

                prev_laps = driver_laps[driver_laps["lap_number"].between(lap_number - 3, lap_number - 1)]["lap_duration"]
                next_laps = driver_laps[driver_laps["lap_number"].between(lap_number + 1, lap_number + 3)]["lap_duration"]

                if len(prev_laps) >= 1 and len(next_laps) >= 1:
                    avg_prev = prev_laps.mean()
                    avg_next = next_laps.mean()
                    lap_delta = avg_next - avg_prev
                    row["avg_prev_3_laps"] = round(avg_prev, 3)
                    row["avg_next_3_laps"] = round(avg_next, 3)
                    row["lap_delta"] = round(lap_delta, 3)
                    if lap_delta < -0.5:
                        row["label"] = "DOWN"
                    elif lap_delta > 0.5:
                        row["label"] = "UP"
                    else:
                        row["label"] = "NEUTRAL"

        # Tyre / stint
        if not stints_df.empty and lap_number is not None and "driver_number" in stints_df.columns:
            driver_stints = stints_df[stints_df["driver_number"] == driver_number]
            for _, stint in driver_stints.iterrows():
                if stint["lap_start"] <= lap_number <= stint["lap_end"]:
                    row["compound"] = stint.get("compound")
                    row["tyre_age"] = lap_number - stint["lap_start"] + stint.get("tyre_age_at_start", 0)
                    row["stint_number"] = stint.get("stint_number")
                    break

        # Position at time of radio
        if not position_df.empty and "driver_number" in position_df.columns:
            driver_pos = position_df[position_df["driver_number"] == driver_number].copy()
            if not driver_pos.empty:
                driver_pos["time_diff"] = (driver_pos["date"] - radio_date).abs()
                row["position"] = driver_pos.sort_values("time_diff").iloc[0].get("position")

        # Interval at time of radio
        if not intervals_df.empty and "driver_number" in intervals_df.columns:
            driver_int = intervals_df[intervals_df["driver_number"] == driver_number].copy()
            if not driver_int.empty:
                driver_int["time_diff"] = (driver_int["date"] - radio_date).abs()
                closest = driver_int.sort_values("time_diff").iloc[0]
                row["interval"] = closest.get("interval")
                row["gap_to_leader"] = closest.get("gap_to_leader")

        # Weather at time of radio
        if not weather_df.empty:
            weather_df["time_diff"] = (weather_df["date"] - radio_date).abs()
            closest_w = weather_df.sort_values("time_diff").iloc[0]
            row["air_temperature"] = closest_w.get("air_temperature")
            row["track_temperature"] = closest_w.get("track_temperature")
            row["rainfall"] = closest_w.get("rainfall")
            row["wind_speed"] = closest_w.get("wind_speed")

        all_rows.append(row)

    time.sleep(0.3)

df = pd.DataFrame(all_rows)
df.to_csv("raw_f1_dataset.csv", index=False)
print(f"\nSaved {len(df)} rows to raw_f1_dataset.csv")
print(f"Columns: {df.columns.tolist()}")
print(f"\nLabel distribution:\n{df['label'].value_counts() if 'label' in df.columns else 'no labels yet'}")
