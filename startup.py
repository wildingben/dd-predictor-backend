"""
DD Predictor — Background Data Loader
Downloads data in background so the app starts immediately.
"""

import os
import io
import threading
import requests
import pandas as pd

DATA_DIR    = "data/processed"
OUTPUT_PATH = os.path.join(DATA_DIR, "all_seasons.csv")
BASE_URL    = "https://www.football-data.co.uk/mmz4281/{season}/E0.csv"

SEASONS = {
    "2122": "2021-22", "2223": "2022-23", "2324": "2023-24",
    "2425": "2024-25", "2526": "2025-26",
}

COLUMNS_NEEDED = [
    "Date","HomeTeam","AwayTeam","FTHG","FTAG","FTR",
    "HTHG","HTAG","HTR","HS","AS","HST","AST",
    "HY","AY","HR","AR","Referee",
    "B365H","B365D","B365A",
    "AvgH","AvgD","AvgA","Avg>2.5","Avg<2.5",
    "MaxH","MaxD","MaxA",
]

def download_season(code, label):
    try:
        resp = requests.get(BASE_URL.format(season=code), timeout=15)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text), encoding="latin1")
        available = [c for c in COLUMNS_NEEDED if c in df.columns]
        df = df[available].copy()
        df["Season"] = label
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        df = df.dropna(subset=["Date","FTR"])
        for col in ["FTHG","FTAG","HTHG","HTAG","HS","AS","HST","AST","HY","AY","HR","AR"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        for col in ["B365H","B365D","B365A","AvgH","AvgD","AvgA","Avg>2.5","Avg<2.5","MaxH","MaxD","MaxA"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df["TotalGoals"]   = df["FTHG"] + df["FTAG"]
        df["GoalsOver25"]  = (df["TotalGoals"] > 2.5).astype(int)
        df["TotalYellows"] = df.get("HY", pd.Series([0]*len(df))) + df.get("AY", pd.Series([0]*len(df)))
        df["TotalReds"]    = df.get("HR", pd.Series([0]*len(df))) + df.get("AR", pd.Series([0]*len(df)))
        df["GW"] = df.groupby("Season").cumcount() // 10 + 1
        print(f"  ✓ {label}: {len(df)} fixtures")
        return df
    except Exception as e:
        print(f"  ✗ {label}: {e}")
        return None

def download_all():
    os.makedirs(DATA_DIR, exist_ok=True)
    frames = []
    for code, label in SEASONS.items():
        df = download_season(code, label)
        if df is not None:
            frames.append(df)
    if frames:
        combined = pd.concat(frames, ignore_index=True).sort_values("Date")
        combined.to_csv(OUTPUT_PATH, index=False)
        print(f"✓ Data ready: {len(combined)} fixtures saved")

def run_background():
    """Start download in background thread — app starts immediately"""
    if os.path.exists(OUTPUT_PATH):
        age_h = (pd.Timestamp.now() - pd.Timestamp(os.path.getmtime(OUTPUT_PATH), unit='s')).total_seconds() / 3600
        if age_h < 12:
            print(f"Data is fresh ({age_h:.1f}h old) — skipping download")
            return
    print("Starting background data download...")
    thread = threading.Thread(target=download_all, daemon=True)
    thread.start()

if __name__ == "__main__":
    # When run directly (from Procfile) — download synchronously
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(OUTPUT_PATH):
        print("No data found — downloading now...")
        download_all()
    else:
        age_h = (pd.Timestamp.now() - pd.Timestamp(os.path.getmtime(OUTPUT_PATH), unit='s')).total_seconds() / 3600
        if age_h > 12:
            print(f"Data is {age_h:.1f}h old — refreshing...")
            download_all()
        else:
            print(f"Data is fresh ({age_h:.1f}h old) — skipping download")
