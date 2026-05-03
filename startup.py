"""
DD Predictor — Startup Data Loader
====================================
Downloads fresh Premier League data from football-data.co.uk
when the Railway server starts up.
Runs automatically before the Flask app starts.
"""

import os
import io
import requests
import pandas as pd

DATA_DIR      = "data/processed"
OUTPUT_PATH   = os.path.join(DATA_DIR, "all_seasons.csv")
BASE_URL      = "https://www.football-data.co.uk/mmz4281/{season}/E0.csv"

SEASONS = ["2122", "2223", "2324", "2425", "2526"]
SEASON_LABELS = {
    "2122": "2021-22", "2223": "2022-23", "2324": "2023-24",
    "2425": "2024-25", "2526": "2025-26",
}

COLUMNS_NEEDED = [
    "Date","HomeTeam","AwayTeam","FTHG","FTAG","FTR",
    "HTHG","HTAG","HTR","HS","AS","HST","AST",
    "HY","AY","HR","AR","Referee",
    "B365H","B365D","B365A",
    "AvgH","AvgD","AvgA",
    "Avg>2.5","Avg<2.5",
    "MaxH","MaxD","MaxA",
]

def download_season(season_code, season_label):
    url = BASE_URL.format(season=season_code)
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text), encoding="latin1")
        available = [c for c in COLUMNS_NEEDED if c in df.columns]
        df = df[available].copy()
        df["Season"] = season_label
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        df = df.dropna(subset=["Date","FTR"])
        numeric = ["FTHG","FTAG","HTHG","HTAG","HS","AS","HST","AST","HY","AY","HR","AR"]
        for col in numeric:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        odds_cols = ["B365H","B365D","B365A","AvgH","AvgD","AvgA","Avg>2.5","Avg<2.5","MaxH","MaxD","MaxA"]
        for col in odds_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df["TotalGoals"]   = df["FTHG"] + df["FTAG"]
        df["GoalsOver25"]  = (df["TotalGoals"] > 2.5).astype(int)
        df["TotalYellows"] = df.get("HY", pd.Series([0]*len(df))) + df.get("AY", pd.Series([0]*len(df)))
        df["TotalReds"]    = df.get("HR", pd.Series([0]*len(df))) + df.get("AR", pd.Series([0]*len(df)))
        df["GW"] = df.groupby("Season").cumcount() // 10 + 1
        print(f"  ✓ {season_label}: {len(df)} fixtures")
        return df
    except Exception as e:
        print(f"  ✗ {season_label}: {e}")
        return None

def run():
    # Skip if data already exists and is recent (less than 6 hours old)
    if os.path.exists(OUTPUT_PATH):
        age_hours = (pd.Timestamp.now() - pd.Timestamp(os.path.getmtime(OUTPUT_PATH), unit='s')).total_seconds() / 3600
        if age_hours < 6:
            print(f"Data is fresh ({age_hours:.1f}h old) — skipping download")
            return
    
    print("Downloading Premier League data...")
    os.makedirs(DATA_DIR, exist_ok=True)
    
    frames = []
    for code, label in SEASON_LABELS.items():
        df = download_season(code, label)
        if df is not None:
            frames.append(df)
    
    if frames:
        combined = pd.concat(frames, ignore_index=True).sort_values("Date")
        combined.to_csv(OUTPUT_PATH, index=False)
        print(f"✓ Saved {len(combined)} fixtures to {OUTPUT_PATH}")
    else:
        print("✗ No data downloaded")

if __name__ == "__main__":
    run()
