"""
Data uploader — copies your processed CSV into the backend folder
Run this once before pushing to GitHub.

Usage:
    python3 prepare_backend.py
"""

import os
import shutil

SRC  = os.path.expanduser("~/pl-predictor/data/processed/all_seasons.csv")
DEST = os.path.expanduser("~/dd-predictor-backend/data/processed/all_seasons.csv")

os.makedirs(os.path.dirname(DEST), exist_ok=True)

if os.path.exists(SRC):
    shutil.copy2(SRC, DEST)
    size = os.path.getsize(DEST) / 1024
    print(f"✓ Copied all_seasons.csv to backend ({size:.0f} KB)")
else:
    print(f"✗ Source file not found: {SRC}")
    print("  Make sure you've run pl_data_loader.py first.")
