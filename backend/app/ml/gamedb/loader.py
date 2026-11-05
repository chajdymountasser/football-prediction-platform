# app/ml/gamedb/loader.py
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


REQUIRED_COLS = {"Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"}


def _parse_date_series(s: pd.Series) -> pd.Series:
    """
    football-data.co.uk dates can be dd/mm/yy or dd/mm/yyyy depending on season.
    We'll parse robustly with dayfirst=True.
    """
    return pd.to_datetime(s, dayfirst=True, errors="coerce")


def load_history_csvs(paths: Iterable[Path]) -> pd.DataFrame:
    """
    Load & concat football-data.co.uk CSVs.
    Output columns (minimum):
      - league (from Div)
      - Date (datetime)
      - HomeTeam, AwayTeam (str)
      - FTHG, FTAG (numeric)
      - FTR (str)
      - HS, AS, HST, AST (optional; used by your tier2 features like shot_accuracy/xg_proxy)
    """
    frames = []
    for p in paths:
        df = pd.read_csv(p)

        missing = REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(f"{p.name} is missing required columns: {sorted(missing)}")

        # Normalize core columns
        df = df.copy()
        df["Date"] = _parse_date_series(df["Date"])
        df = df[df["Date"].notna()]  # drop invalid dates

        df["league"] = df["Div"].astype(str).str.strip()
        df["HomeTeam"] = df["HomeTeam"].astype(str).str.strip()
        df["AwayTeam"] = df["AwayTeam"].astype(str).str.strip()

        # Numeric cleanup
        for c in ["FTHG", "FTAG", "HS", "AS", "HST", "AST"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        frames.append(df)

    out = pd.concat(frames, ignore_index=True)

    # Sort chronologically (critical for “as-of” state)
    out = out.sort_values("Date").reset_index(drop=True)
    return out


def load_history_from_folder(folder: Path, league: Optional[str] = None) -> pd.DataFrame:
    """
    Convenience: load all CSVs in a folder (e.g., app/data/raw/).
    If league is provided, keep only rows for that Div (PL/BL1/PD/SA/FL1).
    """
    paths = sorted(folder.glob("*.csv"))
    if not paths:
        raise ValueError(f"No CSV files found in {folder}")

    df = load_history_csvs(paths)
    if league:
        df = df[df["league"] == str(league).strip()].copy()
    return df

