from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple
import re
import unicodedata

import numpy as np
import pandas as pd


# =========================
# Constants
# =========================
LAST_N = 5
DECAY = 0.85

ELO_BASE = 1500.0
ELO_K = 32.0
ELO_HOME_ADV = 100.0

LEAGUE_AVG_GOALS = 1.4
LEAGUE_AVG_GOALS_CONCEDED = 1.4

LEAGUE_TO_DIV = {
    "PL": "E0",
    "BL1": "D1",
    "PD": "SP1",
    "SA": "I1",
    "FL1": "F1",
}

# =========================
# Team name aliases (PL)
# =========================
PL_ALIASES = {
    "West Ham United FC": "West Ham",
    "Sunderland AFC": "Sunderland",
    "Burnley FC": "Burnley",
    "Tottenham Hotspur FC": "Tottenham",
    "Fulham FC": "Fulham",
    "Brighton & Hove Albion FC": "Brighton",
    "Manchester City FC": "Man City",
    "Wolverhampton Wanderers FC": "Wolves",
    "AFC Bournemouth": "Bournemouth",
    "Liverpool FC": "Liverpool",
    "Brentford FC": "Brentford",
    "Nottingham Forest FC": "Nott'm Forest",
    "Crystal Palace FC": "Crystal Palace",
    "Chelsea FC": "Chelsea",
    "Newcastle United FC": "Newcastle",
    "Aston Villa FC": "Aston Villa",
    "Arsenal FC": "Arsenal",
    "Manchester United FC": "Man United",
    "Everton FC": "Everton",
    "Leeds United FC": "Leeds",
}
PL_ALIASES.update({k.replace(" FC", ""): v for k, v in list(PL_ALIASES.items()) if k.endswith(" FC")})


def _norm_team(s: str) -> str:
    s = (s or "").strip().lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = re.sub(r"\b(fc|cf|ssc|ac|cd|ud|as|sv|afc|sc)\b", "", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _resolve_team_name(league: str, request_name: str, known_teams: List[str]) -> str:
    if not request_name:
        return request_name
    if request_name in known_teams:
        return request_name
    if league == "PL" and request_name in PL_ALIASES:
        mapped = PL_ALIASES[request_name]
        return mapped if mapped else request_name

    req_n = _norm_team(request_name)
    if not req_n:
        return request_name

    norm_map = {}
    for t in known_teams:
        nt = _norm_team(t)
        if nt and nt not in norm_map:
            norm_map[nt] = t

    if req_n in norm_map:
        return norm_map[req_n]

    hits = []
    for nk, original in norm_map.items():
        if nk and (nk in req_n or req_n in nk):
            hits.append(original)
            if len(hits) > 1:
                break
    if len(hits) == 1:
        return hits[0]

    return request_name


# =========================
# Paths + parsing
# =========================
def _here_paths() -> Tuple[Path, Path]:
    app_dir = Path(__file__).resolve().parents[2]  # .../backend/app
    splits_dir = app_dir / "data" / "splits"
    return app_dir, splits_dir


def _parse_fixture_date(date_value: Any) -> pd.Timestamp:
    if date_value is None:
        raise ValueError("Missing fixture date.")
    if isinstance(date_value, str):
        s = date_value.strip()
        if len(s) >= 10 and s[4] == "-" and s[7] == "-":
            ts = pd.to_datetime(s[:10], format="%Y-%m-%d", errors="coerce")
            if not pd.isna(ts):
                return ts.normalize()
    ts = pd.to_datetime(date_value, errors="coerce", dayfirst=True)
    if pd.isna(ts):
        raise ValueError(f"Could not parse fixture date: {date_value!r}")
    return ts.normalize()


def _extract_identity(req: Any, ctx: Any) -> Tuple[str, pd.Timestamp, str, str]:
    league = getattr(req, "league", None) or (getattr(ctx, "competition_code", None) if ctx else None)
    league = (league or "").strip()
    if not league:
        raise ValueError("Missing league.")

    date_value = getattr(req, "date", None) or (getattr(ctx, "utc_date", None) if ctx else None)
    fixture_date = _parse_fixture_date(date_value)

    home = getattr(req, "home_team", None) or getattr(req, "homeTeam", None)
    away = getattr(req, "away_team", None) or getattr(req, "awayTeam", None)
    home = (home or "").strip()
    away = (away or "").strip()
    if not home or not away:
        raise ValueError("Missing team names.")
    return league, fixture_date, home, away


# =========================
# Load history
# =========================
def _load_history_for_league(app_dir: Path, league: str) -> pd.DataFrame:
    raw_dir = app_dir / "data" / "raw"
    paths = sorted(raw_dir.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSVs found in {raw_dir}")

    div_code = LEAGUE_TO_DIV.get(league, league)
    frames = []

    for p in paths:
        df = pd.read_csv(p, encoding="latin-1", low_memory=False, on_bad_lines="skip")
        df.columns = df.columns.str.strip()

        need = {"Div", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"}
        if not need.issubset(set(df.columns)):
            continue

        df = df[df["Div"].astype(str).str.strip() == div_code].copy()
        if df.empty:
            continue

        # Parse Date
        date_raw = df["Date"].astype(str).str.strip()
        d1 = pd.to_datetime(date_raw, format="%d/%m/%Y", errors="coerce")
        d2 = pd.to_datetime(date_raw, format="%d/%m/%y", errors="coerce")
        df["Date"] = d1.fillna(d2).dt.normalize()
        df = df[df["Date"].notna()].copy()

        # Ensure optional stats exist for proxies
        for col in ["HS", "AS", "HST", "AST"]:
            if col not in df.columns:
                df[col] = 0

        for col in ["FTHG", "FTAG", "HS", "AS", "HST", "AST"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        df["HomeTeam"] = df["HomeTeam"].astype(str).str.strip()
        df["AwayTeam"] = df["AwayTeam"].astype(str).str.strip()
        df["FTR"] = df["FTR"].astype(str).str.strip()

        frames.append(df)

    if not frames:
        raise ValueError(f"No rows for league '{league}' (Div={div_code}).")

    out = pd.concat(frames, ignore_index=True).sort_values("Date").reset_index(drop=True)
    return out


# =========================
# Elo
# =========================
def _elo_expected(home_elo: float, away_elo: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((away_elo - home_elo) / 400.0))


def _elo_update(home_elo: float, away_elo: float, result: str) -> Tuple[float, float]:
    home_adv = home_elo + ELO_HOME_ADV
    exp_home = _elo_expected(home_adv, away_elo)

    if result == "H":
        score_home = 1.0
    elif result == "D":
        score_home = 0.5
    else:
        score_home = 0.0

    home_new = home_elo + ELO_K * (score_home - exp_home)
    away_new = away_elo + ELO_K * ((1.0 - score_home) - (1.0 - exp_home))
    return home_new, away_new


# =========================
# Normalization (matches your CSV header Unnamed: 0)
# =========================
def _load_norm_params(splits_dir: Path) -> pd.DataFrame:
    p = splits_dir / "normalization_parameters.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing normalization parameters: {p}")
    df = pd.read_csv(p)
    df.columns = [str(c).strip() for c in df.columns]
    # format: Unnamed: 0, mean, std
    df = df.rename(columns={df.columns[0]: "feature"})
    df["feature"] = df["feature"].astype(str).str.strip()
    df = df.set_index("feature")
    return df


def _z(v: float, mean: float, std: float) -> float:
    if std is None or float(std) == 0.0:
        return 0.0
    return (float(v) - float(mean)) / float(std)


# =========================
# Helpers for last5 + EWMA
# =========================
def _ewma(points_seq: List[float], decay: float = 0.85) -> float:
    if not points_seq:
        return 0.0
    w = 1.0
    s = 0.0
    ws = 0.0
    for p in reversed(points_seq):  # most recent first
        s += p * w
        ws += w
        w *= decay
    return s / ws if ws > 0 else 0.0


def _safe_div(a: float, b: float, eps: float = 1e-9) -> float:
    return float(a) / float(b + eps)


# =========================
# Main builder (EXACT 29 keys)
# =========================
def build_features(req: Any, ctx: Any) -> Dict[str, float]:
    app_dir, splits_dir = _here_paths()
    league, fixture_date, home_req, away_req = _extract_identity(req, ctx)

    history_df = _load_history_for_league(app_dir, league)

    known = sorted(set(history_df["HomeTeam"]).union(set(history_df["AwayTeam"])))
    home = _resolve_team_name(league, home_req, known)
    away = _resolve_team_name(league, away_req, known)

    # only matches BEFORE fixture date
    past = history_df[history_df["Date"] < fixture_date].copy()

    # Elo and rolling state up to fixture
    elo: Dict[str, float] = {}
    def get_elo(team: str) -> float:
        return elo.get(team, ELO_BASE)

    # Track season-to-date PPG components
    pts: Dict[str, float] = {}
    mp: Dict[str, float] = {}
    home_pts: Dict[str, float] = {}
    home_mp: Dict[str, float] = {}
    away_pts: Dict[str, float] = {}
    away_mp: Dict[str, float] = {}

    gf: Dict[str, float] = {}
    ga: Dict[str, float] = {}

    # For last5 metrics:
    last_results: Dict[str, List[str]] = {}         # "W/D/L"
    last_points: Dict[str, List[float]] = {}        # 3/1/0
    last_gd: Dict[str, List[float]] = {}            # goal diff
    last_xgproxy: Dict[str, List[float]] = {}       # shot proxy
    last_cs: Dict[str, List[float]] = {}            # clean sheet 1/0
    last_fts: Dict[str, List[float]] = {}           # failed to score 1/0
    last_btts: Dict[str, List[float]] = {}          # both teams scored 1/0
    last_shotacc: Dict[str, List[float]] = {}       # on_target / shots

    def ensure(team: str):
        elo.setdefault(team, ELO_BASE)
        pts.setdefault(team, 0.0)
        mp.setdefault(team, 0.0)
        home_pts.setdefault(team, 0.0)
        home_mp.setdefault(team, 0.0)
        away_pts.setdefault(team, 0.0)
        away_mp.setdefault(team, 0.0)
        gf.setdefault(team, 0.0)
        ga.setdefault(team, 0.0)
        last_results.setdefault(team, [])
        last_points.setdefault(team, [])
        last_gd.setdefault(team, [])
        last_xgproxy.setdefault(team, [])
        last_cs.setdefault(team, [])
        last_fts.setdefault(team, [])
        last_btts.setdefault(team, [])
        last_shotacc.setdefault(team, [])

    for _, r in past.sort_values("Date").iterrows():
        ht = str(r["HomeTeam"]).strip()
        at = str(r["AwayTeam"]).strip()
        res = str(r["FTR"]).strip()

        ensure(ht); ensure(at)

        fthg = float(r["FTHG"]); ftag = float(r["FTAG"])
        hs = float(r.get("HS", 0)); hst = float(r.get("HST", 0))
        a_s = float(r.get("AS", 0)); ast = float(r.get("AST", 0))

        # update played
        mp[ht] += 1; mp[at] += 1
        home_mp[ht] += 1; away_mp[at] += 1

        # goals
        gf[ht] += fthg; ga[ht] += ftag
        gf[at] += ftag; ga[at] += fthg

        # points by result
        if res == "H":
            ph, pa = 3.0, 0.0
            rh, ra = "W", "L"
        elif res == "D":
            ph, pa = 1.0, 1.0
            rh, ra = "D", "D"
        else:
            ph, pa = 0.0, 3.0
            rh, ra = "L", "W"

        pts[ht] += ph; pts[at] += pa
        home_pts[ht] += ph; away_pts[at] += pa

        # last5 sequences (cap at LAST_N)
        def push(seq: List, val):
            seq.append(val)
            if len(seq) > LAST_N:
                del seq[0]

        push(last_results[ht], rh); push(last_results[at], ra)
        push(last_points[ht], ph); push(last_points[at], pa)

        gd_h = fthg - ftag
        gd_a = ftag - fthg
        push(last_gd[ht], gd_h); push(last_gd[at], gd_a)

        # xg proxy (simple): shots on target + 0.5*shots
        xg_h = hst + 0.5 * hs
        xg_a = ast + 0.5 * a_s
        push(last_xgproxy[ht], xg_h); push(last_xgproxy[at], xg_a)

        # clean sheet
        push(last_cs[ht], 1.0 if ftag == 0 else 0.0)
        push(last_cs[at], 1.0 if fthg == 0 else 0.0)

        # failed to score
        push(last_fts[ht], 1.0 if fthg == 0 else 0.0)
        push(last_fts[at], 1.0 if ftag == 0 else 0.0)

        # BTTS
        push(last_btts[ht], 1.0 if (fthg > 0 and ftag > 0) else 0.0)
        push(last_btts[at], 1.0 if (fthg > 0 and ftag > 0) else 0.0)

        # shot accuracy
        push(last_shotacc[ht], _safe_div(hst, hs) if hs > 0 else 0.0)
        push(last_shotacc[at], _safe_div(ast, a_s) if a_s > 0 else 0.0)

        # Elo update
        home_old = get_elo(ht)
        away_old = get_elo(at)
        home_new, away_new = _elo_update(home_old, away_old, res)
        elo[ht] = home_new
        elo[at] = away_new

    # ---- Tier 1 (13) ----
    home_elo_pre = float(get_elo(home))
    away_elo_pre = float(get_elo(away))
    elo_diff_pre = float(home_elo_pre - away_elo_pre)

    home_ppg_current = float(pts.get(home, 0.0) / max(1.0, mp.get(home, 0.0)))
    away_ppg_current = float(pts.get(away, 0.0) / max(1.0, mp.get(away, 0.0)))

    home_ppg_home = float(home_pts.get(home, 0.0) / max(1.0, home_mp.get(home, 0.0)))
    away_ppg_away = float(away_pts.get(away, 0.0) / max(1.0, away_mp.get(away, 0.0)))

    home_goal_diff_ratio = float((gf.get(home, 0.0) - ga.get(home, 0.0)) / max(1.0, mp.get(home, 0.0)))
    away_goal_diff_ratio = float((gf.get(away, 0.0) - ga.get(away, 0.0)) / max(1.0, mp.get(away, 0.0)))

    home_attack_strength = float((gf.get(home, 0.0) / max(1.0, mp.get(home, 0.0))) / LEAGUE_AVG_GOALS)
    away_attack_strength = float((gf.get(away, 0.0) / max(1.0, mp.get(away, 0.0))) / LEAGUE_AVG_GOALS)

    home_defense_strength = float(LEAGUE_AVG_GOALS_CONCEDED / max(0.01, (ga.get(home, 0.0) / max(1.0, mp.get(home, 0.0)))))
    away_defense_strength = float(LEAGUE_AVG_GOALS_CONCEDED / max(0.01, (ga.get(away, 0.0) / max(1.0, mp.get(away, 0.0)))))

    # ---- Last5 / EWMA features (16) ----
    def last5_rate(seq: List[float]) -> float:
        return float(np.mean(seq)) if seq else 0.0

    def form_last5(results: List[str]) -> float:
        # Points per match over last5
        if not results:
            return 0.0
        pts_map = {"W": 3.0, "D": 1.0, "L": 0.0}
        return float(np.mean([pts_map.get(r, 0.0) for r in results]))

    # goal diff trend: slope over last5 GD (simple linear regression slope)
    def gd_trend(gd_seq: List[float]) -> float:
        if len(gd_seq) < 2:
            return 0.0
        x = np.arange(len(gd_seq), dtype=float)
        y = np.array(gd_seq, dtype=float)
        x_mean = x.mean()
        denom = ((x - x_mean) ** 2).sum()
        if denom == 0:
            return 0.0
        slope = ((x - x_mean) * (y - y.mean())).sum() / denom
        return float(slope)

    # XG proxy last5: mean of proxy
    def xg_proxy_last5(seq: List[float]) -> float:
        return float(np.mean(seq)) if seq else 0.0

    def points_ewma(seq: List[float]) -> float:
        return float(_ewma(seq, decay=DECAY)) if seq else 0.0

    # Build for each side
    home_form_last5 = form_last5(last_results.get(home, []))
    away_form_last5 = form_last5(last_results.get(away, []))

    home_goal_diff_trend = gd_trend(last_gd.get(home, []))
    away_goal_diff_trend = gd_trend(last_gd.get(away, []))

    home_xg_proxy_last5 = xg_proxy_last5(last_xgproxy.get(home, []))
    away_xg_proxy_last5 = xg_proxy_last5(last_xgproxy.get(away, []))

    home_clean_sheet_rate_last5 = last5_rate(last_cs.get(home, []))
    away_clean_sheet_rate_last5 = last5_rate(last_cs.get(away, []))

    home_failed_to_score_rate_last5 = last5_rate(last_fts.get(home, []))
    away_failed_to_score_rate_last5 = last5_rate(last_fts.get(away, []))

    home_btts_rate_last5 = last5_rate(last_btts.get(home, []))
    away_btts_rate_last5 = last5_rate(last_btts.get(away, []))

    home_shot_accuracy_last5 = last5_rate(last_shotacc.get(home, []))
    away_shot_accuracy_last5 = last5_rate(last_shotacc.get(away, []))

    home_points_ewma = points_ewma(last_points.get(home, []))
    away_points_ewma = points_ewma(last_points.get(away, []))

    # EXACT 29 keys (from feature_order.json)
    raw = {
        "home_elo_pre": home_elo_pre,
        "away_elo_pre": away_elo_pre,
        "elo_diff_pre": elo_diff_pre,
        "home_ppg_current": home_ppg_current,
        "away_ppg_current": away_ppg_current,
        "home_ppg_home": home_ppg_home,
        "away_ppg_away": away_ppg_away,
        "home_goal_diff_ratio": home_goal_diff_ratio,
        "away_goal_diff_ratio": away_goal_diff_ratio,
        "home_attack_strength": home_attack_strength,
        "home_defense_strength": home_defense_strength,
        "away_attack_strength": away_attack_strength,
        "away_defense_strength": away_defense_strength,

        "away_form_last5": away_form_last5,
        "away_goal_diff_trend": away_goal_diff_trend,
        "away_xg_proxy_last5": away_xg_proxy_last5,
        "away_clean_sheet_rate_last5": away_clean_sheet_rate_last5,
        "away_failed_to_score_rate_last5": away_failed_to_score_rate_last5,
        "away_btts_rate_last5": away_btts_rate_last5,
        "away_shot_accuracy_last5": away_shot_accuracy_last5,
        "away_points_ewma": away_points_ewma,

        "home_form_last5": home_form_last5,
        "home_goal_diff_trend": home_goal_diff_trend,
        "home_xg_proxy_last5": home_xg_proxy_last5,
        "home_clean_sheet_rate_last5": home_clean_sheet_rate_last5,
        "home_failed_to_score_rate_last5": home_failed_to_score_rate_last5,
        "home_btts_rate_last5": home_btts_rate_last5,
        "home_shot_accuracy_last5": home_shot_accuracy_last5,
        "home_points_ewma": home_points_ewma,
    }

    # normalize
    norm_df = _load_norm_params(splits_dir)

    out: Dict[str, float] = {}
    for k, v in raw.items():
        if k in norm_df.index:
            mean = float(norm_df.loc[k, "mean"])
            std = float(norm_df.loc[k, "std"])
            out[k] = float(_z(v, mean, std))
        else:
            out[k] = float(v)

    return out
