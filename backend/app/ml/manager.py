from __future__ import annotations

from typing import Any, Dict
import json
import os
import time

from app.ml.types import PredictRequest, MatchContext
from app.ml.features.builder import build_features

from app.ml.predictors.base import BasePredictor
from app.ml.predictors.logreg_model import LogisticRegressionPredictor
from app.ml.predictors.rf_model import RandomForestPredictor


def _clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return float(x)


def _normalize_probs(p: Dict[str, float]) -> Dict[str, float]:
    h = _clamp01(float(p.get("homeWin", 0.0)))
    d = _clamp01(float(p.get("draw", 0.0)))
    a = _clamp01(float(p.get("awayWin", 0.0)))
    s = h + d + a
    if s <= 0:
        return {"homeWin": 0.34, "draw": 0.33, "awayWin": 0.33}
    return {"homeWin": h / s, "draw": d / s, "awayWin": a / s}


def _load_metadata() -> Dict[str, Any]:
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "artifacts", "metadata.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {"default": "logreg_v1", "perLeague": {}}


def _build_predictor_by_name(name: str) -> BasePredictor:
    """
    This is the only place that maps a string name -> predictor implementation.
    Add new model families by adding new prefixes.

    Examples:
      - rf_v1, rf_v2, ...
      - logreg_v1, logreg_v2, ...
    """
    name = (name or "").strip()

    if name.startswith("rf_"):
        return RandomForestPredictor(model_id=name)

    if name.startswith("logreg_"):
        return LogisticRegressionPredictor(model_id=name)

    raise ValueError(f"Unknown model name: {name}")


def select_predictor(req: PredictRequest) -> BasePredictor:
    meta = _load_metadata()
    per_league = meta.get("perLeague") or {}
    chosen = per_league.get(req.league) or meta.get("default") or "rf_v1"
    return _build_predictor_by_name(str(chosen))


def _draw_boost_close_elo(features: Dict[str, float], probs: Dict[str, float]) -> Dict[str, float]:
    """
    Safe + explainable draw boost:
    - Only boosts draw when teams are close (small |elo_diff_pre|)
    - Keeps probabilities normalized
    - Uses your *normalized* elo_diff_pre feature (z-scored), so it works without raw Elo

    Tune:
      max_boost: max additional draw probability when teams are extremely close
      close_z: what counts as "close" in normalized units
    """
    if "elo_diff_pre" not in features:
        return probs

    d = abs(float(features["elo_diff_pre"]))  # normalized difference
    close_z = 0.50
    far_z = 0.90
    max_boost = 0.03

    if d >= far_z:
        return probs

    # linear ramp: strongest at d=0, fades to 0 at d=far_z
    closeness = max(0.0, min(1.0, (far_z - d) / (far_z - 0.0)))
    boost = max_boost * closeness

    # Optional: only apply meaningful boost when actually close
    if d > close_z:
        boost *= 0.5

    h = probs["homeWin"]
    dr = probs["draw"]
    a = probs["awayWin"]

    # Take a little from win/lose proportionally to keep sum stable
    take = boost
    win_mass = h + a
    if win_mass <= 1e-9:
        return probs

    h2 = h - take * (h / win_mass)
    a2 = a - take * (a / win_mass)
    d2 = dr + take

    return _normalize_probs({"homeWin": h2, "draw": d2, "awayWin": a2})


def predict_match(req: PredictRequest, ctx: MatchContext) -> Dict[str, Any]:
    predictor = select_predictor(req)
    features = build_features(req, ctx)

    # Debug (keep while testing)
    print("DEBUG match:", req.home_team, "vs", req.away_team, "date", req.date)
    print("DEBUG features head:", list(features.items())[:8])

    probs_raw = predictor.predict_proba(features)
    print("DEBUG probs_raw:", probs_raw)

    probs = _normalize_probs(probs_raw)

    # ✅ realism boost for draws in close matches
    probs = _draw_boost_close_elo(features, probs)
    print("DEBUG probs_final:", probs)   #Terminal debug

    return {
        "homeWin": round(float(probs["homeWin"]), 4),
        "draw": round(float(probs["draw"]), 4),
        "awayWin": round(float(probs["awayWin"]), 4),
        "model": getattr(predictor, "name", "unknown"),
        "generatedAt": int(time.time()),
    }
