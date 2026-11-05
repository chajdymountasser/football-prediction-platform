# app/routes/predict.py
from __future__ import annotations

from flask import Blueprint, request, jsonify, current_app

from app.utils.cache import get_cache, set_cache
from app.ml.types import PredictRequest
from app.ml.manager import predict_match


predict_bp = Blueprint("predict", __name__)


# ----------------------------
# helpers
# ----------------------------
def _pred_cache_key(league: str, date: str, home: str, away: str) -> str:
    return f"pred:{league}:{date}:{home}:{away}"


def _require_str(payload: dict, field: str) -> str:
    v = payload.get(field)
    if v is None:
        raise ValueError(f"{field} is required")
    s = str(v).strip()
    if not s:
        raise ValueError(f"{field} is required")
    return s

# ----------------------------
# route
# ----------------------------

@predict_bp.route("/predict", methods=["POST"])
def predict_route():
    payload = request.get_json(silent=True) or {}

    # ---- validate input (400)
    try:
        league = _require_str(payload, "league")
        date = _require_str(payload, "date")          # YYYY-MM-DD
        home = _require_str(payload, "homeTeam")
        away = _require_str(payload, "awayTeam")
    except ValueError as e:
        return jsonify({"error": str(e), "code": "BAD_REQUEST"}), 400
    except Exception:
        return jsonify({"error": "Invalid request body", "code": "BAD_REQUEST"}), 400

    # ---- cache
    cache_key = _pred_cache_key(league, date, home, away)
    cached = get_cache(cache_key)
    if isinstance(cached, dict):
        cached.setdefault("debug", {})
        cached["debug"]["cacheHit"] = True
        return jsonify(cached), 200

    # ---- build ML request
    # PredictRequest is now a SIMPLE identity carrier
    req_obj = PredictRequest(
        league=league,
        date=date,
        home_team=home,
        away_team=away,
    )

    # ---- call ML manager
    try:
        # ctx is no longer required for CSV-based inference
        resp = predict_match(req_obj, ctx=None)

        if isinstance(resp, dict):
            resp.setdefault("debug", {})
            resp["debug"]["cacheHit"] = False

        set_cache(cache_key, resp, ttl_seconds=3600)
        return jsonify(resp), 200

    except Exception as e:
        current_app.logger.exception("Prediction engine failed")
        return jsonify({
            "error": "Prediction engine failed",
            "code": "PREDICT_FAILED",
            "detail": str(e) if current_app.debug else None
        }), 500
