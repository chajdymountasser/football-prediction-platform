# app/routes/standings.py
from flask import Blueprint, request, jsonify
from app.services.football_api import get_standings
from app.utils.cache import get_cache, set_cache

standings_bp = Blueprint("standings", __name__)

@standings_bp.route("/standings")
def standings():
    league = request.args.get("league", "PL")
    season_str = request.args.get("season", "2025-26")

    # convert "2025-26" → 2025
    try:
        season = int(season_str.split("-")[0])
    except ValueError:
        return jsonify({"error": "Invalid season format"}), 400

    cache_key = f"standings:{league}:{season}"
    cached = get_cache(cache_key)
    if cached:
        return jsonify(cached)

    raw = get_standings(league, season)
    table = raw["standings"][0]["table"]

    rows = []
    for r in table:
        rows.append({
            "team": r["team"]["name"],
            "pl": r["playedGames"],
            "w": r["won"],
            "d": r["draw"],
            "l": r["lost"],
            "gf": r["goalsFor"],
            "ga": r["goalsAgainst"],
            "gd": r["goalDifference"],
            "pts": r["points"],
            "form": list((r.get("form") or "").replace(",", ""))
        })

    set_cache(cache_key, rows, ttl_seconds=600)
    return jsonify(rows)
