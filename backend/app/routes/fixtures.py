# app/routes/fixtures.py
from flask import Blueprint, request, jsonify
from app.services.football_api import get_matches
from app.utils.cache import get_cache, set_cache
from app.utils.normalizers import normalize_fixtures

fixtures_bp = Blueprint("fixtures", __name__)

NOT_FINISHED = {"SCHEDULED", "TIMED", "IN_PLAY", "PAUSED"}  # anything not finished
FINISHED = {"FINISHED"}


def choose_current_matchday(all_matches):
    """
    CURRENT MW logic:
      - If there exists a matchday with ANY match not FINISHED -> return the smallest such matchday
        (means: current MW if ongoing, or next MW if previous MW already finished)
      - If all matches are FINISHED -> return the latest matchday
      - If season not started and we only have future matches -> return the smallest matchday
    """
    md_to_statuses = {}

    for m in all_matches:
        md = m.get("matchday")
        status = m.get("status")
        if md is None or status is None:
            continue
        md_to_statuses.setdefault(md, []).append(status)

    if not md_to_statuses:
        return 1

    # Matchdays that still have something NOT finished
    unfinished_mds = []
    for md, statuses in md_to_statuses.items():
        if any(s != "FINISHED" for s in statuses):
            unfinished_mds.append(md)

    if unfinished_mds:
        # smallest unfinished = "current/next" MW
        return min(unfinished_mds)

    # all finished -> season done or only finished data returned -> show last
    return max(md_to_statuses.keys())


@fixtures_bp.route("/fixtures")
def fixtures():
    league = request.args.get("league", "PL")
    season_str = request.args.get("season", "2025-26")
    matchday = request.args.get("matchday")  # optional
    clubs = request.args.get("clubs")        # optional: "Arsenal,Chelsea"

    # "2025-26" -> 2025
    try:
        season = int(season_str.split("-")[0])
    except ValueError:
        return jsonify({"error": "Invalid season format"}), 400

    auto_matchday = matchday is None or matchday == ""
    matchday_int = None

    if not auto_matchday:
        try:
            matchday_int = int(matchday)
        except ValueError:
            return jsonify({"error": "Invalid matchday"}), 400

    cache_key = f"fixtures:{league}:{season}:{matchday_int if matchday_int is not None else 'AUTO_CURRENT'}:{clubs or 'all'}"
    cached = get_cache(cache_key)
    if cached:
        return jsonify(cached)

    # If auto mode: fetch all season matches (Football-Data supports this),
    # then pick the "current" matchday and filter down.
    if auto_matchday:
        raw_all = get_matches(league, season, matchday=None)
        all_matches = raw_all.get("matches", [])

        chosen_matchday = choose_current_matchday(all_matches)

        filtered_matches = [m for m in all_matches if m.get("matchday") == chosen_matchday]
        raw = {**raw_all, "matches": filtered_matches}
    else:
        chosen_matchday = matchday_int
        raw = get_matches(league, season, matchday=matchday_int)

    normalized = normalize_fixtures(raw)

    # optional club filter (server-side)
    if clubs:
        wanted = set([c.strip() for c in clubs.split(",") if c.strip()])
        if wanted:
            kept_groups = []
            for g in normalized.get("fixtures", []):
                kept = [
                    m for m in g.get("matches", [])
                    if m.get("home", {}).get("name") in wanted or m.get("away", {}).get("name") in wanted
                ]
                if kept:
                    kept_groups.append({"groupLabel": g["groupLabel"], "matches": kept})
            normalized["fixtures"] = kept_groups

    payload = {
        "matchday": chosen_matchday,
        "fixtures": normalized.get("fixtures", []),
    }

    set_cache(cache_key, payload, ttl_seconds=120)
    return jsonify(payload)
