import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
TEAMS_DIR = BASE_DIR / "data" / "team_registry"

_TEAM_CACHE = {}

def load_team_map(league_id: str) -> dict:
    """
    Load team → logo mapping for a league.
    Cached in memory after first load.
    """
    if league_id in _TEAM_CACHE:
        return _TEAM_CACHE[league_id]

    file_path = TEAMS_DIR / f"{league_id}.json"

    if not file_path.exists():
        _TEAM_CACHE[league_id] = {}
        return {}

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    _TEAM_CACHE[league_id] = data
    return data
