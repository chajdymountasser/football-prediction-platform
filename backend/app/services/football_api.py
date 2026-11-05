# app/services/football_api.py
import os
import requests

BASE_URL = "https://api.football-data.org/v4"
API_TOKEN = os.getenv("FOOTBALL_DATA_API_KEY")

if not API_TOKEN:
    raise RuntimeError("FOOTBALL_DATA_API_KEY is not set")

HEADERS = {
    "X-Auth-Token": API_TOKEN
}

def get_competitions():
    r = requests.get(f"{BASE_URL}/competitions", headers=HEADERS)
    r.raise_for_status()
    return r.json()

def get_standings(code, season):
    r = requests.get(
        f"{BASE_URL}/competitions/{code}/standings",
        headers=HEADERS,
        params={"season": season}
    )
    r.raise_for_status()
    return r.json()

def get_matches(code, season, matchday=None):
    params = {"season": season}
    if matchday:
        params["matchday"] = matchday

    r = requests.get(
        f"{BASE_URL}/competitions/{code}/matches",
        headers=HEADERS,
        params=params
    )
    r.raise_for_status()
    return r.json()
