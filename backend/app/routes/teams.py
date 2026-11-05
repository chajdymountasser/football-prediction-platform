# app/routes/teams.py
from flask import Blueprint, request, jsonify
import json, os

teams_bp = Blueprint("teams", __name__)

BASE = os.path.join(os.path.dirname(__file__), "..", "data")

@teams_bp.route("/teams")
def teams():
    league = request.args.get("league")
    if not league:
        return jsonify({})

    #path = os.path.join(BASE, f"{league}.json")
    path = os.path.join(BASE, "team_registry", f"{league}.json")
    if not os.path.exists(path):
        return jsonify({})

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return jsonify({
        name: f"/static/icons/{league}/{logo}"
        for name, logo in data.items()
    })
