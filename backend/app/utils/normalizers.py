from collections import defaultdict

def normalize_fixtures(api_data):
    groups = defaultdict(list)

    for m in api_data.get("matches", []):
        date = m["utcDate"][:10]

        groups[date].append({
            "id": m["id"],
            "status": m["status"].replace("FINISHED", "FT"),
            "kickoff": m["utcDate"][11:16],
            "home": {
                "name": m["homeTeam"]["name"],
                "badge": f"/static/icons/{m['homeTeam']['tla'].lower()}.svg"
            },
            "away": {
                "name": m["awayTeam"]["name"],
                "badge": f"/static/icons/{m['awayTeam']['tla'].lower()}.svg"
            },
            "score": None if m["status"] != "FINISHED" else {
                "home": m["score"]["fullTime"]["home"],
                "away": m["score"]["fullTime"]["away"]
            }
        })

    return {
        "fixtures": [
            {"groupLabel": k, "matches": groups[k]}
            for k in sorted(groups.keys())
        ]
    }
