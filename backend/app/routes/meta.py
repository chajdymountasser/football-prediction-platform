from flask import Blueprint, jsonify

meta_bp = Blueprint("meta", __name__)


def sort_seasons_desc(seasons):
    """
    Sort seasons like '2023-24' so that newest comes first.
    """
    return sorted(
        seasons,
        key=lambda s: int(s.split("-")[0]),
        reverse=True
    )


@meta_bp.route("/meta")
def meta():
    leagues = [
        {
            "id": "PL",
            "label": "Premier League",
            "seasons": sort_seasons_desc([
                "2023-24", "2024-25", "2025-26"
            ]),
            "rules": {
                "ucl": [1, 2, 3, 4],
                "uel": [5],
                "uecl": [6],
                "relegation_auto": [18, 19, 20]
            }
        },
        {
            "id": "BL1",
            "label": "Bundesliga",
            "seasons": sort_seasons_desc([
                "2023-24", "2024-25", "2025-26"
            ]),
            "rules": {
                "ucl": [1, 2, 3, 4],
                "uel": [5],
                "uecl": [6],
                "relegation_auto": [17, 18],
                "relegation_playoff": 16
            }
        },
        {
            "id": "PD",
            "label": "La Liga",
            "seasons": sort_seasons_desc([
                "2023-24", "2024-25", "2025-26"
            ]),
            "rules": {
                "ucl": [1, 2, 3, 4],
                "uel": [5],
                "uecl": [6],
                "relegation_auto": [18, 19, 20]
            }
        },
        {
            "id": "SA",
            "label": "Serie A",
            "seasons": sort_seasons_desc([
                "2023-24", "2024-25", "2025-26"
            ]),
            "rules": {
                "ucl": [1, 2, 3, 4],
                "uel": [5],
                "uecl": [6],
                "relegation_auto": [18, 19, 20]
            }
        },
        {
            "id": "FL1",
            "label": "Ligue 1",
            "seasons": sort_seasons_desc([
                "2023-24", "2024-25", "2025-26"
            ]),
            "rules": {
                "ucl": [1, 2, 3],
                "ucl_qualifier": [4],
                "uel": [5],
                "uecl": [6],
                "relegation_auto": [17, 18],
                "relegation_playoff": 16
            }
        }
    ]

    return jsonify({"leagues": leagues})
