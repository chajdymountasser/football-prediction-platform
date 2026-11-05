# app/ml/predictors/logreg_model.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional
import json

import joblib
import numpy as np

from .base import BasePredictor


class LogisticRegressionPredictor(BasePredictor):
    """
    Loads a trained sklearn LogisticRegression model from app/ml/artifacts/
    and uses app/ml/gamedb/feature_order.json to build X in the exact order
    used during training.
    """

    def __init__(self, model_id: str = "logreg_v1", artifact_name: Optional[str] = None):
        # IMPORTANT: BasePredictor expects .name
        self.name = str(model_id).strip() if model_id else "logreg_v1"

        # Allow passing full filename, otherwise build "<model_id>.joblib"
        if artifact_name is None:
            artifact_name = self.name
            if not artifact_name.endswith(".joblib"):
                artifact_name += ".joblib"

        app_dir = Path(__file__).resolve().parents[2]  # .../backend/app

        model_path = app_dir / "ml" / "artifacts" / artifact_name
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model artifact: {model_path}")

        feature_order_path = app_dir / "ml" / "gamedb" / "feature_order.json"
        if not feature_order_path.exists():
            raise FileNotFoundError(f"Missing feature order file: {feature_order_path}")

        self.model = joblib.load(model_path)

        # feature_order.json is written as a JSON list by your training script
        self.feature_order = json.loads(feature_order_path.read_text(encoding="utf-8"))
        if not isinstance(self.feature_order, list) or len(self.feature_order) == 0:
            raise ValueError("feature_order.json must be a non-empty JSON list of feature names")

    def predict_proba(self, features: Dict[str, float]) -> Dict[str, float]:
        # Fail early with a helpful error if anything is missing
        missing = [k for k in self.feature_order if k not in features]
        if missing:
            raise KeyError(f"Missing {len(missing)} features required by model: {missing[:10]} ...")

        X = np.array([[float(features[k]) for k in self.feature_order]], dtype=float)
        p = self.model.predict_proba(X)[0]  # [p_home, p_draw, p_away] in your label order

        return {"homeWin": float(p[0]), "draw": float(p[1]), "awayWin": float(p[2])}
