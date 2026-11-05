# app/ml/predictors/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class BasePredictor(ABC):
    """
    All predictors must implement predict_proba(features).

    Output must be:
      {"homeWin": float, "draw": float, "awayWin": float}
    Each in [0, 1], sum ~= 1 (manager will normalize again to be safe).
    """

    name: str = "base"

    @abstractmethod
    def predict_proba(self, features: Dict[str, Any]) -> Dict[str, float]:
        raise NotImplementedError
