# app/ml/types.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PredictRequest:
    league: str
    date: str          # "YYYY-MM-DD"
    home_team: str
    away_team: str


@dataclass(frozen=True)
class TeamInfo:
    id: Optional[int]
    name: str
    tla: Optional[str] = None


@dataclass(frozen=True)
class MatchContext:
    utc_date: Optional[str]
    status: str
    home: TeamInfo
    away: TeamInfo
