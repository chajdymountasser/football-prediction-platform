# FOOTBALL PREDICTION FEATURES - MULTI-LEAGUE VERSION
======================================================================

## DATASET OVERVIEW
- **Total matches in training set**: 5841
- **Total features**: 29
- **Target variable**: 0=Home Win, 1=Draw, 2=Away Win
- **Leagues included**: Premier League (PL), Bundesliga (BL1), La Liga (PD), Serie A (SA), Ligue 1 (FL1)

## DATA PROCESSING METHODOLOGY
- **Multi-league support**: Data from 5 major European leagues
- **Global averages**: Single league averages used for all leagues (PL averages)
- **Global Elo ratings**: Single Elo rating system across all leagues
- **Time-based splitting**: Consistent chronological splits across all leagues
- **NO DATA LEAKAGE**: Strict chronological order enforced

## TIER 1: CORE TEAM STRENGTH (13 features)
--------------------------------------------------
### Elo Ratings (3 features)
- **home_elo_pre**: Enhanced Elo rating of home team before match
- **away_elo_pre**: Enhanced Elo rating of away team before match
- **elo_diff_pre**: Elo difference (home - away) before match

### Points Per Game (2 features)
- **home_ppg_current**: Home team points per game in current season
- **away_ppg_current**: Away team points per game in current season

### Home/Away Specific PPG (2 features)
- **home_ppg_home**: Home team points per game at home
- **away_ppg_away**: Away team points per game away (proxy using overall PPG)

### Goal Difference Ratio (2 features)
- **home_goal_diff_ratio**: Home team goals scored / goals conceded
- **away_goal_diff_ratio**: Away team goals scored / goals conceded

### Attack/Defense Strength (4 features)
- **home_attack_strength**: Home team attack strength relative to league average (using PL average: 1.4)
- **home_defense_strength**: Home team defense strength relative to league average (using PL average: 1.4)
- **away_attack_strength**: Away team attack strength relative to league average (using PL average: 1.4)
- **away_defense_strength**: Away team defense strength relative to league average (using PL average: 1.4)

## TIER 2: FORM & MOMENTUM (16 features)
--------------------------------------------------
### For Each Team (8 features × 2 teams = 16 features)
- **form_last5**: Weighted points from last 5 matches (recent matches heavier)
- **goal_diff_trend**: Goal difference trend (last 3 vs previous 2 matches)
- **xg_proxy_last5**: Expected goals proxy: (shots_on_target×0.3)+(shots×0.02)
- **clean_sheet_rate_last5**: Clean sheet rate in last 5 matches
- **failed_to_score_rate_last5**: Failed to score rate in last 5 matches
- **btts_rate_last5**: Both teams scored rate in last 5 matches
- **shot_accuracy_last5**: Shot accuracy (shots on target / total shots)
- **points_ewma**: Exponentially weighted moving average of points
