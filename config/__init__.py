OUTPUT_COL = 'target'
ID_COL = 'id'
CURRENT_COLS = ['home_team_name', 'away_team_name', 'match_date',
    'league_name', 'league_id', 'is_cup', 'home_team_coach_id', 'away_team_coach_id'
]
HIST_HOME_COLS = ['home_team_history_match_date_', 'home_team_history_is_play_home_', 'home_team_history_is_cup_', 'home_team_history_goal_'
    , 'home_team_history_opponent_goal_', 'home_team_history_rating_', 'home_team_history_opponent_rating_'
    , 'home_team_history_coach_', 'home_team_history_league_id_'
]
HIST_AWAY_COLS = ['away_team_history_match_date_', 'away_team_history_is_play_home_', 'away_team_history_is_cup_', 'away_team_history_goal_'
    , 'away_team_history_opponent_goal_', 'away_team_history_rating_', 'away_team_history_opponent_rating_'
    , 'away_team_history_coach_', 'away_team_history_league_id_'
]

DATE_COLS = ['match_date'] + ['home_team_history_match_date_'+str(i) for i in range(1, 11)] \
    + ['away_team_history_match_date_'+str(i) for i in range(1, 11)]

DROP_COLS = DATE_COLS + [ID_COL, 'home_team_name', 'away_team_name', 'league_name']

DROP_NA_COL = ['home_team_name']

MIN_MAX_SCALER_COLS = ['league_id', 'home_team_coach_id', 'away_team_coach_id'] + ['home_team_history_league_id_'+str(i) for i in range(1, 11)] \
    + ['away_team_history_league_id_'+str(i) for i in range(1, 11)] \
    + ['home_team_history_coach_'+str(i) for i in range(1, 11)] \
    + ['away_team_history_coach_'+str(i) for i in range(1, 11)]

custom_config = {
    'OUTPUT_COL': OUTPUT_COL
    , 'ID_COL': ID_COL
    , 'CURRENT_COLS': CURRENT_COLS
    , 'HIST_HOME_COLS': HIST_HOME_COLS
    , 'HIST_AWAY_COLS': HIST_AWAY_COLS
    , 'DATE_COLS': DATE_COLS
    , 'DROP_COLS': DROP_COLS
    , 'DROP_NA_COL': DROP_NA_COL
    , 'MIN_MAX_SCALER_COLS': MIN_MAX_SCALER_COLS
}