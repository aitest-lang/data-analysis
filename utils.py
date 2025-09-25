import pandas as pd
import numpy as np

# Team aliases for data standardization
TEAM_ALIASES = {
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
    "Rising Pune Supergiant": "Rising Pune Supergiant",
    "Kings XI Punjab": "Punjab Kings",
    "Delhi Daredevils": "Delhi Capitals",
    "Pune Warriors": "Pune Warriors India",
}

def preprocess_data(df):
    """
    Preprocess the IPL dataset
    """
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Standardize team names
    df['batting_team'] = df['batting_team'].replace(TEAM_ALIASES)
    df['bowling_team'] = df['bowling_team'].replace(TEAM_ALIASES)
    df['toss_winner'] = df['toss_winner'].replace(TEAM_ALIASES)
    df['match_won_by'] = df['match_won_by'].replace(TEAM_ALIASES)
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Create additional useful columns
    df['is_boundary'] = ~df['runs_not_boundary']
    df['is_four'] = (df['runs_batter'] == 4) & df['is_boundary']
    df['is_six'] = (df['runs_batter'] == 6) & df['is_boundary']
    df['is_wicket'] = df['wicket_kind'].notna()
    df['is_powerplay'] = df['over'] <= 5
    df['is_death_overs'] = df['over'] >= 16
    
    # Fix data types
    df['runs_target'] = pd.to_numeric(df['runs_target'], errors='coerce')
    
    return df

def get_season_runs_trend(df):
    """Get total runs scored by season"""
    season_runs = df.groupby('season').agg({
        'runs_total': 'sum',
        'match_id': 'nunique'
    }).reset_index()
    season_runs.columns = ['season', 'total_runs', 'matches']
    season_runs['runs_per_match'] = season_runs['total_runs'] / season_runs['matches']
    return season_runs

def get_team_match_counts(df):
    """Get match counts by team"""
    batting_matches = df.groupby('batting_team')['match_id'].nunique().reset_index()
    bowling_matches = df.groupby('bowling_team')['match_id'].nunique().reset_index()
    
    # Combine batting and bowling matches
    all_matches = pd.merge(batting_matches, bowling_matches, 
                          left_on='batting_team', right_on='bowling_team', 
                          how='outer', suffixes=('_bat', '_bowl'))
    
    all_matches['team'] = all_matches['batting_team'].fillna(all_matches['bowling_team'])
    all_matches['matches'] = all_matches[['match_id_bat', 'match_id_bowl']].max(axis=1)
    
    return all_matches[['team', 'matches']].sort_values('matches', ascending=False)

def calculate_strike_rate(runs, balls):
    """Calculate strike rate"""
    if balls == 0:
        return 0
    return (runs / balls) * 100

def calculate_economy_rate(runs, overs):
    """Calculate economy rate"""
    if overs == 0:
        return 0
    return runs / overs

def get_match_results(df):
    """Get match results for win/loss analysis"""
    match_results = df.groupby(['match_id', 'match_won_by']).first().reset_index()
    return match_results[['match_id', 'match_won_by', 'toss_winner', 'toss_decision', 'venue', 'season']]

def safe_divide(numerator, denominator, default=0):
    """Safe division to avoid divide by zero errors"""
    if denominator == 0:
        return default
    return numerator / denominator

def get_team_stats(df, team):
    """Get comprehensive team statistics"""
    team_matches = get_match_results(df)
    team_matches = team_matches[(team_matches['match_won_by'] == team) | 
                               (team_matches['toss_winner'] == team)]
    
    matches_played = len(team_matches[team_matches['toss_winner'] == team])
    matches_won = len(team_matches[team_matches['match_won_by'] == team])
    
    # Batting stats
    batting_df = df[df['batting_team'] == team]
    total_runs = batting_df['runs_batter'].sum()
    total_balls = batting_df[batting_df['valid_ball'] == 1].shape[0]
    
    stats = {
        'matches': matches_played,
        'wins': matches_won,
        'win_rate': safe_divide(matches_won, matches_played) * 100,
        'avg_runs': safe_divide(total_runs, matches_played),
        'strike_rate': calculate_strike_rate(total_runs, total_balls)
    }
    
    return stats

def get_player_stats_base(df, group_col, runs_col='runs_batter', balls_col='balls_faced'):
    """Base function for player statistics"""
    stats = df.groupby(group_col).agg({
        runs_col: 'sum',
        balls_col: 'sum',
        'match_id': 'nunique',
        'is_boundary': 'sum',
        'is_four': 'sum',
        'is_six': 'sum'
    }).reset_index()
    
    stats['strike_rate'] = stats.apply(
        lambda row: calculate_strike_rate(row[runs_col], row[balls_col]), axis=1
    )
    
    stats['avg'] = stats.apply(
        lambda row: safe_divide(row[runs_col], row['match_id']), axis=1
    )
    
    return stats
