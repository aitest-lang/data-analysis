import pandas as pd
import numpy as np
from datetime import datetime

def calculate_win_probability(runs_remaining, balls_remaining, wickets_remaining, required_rate):
    """Calculate win probability based on match situation"""
    if runs_remaining <= 0:
        return 100.0
    if balls_remaining <= 0:
        return 0.0
    
    current_rate = runs_remaining / (balls_remaining / 6)
    
    # Simple probability model based on required run rate and resources
    base_prob = max(0, min(100, 60 - (current_rate - 6) * 10))
    
    # Adjust for wickets remaining
    wicket_factor = wickets_remaining / 10
    
    return min(100, max(0, base_prob * wicket_factor))

def get_player_form(df, player_name, player_type='batter', recent_matches=5):
    """Get recent form of a player"""
    if player_type == 'batter':
        player_data = df[df['batter'] == player_name]
        form_data = player_data.groupby('match_id')['runs_batter'].sum().tail(recent_matches)
    else:
        player_data = df[df['bowler'] == player_name]
        form_data = player_data.groupby('match_id').agg({
            'runs_bowler': 'sum',
            'wicket_kind': 'count'
        }).tail(recent_matches)
    
    return form_data

def analyze_powerplay_performance(df, team_name):
    """Analyze powerplay performance for a team"""
    powerplay_data = df[(df['over'] <= 6) & (df['batting_team'] == team_name)]
    
    pp_stats = {
        'total_runs': powerplay_data['runs_total'].sum(),
        'total_balls': powerplay_data['valid_ball'].sum(),
        'wickets_lost': powerplay_data['wicket_kind'].count(),
        'strike_rate': powerplay_data['runs_total'].sum() / powerplay_data['valid_ball'].sum() * 100,
        'average_score': powerplay_data.groupby('match_id')['runs_total'].sum().mean()
    }
    
    return pp_stats
