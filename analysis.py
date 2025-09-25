import pandas as pd
import numpy as np
from utils import *

def get_toss_impact(df, team):
    """Get toss impact analysis for a specific team"""
    match_results = get_match_results(df)
    team_matches = match_results[match_results['toss_winner'] == team].copy()
    
    if len(team_matches) == 0:
        return pd.DataFrame({'toss_won': ['Yes'], 'win_percentage': [0]})
    
    team_matches['won_match'] = team_matches['match_won_by'] == team
    
    toss_impact = team_matches.groupby('won_match').size().reset_index(name='matches')
    total_matches = len(team_matches)
    
    result = []
    won_toss_won_match = len(team_matches[team_matches['won_match'] == True])
    won_toss_lost_match = total_matches - won_toss_won_match
    
    result.append({
        'toss_won': 'Won Match',
        'matches': won_toss_won_match,
        'win_percentage': safe_divide(won_toss_won_match, total_matches) * 100
    })
    
    result.append({
        'toss_won': 'Lost Match', 
        'matches': won_toss_lost_match,
        'win_percentage': safe_divide(won_toss_lost_match, total_matches) * 100
    })
    
    return pd.DataFrame(result)

def get_venue_performance(df, team):
    """Get venue-wise performance for a team"""
    match_results = get_match_results(df)
    team_matches = match_results[
        (match_results['toss_winner'] == team) | 
        (match_results['match_won_by'] == team)
    ].copy()
    
    venue_stats = []
    for venue in team_matches['venue'].unique():
        venue_matches = team_matches[team_matches['venue'] == venue]
        matches_played = len(venue_matches)
        matches_won = len(venue_matches[venue_matches['match_won_by'] == team])
        
        venue_stats.append({
            'venue': venue,
            'matches': matches_played,
            'wins': matches_won,
            'win_rate': safe_divide(matches_won, matches_played) * 100
        })
    
    return pd.DataFrame(venue_stats)

def get_batting_leaderboard(df):
    """Get batting leaderboard with comprehensive stats"""
    batting_df = df[df['valid_ball'] == 1].copy()
    
    batting_stats = batting_df.groupby('batter').agg({
        'runs_batter': 'sum',
        'balls_faced': 'sum', 
        'match_id': 'nunique',
        'is_boundary': 'sum',
        'is_four': 'sum',
        'is_six': 'sum'
    }).reset_index()
    
    # Filter players with minimum matches
    batting_stats = batting_stats[batting_stats['match_id'] >= 10]
    
    batting_stats['strike_rate'] = batting_stats.apply(
        lambda row: calculate_strike_rate(row['runs_batter'], row['balls_faced']), axis=1
    )
    
    batting_stats['average'] = batting_stats.apply(
        lambda row: safe_divide(row['runs_batter'], row['match_id']), axis=1
    )
    
    batting_stats['boundary_percentage'] = batting_stats.apply(
        lambda row: safe_divide(row['is_boundary'], row['balls_faced']) * 100, axis=1
    )
    
    batting_stats.columns = ['batter', 'total_runs', 'balls_faced', 'matches', 
                           'boundaries', 'fours', 'sixes', 'strike_rate', 
                           'average', 'boundary_percentage']
    
    return batting_stats.sort_values('total_runs', ascending=False)

def get_bowling_leaderboard(df):
    """Get bowling leaderboard with comprehensive stats"""
    bowling_df = df[df['valid_ball'] == 1].copy()
    
    bowling_stats = bowling_df.groupby('bowler').agg({
        'runs_bowler': 'sum',
        'valid_ball': 'sum',
        'is_wicket': 'sum',
        'match_id': 'nunique'
    }).reset_index()
    
    # Filter bowlers with minimum matches
    bowling_stats = bowling_stats[bowling_stats['match_id'] >= 10]
    
    bowling_stats['overs'] = bowling_stats['valid_ball'] / 6
    bowling_stats['economy_rate'] = bowling_stats.apply(
        lambda row: calculate_economy_rate(row['runs_bowler'], row['overs']), axis=1
    )
    
    bowling_stats['average'] = bowling_stats.apply(
        lambda row: safe_divide(row['runs_bowler'], row['is_wicket'], 0), axis=1
    )
    
    bowling_stats['strike_rate'] = bowling_stats.apply(
        lambda row: safe_divide(row['valid_ball'], row['is_wicket'], 0), axis=1
    )
    
    bowling_stats.columns = ['bowler', 'runs_conceded', 'balls_bowled', 
                           'total_wickets', 'matches', 'overs', 'economy_rate',
                           'bowling_average', 'bowling_strike_rate']
    
    return bowling_stats.sort_values('total_wickets', ascending=False)

def get_overall_toss_impact(df):
    """Get overall toss impact across all matches"""
    match_results = get_match_results(df)
    
    toss_impact = []
    for _, match in match_results.iterrows():
        toss_winner = match['toss_winner'] 
        match_winner = match['match_won_by']
        
        if pd.notna(toss_winner) and pd.notna(match_winner):
            if toss_winner == match_winner:
                toss_impact.append({'result': 'Toss Winner Won', 'match_id': match['match_id']})
            else:
                toss_impact.append({'result': 'Toss Winner Lost', 'match_id': match['match_id']})
    
    toss_df = pd.DataFrame(toss_impact)
    if len(toss_df) == 0:
        return pd.DataFrame({'result': ['No Data'], 'matches': [0]})
    
    return toss_df.groupby('result').size().reset_index(name='matches')

def get_venue_analysis(df):
    """Get venue-wise scoring analysis"""
    venue_stats = df.groupby(['venue', 'match_id']).agg({
        'team_runs': 'first'  # Get team total for each innings
    }).reset_index()
    
    venue_summary = venue_stats.groupby('venue').agg({
        'team_runs': ['mean', 'count'],
        'match_id': 'nunique'
    }).reset_index()
    
    venue_summary.columns = ['venue', 'avg_runs', 'innings_count', 'matches']
    venue_summary = venue_summary[venue_summary['matches'] >= 5]  # Min 5 matches
    
    return venue_summary.sort_values('avg_runs', ascending=False)

def get_season_trends(df):
    """Get season-wise trends"""
    season_stats = df.groupby(['season', 'match_id']).agg({
        'team_runs': 'first',
        'is_boundary': 'sum'
    }).reset_index()
    
    season_summary = season_stats.groupby('season').agg({
        'team_runs': 'mean',
        'is_boundary': 'mean',
        'match_id': 'nunique'
    }).reset_index()
    
    season_summary.columns = ['season', 'avg_runs', 'boundaries_per_match', 'matches']
    
    return season_summary

def get_powerplay_analysis(df):
    """Get powerplay performance analysis"""
    powerplay_df = df[df['is_powerplay'] == True].copy()
    
    pp_stats = powerplay_df.groupby(['batting_team', 'match_id']).agg({
        'runs_batter': 'sum',
        'is_wicket': 'sum'
    }).reset_index()
    
    pp_summary = pp_stats.groupby('batting_team').agg({
        'runs_batter': 'mean',
        'is_wicket': 'mean',
        'match_id': 'nunique'
    }).reset_index()
    
    pp_summary.columns = ['team', 'pp_runs_per_match', 'pp_wickets_per_match', 'matches']
    pp_summary = pp_summary[pp_summary['matches'] >= 5]
    
    return pp_summary.sort_values('pp_runs_per_match', ascending=False)

def get_death_overs_analysis(df):
    """Get death overs performance analysis"""
    death_df = df[df['is_death_overs'] == True].copy()
    
    death_stats = death_df.groupby(['batting_team', 'match_id']).agg({
        'runs_batter': 'sum',
        'is_wicket': 'sum'
    }).reset_index()
    
    death_summary = death_stats.groupby('batting_team').agg({
        'runs_batter': 'mean',
        'is_wicket': 'mean', 
        'match_id': 'nunique'
    }).reset_index()
    
    death_summary.columns = ['team', 'death_runs_per_match', 'death_wickets_per_match', 'matches']
    death_summary = death_summary[death_summary['matches'] >= 5]
    
    return death_summary

def get_partnership_analysis(df):
    """Get partnership analysis"""
    # This is a simplified version - in reality partnerships are more complex to calculate
    partnership_df = df[df['valid_ball'] == 1].copy()
    
    # Group by team and match to get basic partnership info
    team_partnerships = partnership_df.groupby(['batting_team', 'match_id']).agg({
        'runs_batter': 'sum',
        'match_id': 'nunique'
    }).reset_index()
    
    partnership_summary = team_partnerships.groupby('batting_team').agg({
        'runs_batter': 'mean'
    }).reset_index()
    
    partnership_summary.columns = ['team', 'avg_partnership']
    
    return partnership_summary.sort_values('avg_partnership', ascending=False)

def get_player_of_match_stats(df):
    """Get player of match statistics"""
    pom_stats = df[df['player_of_match'].notna()].groupby('player_of_match').agg({
        'match_id': 'nunique'
    }).reset_index()
    
    pom_stats.columns = ['player', 'pom_awards']
    
    return pom_stats.sort_values('pom_awards', ascending=False)

def get_extras_analysis(df):
    """Get extras analysis"""
    extras_df = df[df['runs_extras'] > 0].copy()
    
    extras_stats = extras_df.groupby(['bowling_team', 'match_id']).agg({
        'runs_extras': 'sum'
    }).reset_index()
    
    extras_summary = extras_stats.groupby('bowling_team').agg({
        'runs_extras': 'mean'
    }).reset_index()
    
    extras_summary.columns = ['team', 'extras_per_match']
    
    return extras_summary.sort_values('extras_per_match', ascending=True)
