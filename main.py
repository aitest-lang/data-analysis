import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import zipfile

# Set page configuration
st.set_page_config(
    page_title="IPL Data Analysis Dashboard",
    page_icon="🏏",
    layout="wide"
)

with zipfile.ZipFile('archive.zip', 'r') as zip_ref:
    zip_ref.extractall()
# Load and preprocess data
@st.cache_data
def load_data():
    df = pd.read_csv("IPL.csv")
    
    # Standardize team names
    df['batting_team'] = df['batting_team'].replace('Royal Challengers Bangalore', 'Royal Challengers Bengaluru')
    df['bowling_team'] = df['bowling_team'].replace('Royal Challengers Bangalore', 'Royal Challengers Bengaluru')
    df['match_won_by'] = df['match_won_by'].replace('Royal Challengers Bangalore', 'Royal Challengers Bengaluru')
    df['toss_winner'] = df['toss_winner'].replace('Royal Challengers Bangalore', 'Royal Challengers Bengaluru')
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Extract more date features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    
    # Ensure season column exists - use year as season if not present
    if 'season' not in df.columns:
        df['season'] = df['year'].astype(str)
    
    # Create match phase feature
    def get_match_phase(over):
        if over <= 6:
            return 'Powerplay (1-6)'
        elif over <= 15:
            return 'Middle (7-15)'
        else:
            return 'Death (16-20)'
    
    df['match_phase'] = df['over'].apply(get_match_phase)
    
    # Create run rate feature
    df['run_rate'] = df.apply(lambda x: x['team_runs'] / (x['team_balls'] / 6) if x['team_balls'] > 0 else 0, axis=1)
    
    return df

# Load the data
df = load_data()

# Get unique values for filters
teams = sorted(df['batting_team'].unique())
seasons = sorted(df['season'].unique())
venues = sorted(df['venue'].unique())
cities = sorted(df['city'].unique())

# App title
st.title("🏏 IPL Data Analysis Dashboard")
st.markdown("Analyze IPL data across different dimensions")

# Sidebar for filters
st.sidebar.title("Filters")
selected_seasons = st.sidebar.multiselect("Select Seasons", seasons, default=seasons)
selected_teams = st.sidebar.multiselect("Select Teams", teams, default=teams)

# Filter data based on selections
filtered_df = df[df['season'].isin(selected_seasons) & 
                 (df['batting_team'].isin(selected_teams) | 
                  df['bowling_team'].isin(selected_teams))]

# Analysis category selection
analysis_category = st.selectbox(
    "Select Analysis Category",
    ["Team Analysis", "Player Analysis", "Match Analysis", "Season Analysis", "Over Analysis"]
)

# Team Analysis
if analysis_category == "Team Analysis":
    st.header("Team Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Team wins
        st.subheader("Team Wins")
        team_wins = filtered_df.drop_duplicates('match_id')['match_won_by'].value_counts().reset_index()
        team_wins.columns = ['Team', 'Wins']
        
        fig = px.bar(team_wins, x='Team', y='Wins', title='Team Wins')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Win percentage
        st.subheader("Win Percentage")
        team_matches = pd.concat([
            filtered_df.drop_duplicates('match_id')['batting_team'].value_counts().reset_index().rename(columns={'batting_team': 'Team', 'count': 'Matches'}),
            filtered_df.drop_duplicates('match_id')['bowling_team'].value_counts().reset_index().rename(columns={'bowling_team': 'Team', 'count': 'Matches'})
        ]).groupby('Team')['Matches'].sum().reset_index()
        
        team_stats = team_wins.merge(team_matches, on='Team')
        team_stats['Win Percentage'] = (team_stats['Wins'] / team_stats['Matches']) * 100
        team_stats = team_stats.sort_values('Win Percentage', ascending=False)
        
        fig = px.bar(team_stats, x='Team', y='Win Percentage', title='Win Percentage')
        st.plotly_chart(fig, use_container_width=True)
    
    # Team performance by season
    st.subheader("Team Performance by Season")
    team_season = filtered_df.drop_duplicates(['match_id', 'season']).groupby(['season', 'match_won_by']).size().reset_index(name='wins')
    
    fig = px.line(team_season, x='season', y='wins', color='match_won_by', title='Team Wins by Season')
    st.plotly_chart(fig, use_container_width=True)
    
    # Head-to-head analysis
    st.subheader("Head-to-Head Analysis")
    if len(selected_teams) == 2:
        h2h_matches = filtered_df[
            ((filtered_df['batting_team'] == selected_teams[0]) & (filtered_df['bowling_team'] == selected_teams[1])) |
            ((filtered_df['batting_team'] == selected_teams[1]) & (filtered_df['bowling_team'] == selected_teams[0]))
        ].drop_duplicates('match_id')
        
        h2h_results = h2h_matches['match_won_by'].value_counts().reset_index()
        h2h_results.columns = ['Team', 'Wins']
        
        fig = px.bar(h2h_results, x='Team', y='Wins', title=f'{selected_teams[0]} vs {selected_teams[1]} Head-to-Head')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Select exactly 2 teams for head-to-head analysis")

# Player Analysis
elif analysis_category == "Player Analysis":
    st.header("Player Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top run scorers
        st.subheader("Top Run Scorers")
        top_scorers = filtered_df.groupby('batter')['runs_batter'].sum().reset_index().sort_values('runs_batter', ascending=False).head(15)
        
        fig = px.bar(top_scorers, x='batter', y='runs_batter', title='Top Run Scorers')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Top wicket takers
        st.subheader("Top Wicket Takers")
        top_wicket_takers = filtered_df.groupby('bowler')['bowler_wicket'].sum().reset_index().sort_values('bowler_wicket', ascending=False).head(15)
        
        fig = px.bar(top_wicket_takers, x='bowler', y='bowler_wicket', title='Top Wicket Takers')
        st.plotly_chart(fig, use_container_width=True)
    
    # Batting statistics
    st.subheader("Batting Statistics")
    batting_stats = filtered_df.groupby('batter').agg({
        'runs_batter': 'sum',
        'balls_faced': 'sum',
        'player_out': lambda x: x.notna().sum()
    }).reset_index()
    batting_stats['batting_average'] = batting_stats.apply(lambda x: x['runs_batter'] / x['player_out'] if x['player_out'] > 0 else 0, axis=1)
    batting_stats['strike_rate'] = batting_stats.apply(lambda x: (x['runs_batter'] / x['balls_faced']) * 100 if x['balls_faced'] > 0 else 0, axis=1)
    batting_stats = batting_stats[batting_stats['balls_faced'] > 100].sort_values('batting_average', ascending=False).head(15)
    
    fig = px.scatter(batting_stats, x='strike_rate', y='batting_average', color='batter', 
                     hover_name='batter', size='runs_batter', title='Batting Average vs Strike Rate')
    st.plotly_chart(fig, use_container_width=True)
    
    # Bowling statistics
    st.subheader("Bowling Statistics")
    bowling_stats = filtered_df.groupby('bowler').agg({
        'runs_total': 'sum',
        'valid_ball': 'sum',
        'bowler_wicket': 'sum'
    }).reset_index()
    bowling_stats['overs'] = bowling_stats['valid_ball'] / 6
    bowling_stats['economy'] = bowling_stats.apply(lambda x: x['runs_total'] / x['overs'] if x['overs'] > 0 else 0, axis=1)
    bowling_stats['bowling_average'] = bowling_stats.apply(lambda x: x['runs_total'] / x['bowler_wicket'] if x['bowler_wicket'] > 0 else 0, axis=1)
    bowling_stats = bowling_stats[bowling_stats['overs'] > 50].sort_values('economy').head(15)
    
    fig = px.scatter(bowling_stats, x='bowling_average', y='economy', color='bowler', 
                     hover_name='bowler', size='bowler_wicket', title='Bowling Average vs Economy')
    st.plotly_chart(fig, use_container_width=True)

# Match Analysis
elif analysis_category == "Match Analysis":
    st.header("Match Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Win by runs vs wickets
        st.subheader("Win by Runs vs Wickets")
        win_types = filtered_df.drop_duplicates('match_id').copy()
        win_types['win_type'] = win_types.apply(lambda x: 'Runs' if 'runs' in str(x['win_outcome']).lower() else 'Wickets', axis=1)
        win_type_counts = win_types['win_type'].value_counts().reset_index()
        win_type_counts.columns = ['Win Type', 'Count']
        
        fig = px.pie(win_type_counts, values='Count', names='Win Type', title='Win by Runs vs Wickets')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Toss decision analysis
        st.subheader("Toss Decision Analysis")
        toss_decision = filtered_df.drop_duplicates('match_id')['toss_decision'].value_counts().reset_index()
        toss_decision.columns = ['Decision', 'Count']
        
        fig = px.pie(toss_decision, values='Count', names='Decision', title='Toss Decisions')
        st.plotly_chart(fig, use_container_width=True)
    
    # Toss impact on match result
    st.subheader("Toss Impact on Match Result")
    toss_impact = filtered_df.drop_duplicates('match_id').copy()
    toss_impact['toss_win_match_win'] = toss_impact.apply(lambda x: 1 if x['toss_winner'] == x['match_won_by'] else 0, axis=1)
    toss_impact_summary = toss_impact.groupby('toss_winner')['toss_win_match_win'].mean().reset_index()
    toss_impact_summary.columns = ['Team', 'Toss Win to Match Win Ratio']
    
    fig = px.bar(toss_impact_summary, x='Team', y='Toss Win to Match Win Ratio', 
                 title='Toss Win to Match Win Ratio')
    st.plotly_chart(fig, use_container_width=True)
    
    # Venue analysis
    st.subheader("Venue Analysis")
    venue_stats = filtered_df.drop_duplicates('match_id').groupby(['venue', 'city']).size().reset_index(name='matches')
    venue_stats = venue_stats.sort_values('matches', ascending=False).head(15)
    
    fig = px.bar(venue_stats, x='venue', y='matches', color='city', title='Most Used Venues')
    st.plotly_chart(fig, use_container_width=True)

# Season Analysis
elif analysis_category == "Season Analysis":
    st.header("Season Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Matches per season
        st.subheader("Matches per Season")
        matches_per_season = filtered_df.drop_duplicates('match_id').groupby('season').size().reset_index(name='matches')
        
        fig = px.bar(matches_per_season, x='season', y='matches', title='Matches per Season')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Average runs per match per season
        st.subheader("Average Runs per Match per Season")
        season_runs = filtered_df.drop_duplicates(['match_id', 'season', 'batting_team']).groupby(['season', 'match_id'])['team_runs'].sum().reset_index()
        avg_runs_per_season = season_runs.groupby('season')['team_runs'].mean().reset_index()
        
        fig = px.line(avg_runs_per_season, x='season', y='team_runs', title='Average Runs per Match per Season')
        st.plotly_chart(fig, use_container_width=True)
    
    # Team performance by season
    st.subheader("Team Performance by Season")
    team_season = filtered_df.drop_duplicates(['match_id', 'season']).groupby(['season', 'match_won_by']).size().reset_index(name='wins')
    
    fig = px.line(team_season, x='season', y='wins', color='match_won_by', title='Team Wins by Season')
    st.plotly_chart(fig, use_container_width=True)
    
    # Powerplay performance by season
    st.subheader("Powerplay Performance by Season")
    powerplay = filtered_df[(filtered_df['over'] >= 1) & (filtered_df['over'] <= 6)]
    powerplay_stats = powerplay.groupby(['season', 'batting_team']).agg({
        'runs_total': 'sum',
        'valid_ball': 'count'
    }).reset_index()
    powerplay_stats['overs'] = powerplay_stats['valid_ball'] / 6
    powerplay_stats['run_rate'] = powerplay_stats['runs_total'] / powerplay_stats['overs']
    
    fig = px.line(powerplay_stats, x='season', y='run_rate', color='batting_team', 
                  title='Powerplay Run Rate by Team per Season')
    st.plotly_chart(fig, use_container_width=True)

# Over Analysis
elif analysis_category == "Over Analysis":
    st.header("Over Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Average runs per over
        st.subheader("Average Runs per Over")
        over_runs = filtered_df.groupby('over').agg({
            'runs_total': 'sum',
            'valid_ball': 'count'
        }).reset_index()
        over_runs['overs'] = over_runs['valid_ball'] / 6
        over_runs['avg_runs'] = over_runs['runs_total'] / over_runs['overs']
        
        fig = px.line(over_runs, x='over', y='avg_runs', title='Average Runs per Over')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Wickets per over
        st.subheader("Wickets per Over")
        over_wickets = filtered_df.groupby('over')['team_wicket'].sum().reset_index()
        
        fig = px.bar(over_wickets, x='over', y='team_wicket', title='Wickets per Over')
        st.plotly_chart(fig, use_container_width=True)
    
    # Run rate comparison by team
    st.subheader("Run Rate Comparison by Team by Over")
    team_over_runs = filtered_df.groupby(['batting_team', 'over']).agg({
        'runs_total': 'sum',
        'valid_ball': 'count'
    }).reset_index()
    team_over_runs['overs'] = team_over_runs['valid_ball'] / 6
    team_over_runs['run_rate'] = team_over_runs['runs_total'] / team_over_runs['overs']
    
    fig = px.line(team_over_runs, x='over', y='run_rate', color='batting_team', 
                  title='Run Rate by Team by Over')
    st.plotly_chart(fig, use_container_width=True)
    
    # Match phase analysis
    st.subheader("Match Phase Analysis")
    phase_stats = filtered_df.groupby(['batting_team', 'match_phase']).agg({
        'runs_total': 'sum',
        'valid_ball': 'count',
        'team_wicket': 'sum'
    }).reset_index()
    phase_stats['overs'] = phase_stats['valid_ball'] / 6
    phase_stats['run_rate'] = phase_stats['runs_total'] / phase_stats['overs']
    phase_stats['wicket_rate'] = phase_stats['team_wicket'] / phase_stats['overs']
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Run rate by phase
        fig = px.bar(phase_stats, x='batting_team', y='run_rate', color='match_phase', barmode='group',
                     title='Run Rate by Match Phase')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Wicket rate by phase
        fig = px.bar(phase_stats, x='batting_team', y='wicket_rate', color='match_phase', barmode='group',
                     title='Wicket Rate by Match Phase')
        st.plotly_chart(fig, use_container_width=True)

# Data Table
st.header("Data Sample")
st.write(filtered_df.head(100))
