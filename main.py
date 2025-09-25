import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import calendar
import math
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')
import zipfile

with zipfile.ZipFile('archive.zip', 'r') as zip_ref:
    zip_ref.extractall()
# Data loading and preprocessing
@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv("IPL.csv")
    
    # Team aliases
    TEAM_ALIASES = {
        "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
        "Rising Pune Supergiants": "Rising Pune Supergiant",
        "Rising Pune Supergiant": "Rising Pune Supergiant",
        "Kings XI Punjab": "Punjab Kings",
        "Delhi Daredevils": "Delhi Capitals",
        "Pune Warriors": "Pune Warriors India",
    }
    
    # Apply team name standardization
    df['batting_team'] = df['batting_team'].replace(TEAM_ALIASES)
    df['bowling_team'] = df['bowling_team'].replace(TEAM_ALIASES)
    df['toss_winner'] = df['toss_winner'].replace(TEAM_ALIASES)
    df['player_of_match'] = df['player_of_match'].replace(TEAM_ALIASES)
    
    # Create derived features
    df['total_runs'] = df['runs_batter'] + df['runs_extras']
    df['ball_number'] = df['over'] * 6 + df['ball']
    df['is_boundary'] = df['runs_batter'].isin([4, 6])
    df['is_wicket'] = df['wicket_kind'].notna()
    df['is_dot'] = df['runs_batter'] == 0
    df['is_powerplay'] = df['over'] <= 6
    df['is_middle'] = (df['over'] > 6) & (df['over'] <= 15)
    df['is_death'] = df['over'] > 15
    
    # Match result calculation
    df['match_result'] = df.apply(
        lambda x: x['batting_team'] if (x['innings'] == 2 and x['runs_total'] > x['runs_target']) or 
        (x['innings'] == 1 and pd.isna(x['runs_target'])) else x['bowling_team'], axis=1
    )
    
    # Create match-level summary
    match_summary = df.groupby(['match_id', 'batting_team']).agg({
        'runs_total': 'max',
        'ball_no': 'max',
        'wicket_kind': 'count'
    }).reset_index()
    match_summary.columns = ['match_id', 'team', 'total_runs', 'total_balls', 'wickets']
    
    return df, match_summary

df, match_summary = load_and_preprocess_data()

# Initialize session state
if 'selected_page' not in st.session_state:
    st.session_state.selected_page = 'Overview'

# Sidebar navigation
st.sidebar.title("🏏 IPL Analytics Platform")
page = st.sidebar.selectbox("Select Analysis", 
                           ['Overview', 'Team Analytics', 'Player Analytics', 'Match Analysis', 'Advanced Analytics'],
                           index=['Overview', 'Team Analytics', 'Player Analytics', 'Match Analysis', 'Advanced Analytics'].index(st.session_state.selected_page))

st.session_state.selected_page = page

# Enhanced styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2e75b6;
        margin-top: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Main content based on selected page
if st.session_state.selected_page == 'Overview':
    st.markdown('<h1 class="main-header">IPL Analytics Platform - Overview</h1>', unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Total Matches", df['match_id'].nunique())
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Total Teams", df['batting_team'].nunique())
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Total Seasons", df['season'].nunique())
        st.markdown('</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric("Total Players", df['batter'].nunique())
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Season-wise match count
    season_matches = df.groupby('season')['match_id'].nunique().reset_index()
    fig_season = px.bar(season_matches, x='season', y='match_id', 
                        title='Matches per Season', 
                        color_discrete_sequence=['#1f4e79'])
    fig_season.update_layout(xaxis_title="Season", yaxis_title="Number of Matches")
    st.plotly_chart(fig_season, use_container_width=True)
    
    # Top performing teams
    team_wins = df[df['innings'] == 2].groupby('batting_team').apply(
        lambda x: (x['runs_total'] > x['runs_target']).sum()
    ).reset_index(name='wins')
    team_wins = team_wins.sort_values('wins', ascending=False).head(10)
    fig_wins = px.bar(team_wins, x='batting_team', y='wins', 
                      title='Top Teams by Wins', 
                      color_discrete_sequence=['#70ad47'])
    fig_wins.update_layout(xaxis_title="Team", yaxis_title="Wins")
    st.plotly_chart(fig_wins, use_container_width=True)
    
    # Most runs by teams
    team_runs = df.groupby('batting_team')['runs_total'].sum().reset_index()
    team_runs = team_runs.sort_values('runs_total', ascending=False).head(10)
    fig_runs = px.bar(team_runs, x='batting_team', y='runs_total', 
                      title='Top Teams by Total Runs', 
                      color_discrete_sequence=['#ffc000'])
    fig_runs.update_layout(xaxis_title="Team", yaxis_title="Total Runs")
    st.plotly_chart(fig_runs, use_container_width=True)
    
    # Match outcomes distribution
    outcome_counts = df[df['innings'] == 2]['win_outcome'].value_counts().reset_index()
    outcome_counts.columns = ['outcome', 'count']
    fig_outcome = px.pie(outcome_counts, values='count', names='outcome', 
                         title='Match Outcome Distribution')
    st.plotly_chart(fig_outcome, use_container_width=True)
    
    # Toss and match outcome relationship
    toss_match = df.groupby(['toss_winner', 'match_won_by']).size().reset_index(name='count')
    fig_toss_match = px.sunburst(toss_match, path=['toss_winner', 'match_won_by'], 
                                 values='count', title='Toss Winner vs Match Winner Relationship')
    st.plotly_chart(fig_toss_match, use_container_width=True)
    
    # Boundary analysis
    boundary_data = df[df['is_boundary']].groupby(['batting_team', 'runs_batter']).size().reset_index(name='count')
    boundary_pivot = boundary_data.pivot(index='batting_team', columns='runs_batter', values='count').fillna(0)
    boundary_pivot.columns = ['Fours', 'Sixes']
    boundary_pivot['Total Boundaries'] = boundary_pivot['Fours'] + boundary_pivot['Sixes']
    boundary_pivot = boundary_pivot.sort_values('Total Boundaries', ascending=False).head(10)
    
    fig_boundaries = px.bar(boundary_pivot, x=boundary_pivot.index, y=['Fours', 'Sixes'],
                            title='Team Boundary Analysis (Top 10)', 
                            barmode='group', 
                            color_discrete_sequence=['#5b9bd5', '#ed7d31'])
    st.plotly_chart(fig_boundaries, use_container_width=True)

elif st.session_state.selected_page == 'Team Analytics':
    st.markdown('<h1 class="main-header">Team Analytics</h1>', unsafe_allow_html=True)
    
    teams = sorted(df['batting_team'].unique())
    selected_team = st.selectbox("Select Team", teams)
    
    team_data = df[df['batting_team'] == selected_team]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        matches_played = team_data['match_id'].nunique()
        st.metric("Matches Played", matches_played)
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        wins = team_data[team_data['innings'] == 2].apply(
            lambda x: 1 if x['runs_total'] > x['runs_target'] else 0, axis=1
        ).sum()
        st.metric("Wins", wins)
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        win_rate = (wins / matches_played * 100) if matches_played > 0 else 0
        st.metric("Win Rate", f"{win_rate:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        total_runs = team_data['runs_total'].max()
        st.metric("Highest Score", int(total_runs))
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Head-to-head analysis
    opponent_teams = sorted([team for team in teams if team != selected_team])
    selected_opponent = st.selectbox("vs Opponent", opponent_teams)
    
    h2h_data = df[((df['batting_team'] == selected_team) & (df['bowling_team'] == selected_opponent)) |
                  ((df['batting_team'] == selected_opponent) & (df['bowling_team'] == selected_team))]
    
    if not h2h_data.empty:
        h2h_summary = h2h_data.groupby('match_id').apply(
            lambda x: x[x['innings'] == 2].iloc[0]['batting_team'] if 
            x[x['innings'] == 2]['runs_total'].iloc[0] > x[x['innings'] == 2]['runs_target'].iloc[0] 
            else x[x['innings'] == 1].iloc[0]['batting_team']
        ).value_counts()
        
        fig_h2h = px.pie(values=h2h_summary.values, names=h2h_summary.index, 
                         title=f"{selected_team} vs {selected_opponent} - Head to Head")
        st.plotly_chart(fig_h2h, use_container_width=True)
    
    # Team performance over seasons
    team_season_performance = df[df['batting_team'] == selected_team].groupby('season').agg({
        'match_id': 'nunique',
        'runs_total': 'max',
        'wicket_kind': 'count'
    }).reset_index()
    team_season_performance.columns = ['season', 'matches', 'highest_score', 'total_wickets']
    
    fig_season_perf = make_subplots(specs=[[{"secondary_y": True}]])
    fig_season_perf.add_trace(
        go.Bar(x=team_season_performance['season'], y=team_season_performance['matches'], 
               name="Matches Played", marker_color='#1f4e79'),
        secondary_y=False,
    )
    fig_season_perf.add_trace(
        go.Scatter(x=team_season_performance['season'], y=team_season_performance['highest_score'], 
                   mode='lines+markers', name="Highest Score", line=dict(color='#70ad47')),
        secondary_y=True,
    )
    
    fig_season_perf.update_layout(title_text=f"{selected_team} Performance Over Seasons")
    fig_season_perf.update_xaxes(title_text="Season")
    fig_season_perf.update_yaxes(title_text="Matches Played", secondary_y=False)
    fig_season_perf.update_yaxes(title_text="Highest Score", secondary_y=True)
    st.plotly_chart(fig_season_perf, use_container_width=True)
    
    # Top 5 players for selected team
    top_players = team_data.groupby('batter')['runs_batter'].sum().sort_values(ascending=False).head(5)
    fig_players = px.bar(top_players, x=top_players.index, y=top_players.values, 
                         title=f'Top 5 Run Scorers for {selected_team}',
                         labels={'y': 'Runs', 'x': 'Player'},
                         color_discrete_sequence=['#ffc000'])
    st.plotly_chart(fig_players, use_container_width=True)
    
    # Team batting analysis over overs
    team_overs = team_data.groupby(['over', 'innings']).agg({
        'runs_total': 'max',
        'wicket_kind': 'count'
    }).reset_index()
    
    fig_overs = make_subplots(specs=[[{"secondary_y": True}]])
    fig_overs.add_trace(
        go.Scatter(x=team_overs['over'], y=team_overs['runs_total'], 
                   mode='lines+markers', name="Runs", line=dict(color='#1f4e79')),
        secondary_y=False,
    )
    fig_overs.add_trace(
        go.Scatter(x=team_overs['over'], y=team_overs['wicket_kind'], 
                   mode='lines+markers', name="Wickets", line=dict(color='#ed7d31')),
        secondary_y=True,
    )
    
    fig_overs.update_layout(title_text=f"{selected_team} - Runs and Wickets Over Overs")
    fig_overs.update_xaxes(title_text="Over")
    fig_overs.update_yaxes(title_text="Runs", secondary_y=False)
    fig_overs.update_yaxes(title_text="Wickets", secondary_y=True)
    st.plotly_chart(fig_overs, use_container_width=True)
    
    # Team bowling analysis
    team_bowling = df[df['bowling_team'] == selected_team].groupby('batter').agg({
        'runs_batter': 'sum',
        'wicket_kind': 'count'
    }).reset_index()
    team_bowling = team_bowling.sort_values('runs_batter', ascending=True).head(10)
    
    fig_bowling = px.bar(team_bowling, x='batter', y='runs_batter', 
                         title=f'Top Batters Conceded by {selected_team}',
                         color_discrete_sequence=['#5b9bd5'])
    st.plotly_chart(fig_bowling, use_container_width=True)

elif st.session_state.selected_page == 'Player Analytics':
    st.markdown('<h1 class="main-header">Player Analytics</h1>', unsafe_allow_html=True)
    
    all_players = sorted(df['batter'].unique())
    selected_player = st.selectbox("Select Player", all_players)
    
    player_data = df[df['batter'] == selected_player]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        total_runs = player_data['runs_batter'].sum()
        st.metric("Total Runs", total_runs)
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        balls_faced = player_data['balls_faced'].sum()
        strike_rate = (total_runs / balls_faced * 100) if balls_faced > 0 else 0
        st.metric("Strike Rate", f"{strike_rate:.1f}")
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        fours = player_data[player_data['runs_batter'] == 4].shape[0]
        sixes = player_data[player_data['runs_batter'] == 6].shape[0]
        st.metric("Boundaries", f"{fours} 4s, {sixes} 6s")
        st.markdown('</div>', unsafe_allow_html=True)
    with col4:
        hundreds = player_data.groupby('match_id')['runs_batter'].sum().apply(
            lambda x: 1 if x >= 100 else 0
        ).sum()
        fifties = player_data.groupby('match_id')['runs_batter'].sum().apply(
            lambda x: 1 if 50 <= x < 100 else 0
        ).sum()
        st.metric("Centuries/Fifties", f"{hundreds}/{fifties}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Runs per season
    player_season_runs = player_data.groupby('season')['runs_batter'].sum().reset_index()
    fig_runs = px.line(player_season_runs, x='season', y='runs_batter', 
                       title=f'{selected_player} - Runs per Season',
                       markers=True, line_shape='spline')
    fig_runs.update_layout(xaxis_title="Season", yaxis_title="Runs")
    st.plotly_chart(fig_runs, use_container_width=True)
    
    # Performance against teams
    player_vs_teams = player_data.groupby('bowling_team')['runs_batter'].sum().reset_index()
    player_vs_teams = player_vs_teams.sort_values('runs_batter', ascending=False)
    fig_teams = px.bar(player_vs_teams, x='bowling_team', y='runs_batter', 
                       title=f'{selected_player} - Performance Against Teams',
                       color_discrete_sequence=['#70ad47'])
    fig_teams.update_layout(xaxis_title="Bowling Team", yaxis_title="Runs")
    st.plotly_chart(fig_teams, use_container_width=True)
    
    # Dismissal types
    dismissals = player_data[player_data['wicket_kind'].notna()]['wicket_kind'].value_counts()
    if not dismissals.empty:
        fig_dismissal = px.pie(values=dismissals.values, names=dismissals.index, 
                               title=f'{selected_player} - Dismissal Types')
        st.plotly_chart(fig_dismissal, use_container_width=True)
    
    # Player performance over match stages
    player_stage = player_data.groupby('stage').agg({
        'runs_batter': 'sum',
        'balls_faced': 'sum'
    }).reset_index()
    player_stage['strike_rate'] = (player_stage['runs_batter'] / player_stage['balls_faced']) * 100
    fig_stage = px.bar(player_stage, x='stage', y='strike_rate', 
                       title=f'{selected_player} - Strike Rate by Match Stage',
                       color_discrete_sequence=['#ffc000'])
    fig_stage.update_layout(xaxis_title="Stage", yaxis_title="Strike Rate")
    st.plotly_chart(fig_stage, use_container_width=True)
    
    # Player consistency analysis
    player_matches = player_data.groupby('match_id')['runs_batter'].sum().reset_index()
    player_matches.columns = ['match_id', 'runs']
    player_matches['performance_category'] = player_matches['runs'].apply(
        lambda x: '0-10' if x <= 10 else 
        '11-25' if x <= 25 else 
        '26-50' if x <= 50 else 
        '51-100' if x <= 100 else '100+'
    )
    
    performance_dist = player_matches['performance_category'].value_counts()
    fig_consistency = px.pie(values=performance_dist.values, names=performance_dist.index, 
                             title=f'{selected_player} - Performance Distribution')
    st.plotly_chart(fig_consistency, use_container_width=True)

elif st.session_state.selected_page == 'Match Analysis':
    st.markdown('<h1 class="main-header">Match Analysis</h1>', unsafe_allow_html=True)
    
    matches = sorted(df['match_id'].unique())
    selected_match = st.selectbox("Select Match ID", matches)
    
    match_data = df[df['match_id'] == selected_match]
    
    # Match summary
    if not match_data.empty:
        team1 = match_data[match_data['innings'] == 1]['batting_team'].iloc[0]
        team2 = match_data[match_data['innings'] == 2]['batting_team'].iloc[0]
        
        team1_score = match_data[match_data['innings'] == 1]['runs_total'].max()
        team2_score = match_data[match_data['innings'] == 2]['runs_total'].max()
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"🏏 {team1}: {team1_score}")
        with col2:
            st.subheader(f"🏏 {team2}: {team2_score}")
        
        # Run rate over overs
        match_overs = match_data.groupby(['innings', 'over'])['runs_total'].max().reset_index()
        fig_match = px.line(match_overs, x='over', y='runs_total', color='innings', 
                            title='Run Rate Over Overs', 
                            labels={'innings': 'Innings', 'over': 'Overs', 'runs_total': 'Runs'})
        st.plotly_chart(fig_match, use_container_width=True)
        
        # Wickets over overs
        wickets_data = match_data[match_data['is_wicket']].groupby(['innings', 'over']).size().reset_index(name='wickets')
        if not wickets_data.empty:
            fig_wickets = px.bar(wickets_data, x='over', y='wickets', color='innings', 
                                 title='Wickets Over Overs')
            st.plotly_chart(fig_wickets, use_container_width=True)
        
        # Powerplay analysis
        powerplay_data = match_data[match_data['is_powerplay']]
        if not powerplay_data.empty:
            pp_runs = powerplay_data.groupby('innings')['runs_total'].max()
            pp_wickets = powerplay_data.groupby('innings')['wicket_kind'].count()
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader(f"Powerplay Runs: Innings 1: {pp_runs.get(1, 0)}, Innings 2: {pp_runs.get(2, 0)}")
            with col2:
                st.subheader(f"Powerplay Wickets: Innings 1: {pp_wickets.get(1, 0)}, Innings 2: {pp_wickets.get(2, 0)}")
        
        # Partnership analysis
        partnerships = match_data.groupby(['match_id', 'batting_partners'])['runs_batter'].sum().reset_index()
        partnerships = partnerships.sort_values('runs_batter', ascending=False).head(10)
        if not partnerships.empty:
            fig_partnerships = px.bar(partnerships, x='batting_partners', y='runs_batter', 
                                      title='Top Partnerships in Match',
                                      color_discrete_sequence=['#5b9bd5'])
            st.plotly_chart(fig_partnerships, use_container_width=True)
        
        # Match result prediction
        team1_total = match_data[match_data['innings'] == 1]['runs_total'].max()
        team2_runs = match_data[match_data['innings'] == 2]['runs_total'].max()
        target = match_data[match_data['innings'] == 2]['runs_target'].max()
        
        st.subheader("Match Result Analysis")
        if team2_runs >= target:
            st.success(f"{team2} won by {20 - match_data[match_data['innings'] == 2]['over'].max()} wickets remaining!")
        else:
            required_runs = target - team2_runs
            st.error(f"{team1} won by {required_runs} runs!")

elif st.session_state.selected_page == 'Advanced Analytics':
    st.markdown('<h1 class="main-header">Advanced Analytics</h1>', unsafe_allow_html=True)
    
    st.subheader("📊 Venue Performance Analysis")
    venues = sorted(df['venue'].unique())
    selected_venue = st.selectbox("Select Venue", venues)
    
    venue_data = df[df['venue'] == selected_venue]
    
    if not venue_data.empty:
        venue_stats = venue_data.groupby('batting_team').agg({
            'runs_total': 'mean',
            'match_id': 'nunique',
            'wicket_kind': 'count'
        }).reset_index()
        venue_stats.columns = ['Team', 'Avg_Score', 'Matches', 'Total_Wickets']
        venue_stats = venue_stats[venue_stats['Matches'] > 2]  # Filter for teams with more than 2 matches
        
        fig_venue = make_subplots(specs=[[{"secondary_y": True}]])
        fig_venue.add_trace(
            go.Bar(x=venue_stats['Team'], y=venue_stats['Avg_Score'], 
                   name="Average Score", marker_color='#1f4e79'),
            secondary_y=False,
        )
        fig_venue.add_trace(
            go.Scatter(x=venue_stats['Team'], y=venue_stats['Total_Wickets'], 
                       mode='lines+markers', name="Total Wickets", line=dict(color='#70ad47')),
            secondary_y=True,
        )
        
        fig_venue.update_layout(title_text=f'Average Score and Wickets at {selected_venue}')
        fig_venue.update_xaxes(title_text="Team")
        fig_venue.update_yaxes(title_text="Average Score", secondary_y=False)
        fig_venue.update_yaxes(title_text="Total Wickets", secondary_y=True)
        st.plotly_chart(fig_venue, use_container_width=True)
    
    st.subheader("🎯 Toss Impact Analysis")
    toss_impact = df.groupby('toss_winner').apply(
        lambda x: ((x['toss_winner'] == x['batting_team']) & (x['innings'] == 2) & 
                   (x['runs_total'] > x['runs_target'])).sum() + 
                  ((x['toss_winner'] != x['batting_team']) & (x['innings'] == 1) & 
                   (x['runs_total'] > x['runs_target'])).sum()
    ).reset_index(name='wins_with_toss')
    
    toss_total = df.groupby('toss_winner')['match_id'].nunique().reset_index(name='total_matches')
    toss_analysis = pd.merge(toss_impact, toss_total, on='toss_winner')
    toss_analysis['win_rate_with_toss'] = (toss_analysis['wins_with_toss'] / toss_analysis['total_matches']) * 100
    
    fig_toss = px.bar(toss_analysis, x='toss_winner', y='win_rate_with_toss', 
                      title='Win Rate When Winning Toss',
                      color_discrete_sequence=['#ffc000'])
    fig_toss.update_layout(xaxis_title="Team", yaxis_title="Win Rate (%)")
    st.plotly_chart(fig_toss, use_container_width=True)
    
    st.subheader("📅 Season Performance Trends")
    season_performance = df.groupby(['season', 'batting_team']).agg({
        'runs_total': 'mean',
        'match_id': 'nunique'
    }).reset_index()
    season_performance = season_performance[season_performance['match_id'] > 2]  # Filter for teams with more than 2 matches per season
    
    fig_season_trend = px.line(season_performance, x='season', y='runs_total', color='batting_team',
                               title='Average Score Trend Across Seasons',
                               line_shape='spline')
    st.plotly_chart(fig_season_trend, use_container_width=True)
    
    st.subheader("🔥 Boundary Analysis")
    boundary_stats = df.groupby('batting_team').agg({
        'is_boundary': 'sum',
        'runs_batter': 'sum',
        'ball_no': 'count'
    }).reset_index()
    boundary_stats['boundary_percentage'] = (boundary_stats['is_boundary'] / boundary_stats['ball_no']) * 100
    boundary_stats['boundary_runs_percentage'] = (boundary_stats['is_boundary'] * 4 / boundary_stats['runs_batter']) * 100
    
    fig_boundary = px.scatter(boundary_stats, x='boundary_percentage', y='boundary_runs_percentage',
                              size='runs_batter', color='batting_team',
                              title='Boundary Percentage vs Boundary Runs Percentage',
                              hover_data=['batting_team'])
    st.plotly_chart(fig_boundary, use_container_width=True)
    
    st.subheader("🎯 Player Performance Clustering")
    # Create player performance metrics
    player_metrics = df.groupby('batter').agg({
        'runs_batter': 'sum',
        'balls_faced': 'sum',
        'wicket_kind': 'count'
    }).reset_index()
    
    # Calculate derived metrics
    player_metrics['strike_rate'] = (player_metrics['runs_batter'] / player_metrics['balls_faced']) * 100
    player_metrics['runs_per_match'] = player_metrics['runs_batter'] / df.groupby('batter')['match_id'].nunique().values
    player_metrics = player_metrics[player_metrics['balls_faced'] > 10]  # Filter active players
    
    # Prepare data for clustering
    clustering_data = player_metrics[['strike_rate', 'runs_per_match']].dropna()
    
    if len(clustering_data) > 5:
        # Perform K-means clustering
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(clustering_data)
        
        kmeans = KMeans(n_clusters=4, random_state=42)
        clusters = kmeans.fit_predict(scaled_data)
        
        player_metrics.loc[player_metrics['balls_faced'] > 10, 'cluster'] = clusters
        
        fig_cluster = px.scatter(player_metrics, x='strike_rate', y='runs_per_match', 
                                 color='cluster', hover_data=['batter'],
                                 title='Player Performance Clustering',
                                 color_continuous_scale='viridis')
        st.plotly_chart(fig_cluster, use_container_width=True)
    
    st.subheader("📈 Predictive Analytics - Match Outcome")
    # Create features for match outcome prediction
    match_features = df.groupby(['match_id', 'batting_team']).agg({
        'runs_total': 'max',
        'wicket_kind': 'count',
        'ball_no': 'max'
    }).reset_index()
    
    match_features['team_performance'] = match_features['runs_total'] / match_features['ball_no']
    match_features['wicket_rate'] = match_features['wicket_kind'] / match_features['ball_no']
    
    # Calculate match results
    match_results = df.groupby('match_id').apply(
        lambda x: x[x['innings'] == 2].iloc[0]['batting_team'] if 
        x[x['innings'] == 2]['runs_total'].iloc[0] > x[x['innings'] == 2]['runs_target'].iloc[0] 
        else x[x['innings'] == 1].iloc[0]['batting_team']
    ).reset_index(name='winner')
    
    # Merge with features
    match_analysis = pd.merge(match_features, match_results, left_on='match_id', right_on='match_id')
    match_analysis['is_winner'] = match_analysis['batting_team'] == match_analysis['winner']
    
    fig_outcome = px.scatter(match_analysis, x='team_performance', y='wicket_rate', 
                             color='is_winner', hover_data=['match_id', 'batting_team'],
                             title='Team Performance vs Wicket Rate - Match Outcome Analysis')
    st.plotly_chart(fig_outcome, use_container_width=True)

# Add comprehensive footer
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    st.markdown("**IPL Analytics Platform**")
with col2:
    st.markdown("*Advanced Data Analysis for Indian Premier League*")
with col3:
    st.markdown(f"*Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
