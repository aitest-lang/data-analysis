import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime
import zipfile

with zipfile.ZipFile('archive.zip', 'r') as zip_ref:
    zip_ref.extractall()
# Team aliases mapping
TEAM_ALIASES = {
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
    "Rising Pune Supergiant": "Rising Pune Supergiant",
    "Kings XI Punjab": "Punjab Kings",
    "Delhi Daredevils": "Delhi Capitals",
    "Pune Warriors": "Pune Warriors India",
}

def load_and_preprocess_data():
    """Load and preprocess IPL data"""
    # Load data
    df = pd.read_csv("IPL.csv")
    
    # Standardize team names
    df['batting_team'] = df['batting_team'].replace(TEAM_ALIASES)
    df['bowling_team'] = df['bowling_team'].replace(TEAM_ALIASES)
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Convert season to string to ensure consistent type
    df['season'] = df['season'].astype(str)
    
    # Extract additional date features
    df['day_of_week'] = df['date'].dt.day_name()
    df['is_weekend'] = df['date'].dt.dayofweek >= 5
    
    # Handle missing values
    df = df.fillna({
        'wicket_kind': 'None',
        'player_out': 'None',
        'fielders': 'None',
        'extra_type': 'None',
        'review_batter': 'None',
        'team_reviewed': 'None',
        'review_decision': 'None',
        'umpire': 'None',
        'superover_winner': 'None',
        'method': 'Normal'
    })
    
    # Create match phase feature
    df['phase'] = np.select(
        [df['over'] < 6, (df['over'] >= 6) & (df['over'] < 16), df['over'] >= 16],
        ['Powerplay', 'Middle', 'Death'],
        default='Unknown'
    )
    
    # Create partnership identifier
    df['partnership'] = df.groupby(['match_id', 'innings', 'batting_team']).ngroup()
    
    return df

def create_derived_features(df):
    """Create derived features for analysis"""
    # Batting features
    batting_df = df.groupby(['match_id', 'innings', 'batting_team', 'batter']).agg(
        runs_batter=('runs_batter', 'sum'),
        balls_faced=('valid_ball', 'sum'),
        fours=('runs_batter', lambda x: (x == 4).sum()),
        sixes=('runs_batter', lambda x: (x == 6).sum()),
        dismissals=('player_out', lambda x: (x != 'None').sum())
    ).reset_index()
    
    # Calculate strike rate
    batting_df['strike_rate'] = np.where(
        batting_df['balls_faced'] > 0,
        (batting_df['runs_batter'] / batting_df['balls_faced']) * 100,
        0
    )
    
    # Bowling features
    bowling_df = df.groupby(['match_id', 'innings', 'bowling_team', 'bowler']).agg(
        runs_conceded=('runs_bowler', 'sum'),
        balls_bowled=('valid_ball', 'sum'),
        wickets=('wicket_kind', lambda x: (x != 'None').sum()),
        dots=('runs_total', lambda x: (x == 0).sum()),
        extras=('runs_extras', 'sum')
    ).reset_index()
    
    # Calculate economy rate
    bowling_df['economy_rate'] = np.where(
        bowling_df['balls_bowled'] > 0,
        (bowling_df['runs_conceded'] / bowling_df['balls_bowled']) * 6,
        0
    )
    
    # Team match features
    team_match_df = df.groupby(['match_id', 'innings', 'batting_team']).agg(
        total_runs=('runs_total', 'sum'),
        total_wickets=('wicket_kind', lambda x: (x != 'None').sum()),
        total_balls=('valid_ball', 'sum'),
        extras=('runs_extras', 'sum')
    ).reset_index()
    
    # Calculate run rate
    team_match_df['run_rate'] = np.where(
        team_match_df['total_balls'] > 0,
        (team_match_df['total_runs'] / team_match_df['total_balls']) * 6,
        0
    )
    
    return batting_df, bowling_df, team_match_df

def get_team_performance_data(df, season=None):
    """Get team performance data aggregated by season"""
    # Filter by season if specified
    if season:
        df = df[df['season'] == season]
    
    # Team wins
    team_wins = df[df['innings'] == 2].groupby(['season', 'batting_team', 'match_won_by']).size().reset_index(name='matches')
    team_wins = team_wins[team_wins['batting_team'] == team_wins['match_won_by']]
    team_wins = team_wins.groupby(['season', 'batting_team'])['matches'].sum().reset_index(name='wins')
    
    # Team matches
    team_matches = df.groupby(['season', 'batting_team'])['match_id'].nunique().reset_index(name='matches')
    
    # Merge and calculate win percentage
    team_perf = pd.merge(team_matches, team_wins, on=['season', 'batting_team'], how='left')
    team_perf['win_percentage'] = (team_perf['wins'] / team_perf['matches']) * 100
    team_perf['win_percentage'] = team_perf['win_percentage'].fillna(0)
    
    # Team runs and wickets
    team_stats = df.groupby(['season', 'batting_team']).agg(
        avg_runs=('runs_total', 'mean'),
        avg_wickets=('wicket_kind', lambda x: (x != 'None').mean()),
        avg_run_rate=('runs_total', lambda x: (x.sum() / x.count()) * 6)
    ).reset_index()
    
    # Merge all team stats
    team_perf = pd.merge(team_perf, team_stats, on=['season', 'batting_team'])
    
    return team_perf

def get_player_performance_data(df, role='batter', season=None):
    """Get player performance data"""
    if role == 'batter':
        player_df = batting_df.copy()
        if season:
            player_df = player_df[player_df['match_id'].isin(df[df['season'] == season]['match_id'])]
        
        player_stats = player_df.groupby('batter').agg(
            matches=('match_id', 'nunique'),
            innings=('innings', 'nunique'),
            runs=('runs_batter', 'sum'),
            balls=('balls_faced', 'sum'),
            dismissals=('dismissals', 'sum'),
            fours=('fours', 'sum'),
            sixes=('sixes', 'sum')
        ).reset_index()
        
        # Calculate averages and strike rate
        player_stats['average'] = np.where(
            player_stats['dismissals'] > 0,
            player_stats['runs'] / player_stats['dismissals'],
            player_stats['runs']
        )
        player_stats['strike_rate'] = np.where(
            player_stats['balls'] > 0,
            (player_stats['runs'] / player_stats['balls']) * 100,
            0
        )
        player_stats['fours_per_match'] = player_stats['fours'] / player_stats['matches']
        player_stats['sixes_per_match'] = player_stats['sixes'] / player_stats['matches']
        
    else:  # bowler
        player_df = bowling_df.copy()
        if season:
            player_df = player_df[player_df['match_id'].isin(df[df['season'] == season]['match_id'])]
        
        player_stats = player_df.groupby('bowler').agg(
            matches=('match_id', 'nunique'),
            innings=('innings', 'nunique'),
            wickets=('wickets', 'sum'),
            runs_conceded=('runs_conceded', 'sum'),
            balls_bowled=('balls_bowled', 'sum'),
            dots=('dots', 'sum'),
            extras=('extras', 'sum')
        ).reset_index()
        
        # Calculate economy and bowling average
        player_stats['economy_rate'] = np.where(
            player_stats['balls_bowled'] > 0,
            (player_stats['runs_conceded'] / player_stats['balls_bowled']) * 6,
            0
        )
        player_stats['bowling_average'] = np.where(
            player_stats['wickets'] > 0,
            player_stats['runs_conceded'] / player_stats['wickets'],
            player_stats['runs_conceded']
        )
        player_stats['dot_percentage'] = np.where(
            player_stats['balls_bowled'] > 0,
            (player_stats['dots'] / player_stats['balls_bowled']) * 100,
            0
        )
    
    return player_stats

def get_venue_analysis_data(df):
    """Get venue analysis data"""
    venue_stats = df.groupby('venue').agg(
        matches=('match_id', 'nunique'),
        avg_runs=('runs_total', 'mean'),
        avg_wickets=('wicket_kind', lambda x: (x != 'None').mean()),
        avg_first_innings_score=('runs_total', lambda x: x[df[df['match_id'].isin(x.index)]['innings'] == 1].mean()),
        avg_second_innings_score=('runs_total', lambda x: x[df[df['match_id'].isin(x.index)]['innings'] == 2].mean())
    ).reset_index()
    
    # Calculate win percentage for teams batting first
    first_innings_wins = df[df['innings'] == 1].groupby(['venue', 'batting_team', 'match_won_by']).size().reset_index(name='matches')
    first_innings_wins = first_innings_wins[first_innings_wins['batting_team'] == first_innings_wins['match_won_by']]
    first_innings_wins = first_innings_wins.groupby('venue')['matches'].sum().reset_index(name='first_innings_wins')
    
    # Total first innings matches
    first_innings_matches = df[df['innings'] == 1].groupby('venue')['match_id'].nunique().reset_index(name='first_innings_matches')
    
    # Merge and calculate win percentage
    venue_stats = pd.merge(venue_stats, first_innings_matches, on='venue')
    venue_stats = pd.merge(venue_stats, first_innings_wins, on='venue', how='left')
    venue_stats['first_innings_win_percentage'] = (venue_stats['first_innings_wins'] / venue_stats['first_innings_matches']) * 100
    venue_stats['first_innings_win_percentage'] = venue_stats['first_innings_win_percentage'].fillna(0)
    
    return venue_stats

def get_phase_analysis_data(df):
    """Get phase analysis data"""
    phase_stats = df.groupby(['batting_team', 'phase']).agg(
        runs=('runs_total', 'sum'),
        balls=('valid_ball', 'sum'),
        wickets=('wicket_kind', lambda x: (x != 'None').sum()),
        dots=('runs_total', lambda x: (x == 0).sum()),
        boundaries=('runs_batter', lambda x: ((x == 4) | (x == 6)).sum())
    ).reset_index()
    
    # Calculate metrics
    phase_stats['run_rate'] = (phase_stats['runs'] / phase_stats['balls']) * 6
    phase_stats['wicket_rate'] = (phase_stats['wickets'] / phase_stats['balls']) * 6
    phase_stats['dot_percentage'] = (phase_stats['dots'] / phase_stats['balls']) * 100
    phase_stats['boundary_percentage'] = (phase_stats['boundaries'] / phase_stats['balls']) * 100
    
    return phase_stats

# Load and preprocess data
df = load_and_preprocess_data()
batting_df, bowling_df, team_match_df = create_derived_features(df)

# Get unique values for filters
seasons = sorted(df['season'].astype(str).unique())
teams = sorted(df['batting_team'].astype(str).unique())
venues = sorted(df['venue'].astype(str).unique())
players_batting = sorted(batting_df['batter'].astype(str).unique())
players_bowling = sorted(bowling_df['bowler'].astype(str).unique())

# Streamlit app
st.set_page_config(page_title="IPL Data Analysis Platform", layout="wide")

st.title("IPL Data Analysis Platform")

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Team Performance", "Player Performance", "Match Analysis", "Historical Trends", "Advanced Analytics"])

# Team Performance Tab
with tab1:
    st.header("Team Performance Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        selected_season = st.selectbox("Select Season", seasons, index=len(seasons)-1)
    with col2:
        selected_team = st.selectbox("Select Team", teams)
    
    # Get team performance data
    team_perf = get_team_performance_data(df, selected_season)
    team_data = team_perf[team_perf['batting_team'] == selected_team]
    
    # Team performance overview
    st.subheader("Team Performance Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Win Percentage", f"{team_data['win_percentage'].values[0]:.1f}%")
    
    with col2:
        st.metric("Average Runs", f"{team_data['avg_runs'].values[0]:.1f}")
    
    with col3:
        st.metric("Average Wickets", f"{team_data['avg_wickets'].values[0]:.1f}")
    
    with col4:
        st.metric("Run Rate", f"{team_data['avg_run_rate'].values[0]:.1f}")
    
    # Win percentage by season
    st.subheader("Win Percentage by Season")
    team_perf_all = get_team_performance_data(df)
    team_data_all = team_perf_all[team_perf_all['batting_team'] == selected_team]
    
    fig_win_pct = px.line(
        team_data_all,
        x='season',
        y='win_percentage',
        title=f'{selected_team} Win Percentage by Season',
        labels={'win_percentage': 'Win %', 'season': 'Season'},
        markers=True
    )
    st.plotly_chart(fig_win_pct, use_container_width=True)
    
    # Phase-wise performance
    st.subheader("Phase-wise Performance")
    phase_data = get_phase_analysis_data(df)
    if selected_season:
        phase_data = phase_data[phase_data['match_id'].isin(df[df['season'] == selected_season]['match_id'])]
    
    team_phase_data = phase_data[phase_data['batting_team'] == selected_team]
    
    fig_phase = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Run Rate', 'Wicket Rate', 'Dot Ball %', 'Boundary %'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    phases = ['Powerplay', 'Middle', 'Death']
    colors = ['blue', 'green', 'red']
    
    for i, phase in enumerate(phases):
        phase_df = team_phase_data[team_phase_data['phase'] == phase]
        if not phase_df.empty:
            fig_phase.add_trace(
                go.Bar(x=[phase], y=[phase_df['run_rate'].values[0]], name=phase, marker_color=colors[i]),
                row=1, col=1
            )
            fig_phase.add_trace(
                go.Bar(x=[phase], y=[phase_df['wicket_rate'].values[0]], name=phase, marker_color=colors[i]),
                row=1, col=2
            )
            fig_phase.add_trace(
                go.Bar(x=[phase], y=[phase_df['dot_percentage'].values[0]], name=phase, marker_color=colors[i]),
                row=2, col=1
            )
            fig_phase.add_trace(
                go.Bar(x=[phase], y=[phase_df['boundary_percentage'].values[0]], name=phase, marker_color=colors[i]),
                row=2, col=2
            )
    
    fig_phase.update_layout(height=500, showlegend=False)
    st.plotly_chart(fig_phase, use_container_width=True)
    
    # Head-to-Head Record
    st.subheader("Head-to-Head Record")
    if selected_season:
        match_df = df[df['season'] == selected_season]
    else:
        match_df = df.copy()
    
    # Get matches involving the selected team
    team_matches = match_df[(match_df['batting_team'] == selected_team) | (match_df['bowling_team'] == selected_team)]
    
    # Get unique match IDs
    match_ids = team_matches['match_id'].unique()
    
    # Get match results
    match_results = match_df[match_df['match_id'].isin(match_ids)][['match_id', 'match_won_by']].drop_duplicates()
    
    # Get opponent teams
    team_matches_summary = team_matches.groupby('match_id').agg(
        team1=('batting_team', 'first'),
        team2=('bowling_team', 'first')
    ).reset_index()
    
    # Merge with results
    match_summary = pd.merge(team_matches_summary, match_results, on='match_id')
    
    # Determine opponent and result
    match_summary['opponent'] = np.where(
        match_summary['team1'] == selected_team,
        match_summary['team2'],
        match_summary['team1']
    )
    
    match_summary['result'] = np.where(
        match_summary['match_won_by'] == selected_team,
        'Win',
        'Loss'
    )
    
    # Count wins and losses against each opponent
    h2h = match_summary.groupby(['opponent', 'result']).size().unstack().fillna(0)
    h2h = h2h.reset_index()
    
    # Create bar chart
    fig_h2h = go.Figure()
    
    fig_h2h.add_trace(go.Bar(
        x=h2h['opponent'],
        y=h2h['Win'],
        name='Wins',
        marker_color='green'
    ))
    
    fig_h2h.add_trace(go.Bar(
        x=h2h['opponent'],
        y=h2h['Loss'],
        name='Losses',
        marker_color='red'
    ))
    
    fig_h2h.update_layout(
        title=f'{selected_team} Head-to-Head Record',
        xaxis_title='Opponent',
        yaxis_title='Matches',
        barmode='stack'
    )
    
    st.plotly_chart(fig_h2h, use_container_width=True)

# Player Performance Tab
with tab2:
    st.header("Player Performance Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        player_type = st.radio("Player Type", ["Batter", "Bowler"])
    with col2:
        selected_season_player = st.selectbox("Select Season", seasons, index=len(seasons)-1)
    
    if player_type == "Batter":
        selected_player = st.selectbox("Select Player", players_batting)
    else:
        selected_player = st.selectbox("Select Player", players_bowling)
    
    # Player performance overview
    st.subheader("Player Performance Overview")
    
    if player_type == "Batter":
        player_data = get_player_performance_data(df, 'batter', selected_season_player)
        player_stats = player_data[player_data['batter'] == selected_player]
        
        if not player_stats.empty:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Runs", f"{player_stats['runs'].values[0]}")
            
            with col2:
                st.metric("Average", f"{player_stats['average'].values[0]:.1f}")
            
            with col3:
                st.metric("Strike Rate", f"{player_stats['strike_rate'].values[0]:.1f}")
            
            with col4:
                boundary_pct = ((player_stats['fours'].values[0] + player_stats['sixes'].values[0]) / player_stats['balls'].values[0]) * 100
                st.metric("Boundary %", f"{boundary_pct:.1f}%")
        
        # Season-wise performance
        st.subheader("Season-wise Performance")
        player_season_data = batting_df[batting_df['batter'] == selected_player]
        season_stats = player_season_data.groupby('season').agg(
            runs=('runs_batter', 'sum'),
            matches=('match_id', 'nunique'),
            average=('runs_batter', lambda x: x.sum() / player_season_data[player_season_data['season'] == x.name]['dismissals'].sum() if player_season_data[player_season_data['season'] == x.name]['dismissals'].sum() > 0 else x.sum()),
            strike_rate=('runs_batter', lambda x: (x.sum() / player_season_data[player_season_data['season'] == x.name]['balls_faced'].sum()) * 100 if player_season_data[player_season_data['season'] == x.name]['balls_faced'].sum() > 0 else 0)
        ).reset_index()
        
        fig_season = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            subplot_titles=('Runs by Season', 'Strike Rate by Season'),
            vertical_spacing=0.1
        )
        
        fig_season.add_trace(
            go.Bar(x=season_stats['season'], y=season_stats['runs'], name='Runs'),
            row=1, col=1
        )
        
        fig_season.add_trace(
            go.Scatter(x=season_stats['season'], y=season_stats['strike_rate'], name='Strike Rate', mode='lines+markers'),
            row=2, col=1
        )
        
        fig_season.update_layout(height=500)
        st.plotly_chart(fig_season, use_container_width=True)
        
        # Performance against teams
        st.subheader("Performance Against Teams")
        player_vs_team = batting_df[batting_df['batter'] == selected_player]
        if selected_season_player:
            player_vs_team = player_vs_team[player_vs_team['match_id'].isin(df[df['season'] == selected_season_player]['match_id'])]
        
        # Get opponent team (bowling team)
        player_vs_team = player_vs_team.merge(
            df[['match_id', 'bowling_team']].drop_duplicates(),
            on='match_id'
        )
        
        vs_team_stats = player_vs_team.groupby('bowling_team').agg(
            runs=('runs_batter', 'sum'),
            balls=('balls_faced', 'sum'),
            dismissals=('dismissals', 'sum')
        ).reset_index()
        
        vs_team_stats['average'] = np.where(
            vs_team_stats['dismissals'] > 0,
            vs_team_stats['runs'] / vs_team_stats['dismissals'],
            vs_team_stats['runs']
        )
        
        vs_team_stats['strike_rate'] = np.where(
            vs_team_stats['balls'] > 0,
            (vs_team_stats['runs'] / vs_team_stats['balls']) * 100,
            0
        )
        
        fig_vs_team = px.bar(
            vs_team_stats,
            x='bowling_team',
            y='runs',
            title=f'{selected_player} - Runs Against Teams',
            labels={'runs': 'Runs', 'bowling_team': 'Opponent Team'},
            hover_data=['average', 'strike_rate']
        )
        
        st.plotly_chart(fig_vs_team, use_container_width=True)
        
        # Performance by phase
        st.subheader("Performance by Phase")
        player_phase = df[df['batter'] == selected_player]
        if selected_season_player:
            player_phase = player_phase[player_phase['season'] == selected_season_player]
        
        phase_stats = player_phase.groupby('phase').agg(
            runs=('runs_batter', 'sum'),
            balls=('valid_ball', 'sum'),
            dismissals=('player_out', lambda x: (x != 'None').sum())
        ).reset_index()
        
        phase_stats['strike_rate'] = np.where(
            phase_stats['balls'] > 0,
            (phase_stats['runs'] / phase_stats['balls']) * 100,
            0
        )
        
        phase_stats['average'] = np.where(
            phase_stats['dismissals'] > 0,
            phase_stats['runs'] / phase_stats['dismissals'],
            phase_stats['runs']
        )
        
        fig_phase = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            subplot_titles=('Strike Rate by Phase', 'Average by Phase'),
            vertical_spacing=0.1
        )
        
        fig_phase.add_trace(
            go.Bar(x=phase_stats['phase'], y=phase_stats['strike_rate'], name='Strike Rate'),
            row=1, col=1
        )
        
        fig_phase.add_trace(
            go.Bar(x=phase_stats['phase'], y=phase_stats['average'], name='Average'),
            row=2, col=1
        )
        
        fig_phase.update_layout(height=500)
        st.plotly_chart(fig_phase, use_container_width=True)
    
    else:  # Bowler
        player_data = get_player_performance_data(df, 'bowler', selected_season_player)
        player_stats = player_data[player_data['bowler'] == selected_player]
        
        if not player_stats.empty:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Wickets", f"{player_stats['wickets'].values[0]}")
            
            with col2:
                st.metric("Economy Rate", f"{player_stats['economy_rate'].values[0]:.1f}")
            
            with col3:
                st.metric("Bowling Average", f"{player_stats['bowling_average'].values[0]:.1f}")
            
            with col4:
                st.metric("Dot Ball %", f"{player_stats['dot_percentage'].values[0]:.1f}%")
        
        # Season-wise performance
        st.subheader("Season-wise Performance")
        player_season_data = bowling_df[bowling_df['bowler'] == selected_player]
        season_stats = player_season_data.groupby('season').agg(
            wickets=('wickets', 'sum'),
            matches=('match_id', 'nunique'),
            economy_rate=('runs_conceded', lambda x: (x.sum() / player_season_data[player_season_data['season'] == x.name]['balls_bowled'].sum()) * 6 if player_season_data[player_season_data['season'] == x.name]['balls_bowled'].sum() > 0 else 0),
            bowling_average=('runs_conceded', lambda x: x.sum() / player_season_data[player_season_data['season'] == x.name]['wickets'].sum() if player_season_data[player_season_data['season'] == x.name]['wickets'].sum() > 0 else x.sum())
        ).reset_index()
        
        fig_season = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            subplot_titles=('Wickets by Season', 'Economy Rate by Season'),
            vertical_spacing=0.1
        )
        
        fig_season.add_trace(
            go.Bar(x=season_stats['season'], y=season_stats['wickets'], name='Wickets'),
            row=1, col=1
        )
        
        fig_season.add_trace(
            go.Scatter(x=season_stats['season'], y=season_stats['economy_rate'], name='Economy Rate', mode='lines+markers'),
            row=2, col=1
        )
        
        fig_season.update_layout(height=500)
        st.plotly_chart(fig_season, use_container_width=True)
        
        # Performance against teams
        st.subheader("Performance Against Teams")
        player_vs_team = bowling_df[bowling_df['bowler'] == selected_player]
        if selected_season_player:
            player_vs_team = player_vs_team[player_vs_team['match_id'].isin(df[df['season'] == selected_season_player]['match_id'])]
        
        # Get opponent team (batting team)
        player_vs_team = player_vs_team.merge(
            df[['match_id', 'batting_team']].drop_duplicates(),
            on='match_id'
        )
        
        vs_team_stats = player_vs_team.groupby('batting_team').agg(
            wickets=('wickets', 'sum'),
            runs_conceded=('runs_conceded', 'sum'),
            balls_bowled=('balls_bowled', 'sum')
        ).reset_index()
        
        vs_team_stats['economy_rate'] = np.where(
            vs_team_stats['balls_bowled'] > 0,
            (vs_team_stats['runs_conceded'] / vs_team_stats['balls_bowled']) * 6,
            0
        )
        
        vs_team_stats['bowling_average'] = np.where(
            vs_team_stats['wickets'] > 0,
            vs_team_stats['runs_conceded'] / vs_team_stats['wickets'],
            vs_team_stats['runs_conceded']
        )
        
        fig_vs_team = px.bar(
            vs_team_stats,
            x='batting_team',
            y='wickets',
            title=f'{selected_player} - Wickets Against Teams',
            labels={'wickets': 'Wickets', 'batting_team': 'Opponent Team'},
            hover_data=['economy_rate', 'bowling_average']
        )
        
        st.plotly_chart(fig_vs_team, use_container_width=True)
        
        # Performance by phase
        st.subheader("Performance by Phase")
        player_phase = df[df['bowler'] == selected_player]
        if selected_season_player:
            player_phase = player_phase[player_phase['season'] == selected_season_player]
        
        phase_stats = player_phase.groupby('phase').agg(
            wickets=('wicket_kind', lambda x: (x != 'None').sum()),
            runs_conceded=('runs_bowler', 'sum'),
            balls_bowled=('valid_ball', 'sum')
        ).reset_index()
        
        phase_stats['economy_rate'] = np.where(
            phase_stats['balls_bowled'] > 0,
            (phase_stats['runs_conceded'] / phase_stats['balls_bowled']) * 6,
            0
        )
        
        phase_stats['bowling_average'] = np.where(
            phase_stats['wickets'] > 0,
            phase_stats['runs_conceded'] / phase_stats['wickets'],
            phase_stats['runs_conceded']
        )
        
        fig_phase = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            subplot_titles=('Economy Rate by Phase', 'Bowling Average by Phase'),
            vertical_spacing=0.1
        )
        
        fig_phase.add_trace(
            go.Bar(x=phase_stats['phase'], y=phase_stats['economy_rate'], name='Economy Rate'),
            row=1, col=1
        )
        
        fig_phase.add_trace(
            go.Bar(x=phase_stats['phase'], y=phase_stats['bowling_average'], name='Bowling Average'),
            row=2, col=1
        )
        
        fig_phase.update_layout(height=500)
        st.plotly_chart(fig_phase, use_container_width=True)

# Match Analysis Tab
with tab3:
    st.header("Match Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        selected_season_match = st.selectbox("Select Season", seasons, index=len(seasons)-1)
    with col2:
        selected_venue = st.selectbox("Select Venue", venues)
    
    # Venue Analysis
    st.subheader("Venue Analysis")
    venue_data = get_venue_analysis_data(df)
    if selected_season_match:
        venue_data = venue_data[venue_data['match_id'].isin(df[df['season'] == selected_season_match]['match_id'])]
    
    venue_stats = venue_data[venue_data['venue'] == selected_venue]
    
    if not venue_stats.empty:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Matches", f"{venue_stats['matches'].values[0]}")
        
        with col2:
            st.metric("Avg 1st Innings Score", f"{venue_stats['avg_first_innings_score'].values[0]:.1f}")
        
        with col3:
            st.metric("Avg 2nd Innings Score", f"{venue_stats['avg_second_innings_score'].values[0]:.1f}")
        
        with col4:
            st.metric("1st Innings Win %", f"{venue_stats['first_innings_win_percentage'].values[0]:.1f}%")
    
    # Toss Impact
    st.subheader("Toss Impact")
    match_df = df.copy()
    if selected_season_match:
        match_df = match_df[match_df['season'] == selected_season_match]
    if selected_venue:
        match_df = match_df[match_df['venue'] == selected_venue]
    
    # Get unique matches
    matches = match_df[['match_id', 'toss_winner', 'toss_decision', 'match_won_by']].drop_duplicates()
    
    # Determine if toss winner won the match
    matches['toss_winner_won'] = matches['toss_winner'] == matches['match_won_by']
    
    # Group by toss decision
    toss_impact = matches.groupby('toss_decision')['toss_winner_won'].agg(['count', 'sum']).reset_index()
    toss_impact['win_percentage'] = (toss_impact['sum'] / toss_impact['count']) * 100
    
    fig_toss = px.bar(
        toss_impact,
        x='toss_decision',
        y='win_percentage',
        title='Toss Impact on Match Outcome',
        labels={'win_percentage': 'Win %', 'toss_decision': 'Toss Decision'},
        text=toss_impact['win_percentage'].apply(lambda x: f'{x:.1f}%')
    )
    
    fig_toss.update_traces(texttemplate='%{text}', textposition='outside')
    fig_toss.update_layout(yaxis_title="Win Percentage")
    
    st.plotly_chart(fig_toss, use_container_width=True)
    
    # Match Outcome Distribution
    st.subheader("Match Outcome Distribution")
    match_outcomes = match_df[['match_id', 'win_outcome']].drop_duplicates()
    
    # Count outcomes
    outcomes = match_outcomes['win_outcome'].value_counts().reset_index()
    outcomes.columns = ['win_outcome', 'count']
    
    fig_outcome = px.pie(
        outcomes,
        values='count',
        names='win_outcome',
        title='Match Outcome Distribution',
        hole=0.3
    )
    
    st.plotly_chart(fig_outcome, use_container_width=True)
    
    # Score Distribution
    st.subheader("Score Distribution")
    first_innings = match_df[match_df['innings'] == 1].groupby('match_id')['team_runs'].max().reset_index()
    
    fig_score = px.histogram(
        first_innings,
        x='team_runs',
        title='First Innings Score Distribution',
        nbins=20,
        labels={'team_runs': 'Score', 'count': 'Frequency'}
    )
    
    fig_score.update_layout(bargap=0.1)
    st.plotly_chart(fig_score, use_container_width=True)

# Historical Trends Tab
with tab4:
    st.header("Historical Trends")
    
    col1, col2 = st.columns(2)
    with col1:
        metric = st.selectbox("Select Metric", [
            'Average Runs per Match',
            'Average Wickets per Match',
            'Run Rate',
            'Boundary Percentage'
        ])
        
        metric_map = {
            'Average Runs per Match': 'avg_runs',
            'Average Wickets per Match': 'avg_wickets',
            'Run Rate': 'run_rate',
            'Boundary Percentage': 'boundary_percentage'
        }
        
        selected_metric = metric_map[metric]
    
    with col2:
        selected_team_hist = st.selectbox("Select Team (Optional)", ["All Teams"] + teams)
    
    # Historical trends
    st.subheader("Historical Trends")
    team_perf = get_team_performance_data(df)
    
    if selected_team_hist != "All Teams":
        team_perf = team_perf[team_perf['batting_team'] == selected_team_hist]
    
    fig_trends = px.line(
        team_perf,
        x='season',
        y=selected_metric,
        title=f'{metric} Over Time',
        labels={selected_metric: metric, 'season': 'Season'},
        color='batting_team' if selected_team_hist == "All Teams" else None,
        markers=True
    )
    
    fig_trends.update_layout(
        yaxis_title=metric,
        xaxis_title="Season",
        hovermode='x unified'
    )
    
    st.plotly_chart(fig_trends, use_container_width=True)
    
    # Season Comparison
    st.subheader("Season Comparison")
    team_perf_comp = get_team_performance_data(df)
    
    # Get top 5 teams by the selected metric in the latest season
    latest_season = team_perf_comp['season'].max()
    top_teams = team_perf_comp[team_perf_comp['season'] == latest_season].nlargest(5, selected_metric)['batting_team'].tolist()
    
    # Filter data for top teams
    top_team_data = team_perf_comp[team_perf_comp['batting_team'].isin(top_teams)]
    
    fig_comp = px.bar(
        top_team_data,
        x='season',
        y=selected_metric,
        color='batting_team',
        title=f'Top 5 Teams by {metric}',
        labels={selected_metric: metric, 'season': 'Season'},
        barmode='group'
    )
    
    st.plotly_chart(fig_comp, use_container_width=True)
    
    # Team Evolution
    if selected_team_hist != "All Teams":
        st.subheader("Team Evolution")
        team_perf_evo = get_team_performance_data(df)
        team_data_evo = team_perf_evo[team_perf_evo['batting_team'] == selected_team_hist]
        
        fig_evo = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Win Percentage', 'Average Runs', 'Average Wickets', 'Run Rate'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        fig_evo.add_trace(
            go.Scatter(x=team_data_evo['season'], y=team_data_evo['win_percentage'], name='Win %', mode='lines+markers'),
            row=1, col=1
        )
        
        fig_evo.add_trace(
            go.Scatter(x=team_data_evo['season'], y=team_data_evo['avg_runs'], name='Avg Runs', mode='lines+markers'),
            row=1, col=2
        )
        
        fig_evo.add_trace(
            go.Scatter(x=team_data_evo['season'], y=team_data_evo['avg_wickets'], name='Avg Wickets', mode='lines+markers'),
            row=2, col=1
        )
        
        fig_evo.add_trace(
            go.Scatter(x=team_data_evo['season'], y=team_data_evo['avg_run_rate'], name='Run Rate', mode='lines+markers'),
            row=2, col=2
        )
        
        fig_evo.update_layout(height=600, title_text=f"{selected_team_hist} Team Evolution")
        st.plotly_chart(fig_evo, use_container_width=True)

# Advanced Analytics Tab
with tab5:
    st.header("Advanced Analytics")
    
    col1, col2 = st.columns(2)
    with col1:
        analysis_type = st.selectbox("Select Analysis Type", [
            'Partnership Analysis',
            'Pressure Performance',
            'Momentum Shifts',
            'Player Impact'
        ])
    with col2:
        selected_season_adv = st.selectbox("Select Season", seasons, index=len(seasons)-1)
    
    match_df = df.copy()
    if selected_season_adv:
        match_df = match_df[match_df['season'] == selected_season_adv]
    
    # Advanced Analysis
    st.subheader("Advanced Analysis")
    
    if analysis_type == 'Partnership Analysis':
        # Get partnership data
        partnership_data = match_df.groupby(['match_id', 'innings', 'partnership', 'batting_team']).agg(
            runs=('runs_total', 'sum'),
            balls=('valid_ball', 'sum'),
            wickets=('wicket_kind', lambda x: (x != 'None').sum())
        ).reset_index()
        
        # Filter partnerships with at least 20 runs
        partnership_data = partnership_data[partnership_data['runs'] >= 20]
        
        # Get top partnerships
        top_partnerships = partnership_data.nlargest(20, 'runs')
        
        # Create partnership identifier
        top_partnerships['partnership_id'] = top_partnerships.apply(
            lambda x: f"{x['match_id']} - {x['batting_team']} - Inn {x['innings']}", axis=1
        )
        
        fig_adv = px.bar(
            top_partnerships,
            x='partnership_id',
            y='runs',
            title='Top Partnerships',
            labels={'runs': 'Runs', 'partnership_id': 'Partnership'},
            hover_data=['balls', 'wickets']
        )
        
        fig_adv.update_layout(xaxis_title="Partnership", yaxis_title="Runs")
        st.plotly_chart(fig_adv, use_container_width=True)
        
        # Key insights
        st.subheader("Key Insights")
        top_partnership = partnership_data.nlargest(1, 'runs')
        
        st.write(f"Highest partnership: {top_partnership['runs'].values[0]} runs in {top_partnership['balls'].values[0]} balls")
        st.write(f"Team: {top_partnership['batting_team'].values[0]}")
        st.write(f"Match ID: {top_partnership['match_id'].values[0]}, Innings: {top_partnership['innings'].values[0]}")
    
    elif analysis_type == 'Pressure Performance':
        # Define pressure situations (last 4 overs of chase with close game)
        chase_matches = match_df[match_df['innings'] == 2].copy()
        chase_matches['target'] = chase_matches.groupby('match_id')['runs_target'].transform('first')
        chase_matches['remaining_runs'] = chase_matches['target'] - chase_matches.groupby('match_id')['team_runs'].transform('cumsum')
        chase_matches['remaining_balls'] = chase_matches.groupby('match_id').cumcount(ascending=False)
        
        # Pressure situations: last 4 overs with less than 40 runs to win
        pressure_situations = chase_matches[
            (chase_matches['over'] >= 16) & 
            (chase_matches['remaining_runs'] < 40) & 
            (chase_matches['remaining_runs'] > 0)
        ]
        
        # Get batter performance in pressure
        pressure_batters = pressure_situations.groupby('batter').agg(
            runs=('runs_batter', 'sum'),
            balls=('valid_ball', 'sum'),
            dismissals=('player_out', lambda x: (x != 'None').sum())
        ).reset_index()
        
        # Calculate strike rate
        pressure_batters['strike_rate'] = np.where(
            pressure_batters['balls'] > 0,
            (pressure_batters['runs'] / pressure_batters['balls']) * 100,
            0
        )
        
        # Filter batters with at least 20 balls faced
        pressure_batters = pressure_batters[pressure_batters['balls'] >= 20]
        
        # Get top performers
        top_pressure_batters = pressure_batters.nlargest(10, 'strike_rate')
        
        fig_adv = px.scatter(
            top_pressure_batters,
            x='runs',
            y='strike_rate',
            size='balls',
            color='dismissals',
            title='Batter Performance in Pressure Situations',
            labels={'runs': 'Runs', 'strike_rate': 'Strike Rate', 'dismissals': 'Dismissals'},
            hover_name='batter'
        )
        
        st.plotly_chart(fig_adv, use_container_width=True)
        
        # Key insights
        st.subheader("Key Insights")
        total_pressure_balls = pressure_situations['valid_ball'].sum()
        total_pressure_runs = pressure_situations['runs_batter'].sum()
        pressure_run_rate = (total_pressure_runs / total_pressure_balls) * 6 if total_pressure_balls > 0 else 0
        
        st.write(f"Total pressure situations balls: {total_pressure_balls}")
        st.write(f"Total runs scored in pressure: {total_pressure_runs}")
        st.write(f"Pressure run rate: {pressure_run_rate:.2f} runs per over")
    
    elif analysis_type == 'Momentum Shifts':
        # Calculate momentum shifts (wickets in quick succession)
        match_df['over_ball'] = match_df['over'] + (match_df['ball'] / 10)
        
        # Get wicket events
        wickets = match_df[match_df['wicket_kind'] != 'None'].copy()
        
        # Group by match and innings
        wickets['wicket_number'] = wickets.groupby(['match_id', 'innings']).cumcount() + 1
        
        # Find quick wickets (within 12 balls)
        quick_wickets = wickets.copy()
        quick_wickets['prev_wicket_time'] = quick_wickets.groupby(['match_id', 'innings'])['over_ball'].shift(1)
        quick_wickets['balls_since_prev'] = (quick_wickets['over_ball'] - quick_wickets['prev_wicket_time']) * 10
        
        # Filter wickets within 12 balls of previous wicket
        quick_wickets = quick_wickets[
            (quick_wickets['wicket_number'] > 1) & 
            (quick_wickets['balls_since_prev'] <= 12)
        ]
        
        # Count quick wicket clusters by team
        momentum_shifts = quick_wickets.groupby('bowling_team').size().reset_index(name='quick_wicket_clusters')
        
        fig_adv = px.bar(
            momentum_shifts,
            x='bowling_team',
            y='quick_wicket_clusters',
            title='Momentum Shifts (Quick Wicket Clusters)',
            labels={'quick_wicket_clusters': 'Quick Wicket Clusters', 'bowling_team': 'Team'}
        )
        
        st.plotly_chart(fig_adv, use_container_width=True)
        
        # Key insights
        st.subheader("Key Insights")
        total_quick_wicket_clusters = len(quick_wickets.groupby(['match_id', 'innings']))
        total_matches = match_df['match_id'].nunique()
        
        st.write(f"Total quick wicket clusters: {total_quick_wicket_clusters}")
        st.write(f"Percentage of matches with momentum shifts: {(total_quick_wicket_clusters / total_matches) * 100:.1f}%")
    
    else:  # Player Impact
        # Calculate player impact score (simplified version)
        # Impact = (runs + wickets*25) / matches
        
        # Get batter impact
        batters = get_player_performance_data(match_df, 'batter')
        batters['impact'] = (batters['runs'] + batters['dismissals']*10) / batters['matches']
        
        # Get bowler impact
        bowlers = get_player_performance_data(match_df, 'bowler')
        bowlers['impact'] = (bowlers['wickets']*25 + bowlers['dots']*0.5) / bowlers['matches']
        
        # Combine and get top impact players
        batters['role'] = 'Batter'
        bowlers['role'] = 'Bowler'
        
        batters = batters.rename(columns={'batter': 'player'})
        bowlers = bowlers.rename(columns={'bowler': 'player'})
        
        impact_players = pd.concat([
            batters[['player', 'impact', 'role']],
            bowlers[['player', 'impact', 'role']]
        ])
        
        # Get top 20 impact players
        top_impact = impact_players.nlargest(20, 'impact')
        
        fig_adv = px.bar(
            top_impact,
            x='player',
            y='impact',
            color='role',
            title='Player Impact Scores',
            labels={'impact': 'Impact Score', 'player': 'Player'}
        )
        
        st.plotly_chart(fig_adv, use_container_width=True)
        
        # Key insights
        st.subheader("Key Insights")
        top_batter = batters.nlargest(1, 'impact')
        top_bowler = bowlers.nlargest(1, 'impact')
        
        st.write(f"Highest impact batter: {top_batter['player'].values[0]} (Impact: {top_batter['impact'].values[0]:.2f})")
        st.write(f"Highest impact bowler: {top_bowler['player'].values[0]} (Impact: {top_bowler['impact'].values[0]:.2f})")
