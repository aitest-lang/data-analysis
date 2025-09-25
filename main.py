import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')
import zipfile

with zipfile.ZipFile('archive.zip', 'r') as zip_ref:
    zip_ref.extractall()
# Page configuration
st.set_page_config(
    page_title="IPL Analytics Platform",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Team aliases for data normalization
TEAM_ALIASES = {
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
    "Rising Pune Supergiant": "Rising Pune Supergiant",
    "Kings XI Punjab": "Punjab Kings",
    "Delhi Daredevils": "Delhi Capitals",
    "Pune Warriors": "Pune Warriors India",
}

@st.cache_data
def load_and_process_data():
    """Load and preprocess IPL data"""
    try:
        df = pd.read_csv("IPL.csv")
        
        # Normalize team names
        for col in ['batting_team', 'bowling_team', 'toss_winner', 'match_won_by']:
            if col in df.columns:
                df[col] = df[col].replace(TEAM_ALIASES)
        
        # Convert date column
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Create derived metrics
        df['strike_rate'] = (df['runs_batter'] / df['balls_faced'] * 100).round(2)
        df['economy_rate'] = (df['runs_bowler'] / (df['valid_ball'] / 6)).round(2)
        
        return df
    except FileNotFoundError:
        st.error("IPL.csv file not found. Please ensure the file is in the correct directory.")
        return None

@st.cache_data
def get_team_stats(df):
    """Calculate comprehensive team statistics"""
    teams = df['batting_team'].unique()
    team_stats = []
    
    for team in teams:
        team_data = df[df['batting_team'] == team]
        bowling_data = df[df['bowling_team'] == team]
        
        # Batting stats
        total_runs = team_data['runs_total'].sum()
        total_balls = team_data['valid_ball'].sum()
        total_matches = team_data['match_id'].nunique()
        
        # Bowling stats
        runs_conceded = bowling_data['runs_total'].sum()
        balls_bowled = bowling_data['valid_ball'].sum()
        wickets_taken = bowling_data['wicket_kind'].count()
        
        team_stats.append({
            'Team': team,
            'Matches': total_matches,
            'Runs Scored': total_runs,
            'Balls Faced': total_balls,
            'Strike Rate': round((total_runs / total_balls * 100), 2) if total_balls > 0 else 0,
            'Runs Conceded': runs_conceded,
            'Balls Bowled': balls_bowled,
            'Economy Rate': round((runs_conceded / balls_bowled * 6), 2) if balls_bowled > 0 else 0,
            'Wickets Taken': wickets_taken
        })
    
    return pd.DataFrame(team_stats)

def main():
    """Main application function"""
    st.title("🏏 IPL Analytics Platform")
    st.markdown("*Comprehensive analysis of IPL ball-by-ball data*")
    
    # Load data
    df = load_and_process_data()
    if df is None:
        return
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose Analysis",
        ["🏠 Overview", "🏆 Team Analytics", "👤 Player Analytics", 
         "🎯 Match Analytics", "📊 Insights & Trends"]
    )
    
    # Global filters
    st.sidebar.markdown("### Filters")
    
    seasons = sorted(df['season'].unique())
    selected_seasons = st.sidebar.multiselect(
        "Select Seasons", seasons, default=seasons[-3:] if len(seasons) >= 3 else seasons
    )
    
    teams = sorted(df['batting_team'].unique())
    selected_teams = st.sidebar.multiselect(
        "Select Teams", teams, default=teams
    )
    
    # Filter data
    filtered_df = df[
        (df['season'].isin(selected_seasons)) & 
        (df['batting_team'].isin(selected_teams))
    ]
    
    # Route to appropriate page
    if page == "🏠 Overview":
        show_overview(filtered_df)
    elif page == "🏆 Team Analytics":
        show_team_analytics(filtered_df)
    elif page == "👤 Player Analytics":
        show_player_analytics(filtered_df)
    elif page == "🎯 Match Analytics":
        show_match_analytics(filtered_df)
    elif page == "📊 Insights & Trends":
        show_insights_trends(filtered_df)

def show_overview(df):
    """Display overview dashboard"""
    st.header("📈 IPL Overview Dashboard")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_matches = df['match_id'].nunique()
        st.metric("Total Matches", total_matches)
    
    with col2:
        total_runs = df['runs_total'].sum()
        st.metric("Total Runs", f"{total_runs:,}")
    
    with col3:
        total_sixes = len(df[df['runs_batter'] == 6])
        st.metric("Total Sixes", total_sixes)
    
    with col4:
        total_wickets = df['wicket_kind'].count()
        st.metric("Total Wickets", total_wickets)
    
    # Team performance overview
    st.subheader("Team Performance Summary")
    team_stats = get_team_stats(df)
    st.dataframe(team_stats, use_container_width=True)
    
    # Runs distribution by season
    st.subheader("Runs Scored by Season")
    season_runs = df.groupby('season')['runs_total'].sum().reset_index()
    fig = px.bar(season_runs, x='season', y='runs_total', 
                title="Total Runs by Season")
    st.plotly_chart(fig, use_container_width=True)

def show_team_analytics(df):
    """Team analytics dashboard"""
    st.header("🏆 Team Analytics")
    
    # Team selection
    teams = sorted(df['batting_team'].unique())
    col1, col2 = st.columns(2)
    
    with col1:
        team1 = st.selectbox("Select Team 1", teams, index=0)
    with col2:
        team2 = st.selectbox("Select Team 2", teams, index=1 if len(teams) > 1 else 0)
    
    # Head-to-head analysis
    st.subheader(f"Head-to-Head: {team1} vs {team2}")
    
    h2h_matches = df[
        ((df['batting_team'] == team1) & (df['bowling_team'] == team2)) |
        ((df['batting_team'] == team2) & (df['bowling_team'] == team1))
    ]
    
    if not h2h_matches.empty:
        # Match outcomes
        outcomes = h2h_matches.groupby('match_won_by').size().reset_index(name='matches')
        
        fig = px.pie(outcomes, values='matches', names='match_won_by',
                    title=f"Head-to-Head Record")
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance metrics comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"{team1} Performance")
            team1_batting = df[df['batting_team'] == team1]
            team1_bowling = df[df['bowling_team'] == team1]
            
            if not team1_batting.empty:
                avg_score = team1_batting.groupby('match_id')['runs_total'].sum().mean()
                st.metric("Average Score", f"{avg_score:.1f}")
                
                strike_rate = team1_batting['runs_batter'].sum() / team1_batting['balls_faced'].sum() * 100
                st.metric("Team Strike Rate", f"{strike_rate:.2f}")
        
        with col2:
            st.subheader(f"{team2} Performance")
            team2_batting = df[df['batting_team'] == team2]
            team2_bowling = df[df['bowling_team'] == team2]
            
            if not team2_batting.empty:
                avg_score = team2_batting.groupby('match_id')['runs_total'].sum().mean()
                st.metric("Average Score", f"{avg_score:.1f}")
                
                strike_rate = team2_batting['runs_batter'].sum() / team2_batting['balls_faced'].sum() * 100
                st.metric("Team Strike Rate", f"{strike_rate:.2f}")
    
    # Powerplay analysis
    st.subheader("Powerplay Performance")
    powerplay_data = df[df['over'] <= 6]
    
    team_pp_stats = []
    for team in [team1, team2]:
        team_pp = powerplay_data[powerplay_data['batting_team'] == team]
        if not team_pp.empty:
            pp_runs = team_pp['runs_total'].sum()
            pp_balls = team_pp['valid_ball'].sum()
            pp_wickets = team_pp['wicket_kind'].count()
            
            team_pp_stats.append({
                'Team': team,
                'PP Runs': pp_runs,
                'PP Strike Rate': round(pp_runs/pp_balls*100, 2) if pp_balls > 0 else 0,
                'PP Wickets': pp_wickets
            })
    
    pp_df = pd.DataFrame(team_pp_stats)
    st.dataframe(pp_df, use_container_width=True)

def show_player_analytics(df):
    """Player analytics dashboard"""
    st.header("👤 Player Analytics")
    
    tab1, tab2 = st.tabs(["🏏 Batsmen", "⚾ Bowlers"])
    
    with tab1:
        st.subheader("Batsman Performance")
        
        # Top batsmen
        batsman_stats = df.groupby('batter').agg({
            'runs_batter': 'sum',
            'balls_faced': 'sum',
            'match_id': 'nunique'
        }).reset_index()
        
        batsman_stats['strike_rate'] = (batsman_stats['runs_batter'] / batsman_stats['balls_faced'] * 100).round(2)
        batsman_stats = batsman_stats[batsman_stats['balls_faced'] >= 100]  # Minimum balls faced
        batsman_stats = batsman_stats.sort_values('runs_batter', ascending=False).head(20)
        
        # Player selection
        selected_batsman = st.selectbox("Select Batsman", batsman_stats['batter'].tolist())
        
        # Player stats
        player_data = batsman_stats[batsman_stats['batter'] == selected_batsman].iloc[0]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Runs", int(player_data['runs_batter']))
        with col2:
            st.metric("Matches", int(player_data['match_id']))
        with col3:
            st.metric("Balls Faced", int(player_data['balls_faced']))
        with col4:
            st.metric("Strike Rate", f"{player_data['strike_rate']:.2f}")
        
        # Performance by season
        player_season = df[df['batter'] == selected_batsman].groupby('season').agg({
            'runs_batter': 'sum',
            'balls_faced': 'sum'
        }).reset_index()
        
        player_season['strike_rate'] = (player_season['runs_batter'] / player_season['balls_faced'] * 100).round(2)
        
        fig = px.bar(player_season, x='season', y='runs_batter',
                    title=f"{selected_batsman} - Runs by Season")
        st.plotly_chart(fig, use_container_width=True)
        
        # Top batsmen table
        st.subheader("Top Batsmen")
        st.dataframe(batsman_stats.head(10), use_container_width=True)
    
    with tab2:
        st.subheader("Bowler Performance")
        
        # Top bowlers
        bowler_stats = df.groupby('bowler').agg({
            'runs_bowler': 'sum',
            'valid_ball': 'sum',
            'wicket_kind': 'count',
            'match_id': 'nunique'
        }).reset_index()
        
        bowler_stats['economy_rate'] = (bowler_stats['runs_bowler'] / bowler_stats['valid_ball'] * 6).round(2)
        bowler_stats = bowler_stats[bowler_stats['valid_ball'] >= 100]  # Minimum balls bowled
        bowler_stats = bowler_stats.sort_values('wicket_kind', ascending=False).head(20)
        
        # Player selection
        selected_bowler = st.selectbox("Select Bowler", bowler_stats['bowler'].tolist())
        
        # Player stats
        bowler_data = bowler_stats[bowler_stats['bowler'] == selected_bowler].iloc[0]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Wickets", int(bowler_data['wicket_kind']))
        with col2:
            st.metric("Matches", int(bowler_data['match_id']))
        with col3:
            st.metric("Balls Bowled", int(bowler_data['valid_ball']))
        with col4:
            st.metric("Economy Rate", f"{bowler_data['economy_rate']:.2f}")
        
        # Top bowlers table
        st.subheader("Top Bowlers")
        st.dataframe(bowler_stats.head(10), use_container_width=True)

def show_match_analytics(df):
    """Match analytics dashboard"""
    st.header("🎯 Match Analytics")
    
    # Toss analysis
    st.subheader("Toss Impact Analysis")
    
    toss_data = df.groupby(['toss_winner', 'match_won_by']).size().reset_index(name='count')
    toss_wins = toss_data[toss_data['toss_winner'] == toss_data['match_won_by']]['count'].sum()
    total_matches = df['match_id'].nunique()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Toss Win %", f"{(toss_wins/total_matches*100):.1f}%")
    with col2:
        st.metric("Total Matches Analyzed", total_matches)
    
    # Toss decision impact
    toss_decision = df.groupby(['toss_decision', 'toss_winner', 'match_won_by']).size().reset_index(name='count')
    
    fig = px.sunburst(toss_decision, path=['toss_decision', 'toss_winner'], 
                     values='count', title="Toss Decision Analysis")
    st.plotly_chart(fig, use_container_width=True)
    
    # Venue analysis
    st.subheader("Venue Performance")
    
    venue_stats = df.groupby('venue').agg({
        'runs_total': 'mean',
        'match_id': 'nunique'
    }).reset_index()
    
    venue_stats = venue_stats[venue_stats['match_id'] >= 5]  # Minimum matches
    venue_stats = venue_stats.sort_values('runs_total', ascending=False)
    
    fig = px.bar(venue_stats.head(15), x='venue', y='runs_total',
                title="Average Runs by Venue")
    fig.update_xaxis(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Over-wise run rate
    st.subheader("Over-wise Scoring Pattern")
    
    over_stats = df.groupby('over')['runs_total'].mean().reset_index()
    
    fig = px.line(over_stats, x='over', y='runs_total',
                 title="Average Runs per Over")
    st.plotly_chart(fig, use_container_width=True)

def show_insights_trends(df):
    """Insights and trends dashboard"""
    st.header("📊 Insights & Trends")
    
    # Seasonal trends
    st.subheader("IPL Evolution Over Years")
    
    season_trends = df.groupby('season').agg({
        'runs_total': ['sum', 'mean'],
        'runs_batter': lambda x: (x == 6).sum(),  # Sixes
        'runs_batter': lambda x: (x == 4).sum(),  # Fours
        'wicket_kind': 'count'
    }).reset_index()
    
    season_trends.columns = ['season', 'total_runs', 'avg_runs', 'sixes', 'fours', 'wickets']
    
    # Multiple metrics plot
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Total Runs by Season', 'Average Runs per Ball', 
                       'Sixes Hit', 'Wickets Taken')
    )
    
    fig.add_trace(go.Scatter(x=season_trends['season'], y=season_trends['total_runs'],
                            name='Total Runs'), row=1, col=1)
    fig.add_trace(go.Scatter(x=season_trends['season'], y=season_trends['avg_runs'],
                            name='Avg Runs'), row=1, col=2)
    fig.add_trace(go.Scatter(x=season_trends['season'], y=season_trends['sixes'],
                            name='Sixes'), row=2, col=1)
    fig.add_trace(go.Scatter(x=season_trends['season'], y=season_trends['wickets'],
                            name='Wickets'), row=2, col=2)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Records section
    st.subheader("🏆 Record Holders")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Highest Individual Scores**")
        top_scores = df.groupby(['match_id', 'batter'])['runs_batter'].sum().reset_index()
        top_scores = top_scores.nlargest(10, 'runs_batter')
        st.dataframe(top_scores, use_container_width=True)
    
    with col2:
        st.write("**Best Bowling Figures**")
        best_bowling = df.groupby(['match_id', 'bowler']).agg({
            'wicket_kind': 'count',
            'runs_bowler': 'sum'
        }).reset_index()
        best_bowling = best_bowling.nlargest(10, 'wicket_kind')
        st.dataframe(best_bowling, use_container_width=True)
    
    # Fun facts
    st.subheader("🎯 Fun Facts")
    
    facts_col1, facts_col2 = st.columns(2)
    
    with facts_col1:
        most_sixes_match = df[df['runs_batter'] == 6].groupby('match_id').size().max()
        st.info(f"🚀 Most sixes in a match: {most_sixes_match}")
        
        highest_partnership = df.groupby(['match_id', 'batting_partners'])['runs_batter'].sum().max()
        st.info(f"🤝 Highest partnership: {highest_partnership} runs")
    
    with facts_col2:
        most_expensive_over = df.groupby(['match_id', 'over'])['runs_total'].sum().max()
        st.info(f"💸 Most expensive over: {most_expensive_over} runs")
        
        most_wickets_match = df.groupby('match_id')['wicket_kind'].count().max()
        st.info(f"🎯 Most wickets in a match: {most_wickets_match}")

if __name__ == "__main__":
    main()
