import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure the page
st.set_page_config(
    page_title="IPL Data Analysis",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("IPL.csv")
        
        # Check available columns
        st.write("Available columns in the dataset:", list(df.columns))
        
        # Check if required columns exist
        required_columns = ['season', 'match_id', 'batting_team', 'bowling_team', 'runs_total', 'batter', 'bowler']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"Missing required columns: {missing_columns}")
            st.stop()
        
        # Fix team name inconsistencies
        df['batting_team'] = df['batting_team'].replace({
            'Royal Challengers Bangalore': 'Royal Challengers Bengaluru',
            'Royal Challengers Bengaluru': 'Royal Challengers Bengaluru'
        })
        df['bowling_team'] = df['bowling_team'].replace({
            'Royal Challengers Bangalore': 'Royal Challengers Bengaluru',
            'Royal Challengers Bengaluru': 'Royal Challengers Bengaluru'
        })
        
        # Fix other team name inconsistencies if any
        team_mapping = {
            'Delhi Daredevils': 'Delhi Capitals',
            'Kings XI Punjab': 'Punjab Kings',
            'Deccan Chargers': 'Sunrisers Hyderabad'  # Note: This is approximate
        }
        
        df['batting_team'] = df['batting_team'].replace(team_mapping)
        df['bowling_team'] = df['bowling_team'].replace(team_mapping)
        
        # Convert date to datetime if column exists
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # If 'season' column doesn't exist, create it from date or use a default
        if 'season' not in df.columns:
            if 'date' in df.columns:
                df['season'] = df['date'].dt.year
            else:
                # If no date column, create a default season column
                df['season'] = 2020  # Default value
        
        # Ensure all required columns exist with default values if missing
        default_columns = {
            'runs_total': 0,
            'runs_batter': 0,
            'runs_bowler': 0,
            'balls_faced': 0,
            'bowler_wicket': 0,
            'valid_ball': 1,
            'win_outcome': '',
            'venue': 'Unknown',
            'innings': 1,
            'toss_winner': '',
            'toss_decision': 'bat',
            'balls': 0
        }
        
        for col, default_val in default_columns.items():
            if col not in df.columns:
                df[col] = default_val
        
        return df
        
    except FileNotFoundError:
        st.error("IPL.csv file not found. Please make sure the file exists in the correct location.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

def main():
    st.title("🏏 IPL Data Analysis Dashboard")
    st.markdown("Comprehensive analysis of IPL cricket data from 2008 onwards")
    
    # Load data
    df = load_data()
    
    st.success(f"Dataset loaded successfully! Shape: {df.shape}")
    
    # Sidebar
    st.sidebar.title("Analysis Categories")
    analysis_type = st.sidebar.selectbox(
        "Select Analysis Type",
        [
            "📊 Overview Dashboard",
            "🏆 Team Performance", 
            "👤 Player Performance",
            "📈 Season-wise Analysis",
            "⚔️ Head-to-Head Analysis",
            "🏟️ Venue Analysis",
            "🎯 Toss Analysis"
        ]
    )
    
    # Filters
    st.sidebar.markdown("---")
    st.sidebar.subheader("Filters")
    
    # Season filter
    seasons = sorted(df['season'].unique())
    selected_seasons = st.sidebar.multiselect(
        "Select Seasons",
        options=seasons,
        default=seasons
    )
    
    # Team filter
    teams = sorted(df['batting_team'].unique())
    selected_teams = st.sidebar.multiselect(
        "Select Teams",
        options=teams,
        default=teams
    )
    
    # Apply filters
    filtered_df = df[
        (df['season'].isin(selected_seasons)) & 
        (df['batting_team'].isin(selected_teams))
    ]
    
    # Main content based on selection
    if analysis_type == "📊 Overview Dashboard":
        show_overview(filtered_df)
    elif analysis_type == "🏆 Team Performance":
        show_team_performance(filtered_df)
    elif analysis_type == "👤 Player Performance":
        show_player_performance(filtered_df)
    elif analysis_type == "📈 Season-wise Analysis":
        show_season_analysis(filtered_df)
    elif analysis_type == "⚔️ Head-to-Head Analysis":
        show_head_to_head(filtered_df)
    elif analysis_type == "🏟️ Venue Analysis":
        show_venue_analysis(filtered_df)
    elif analysis_type == "🎯 Toss Analysis":
        show_toss_analysis(filtered_df)

def show_overview(df):
    st.header("📊 IPL Overview Dashboard")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_matches = df['match_id'].nunique()
        st.metric("Total Matches", total_matches)
    
    with col2:
        total_seasons = df['season'].nunique()
        st.metric("Total Seasons", total_seasons)
    
    with col3:
        total_teams = df['batting_team'].nunique()
        st.metric("Total Teams", total_teams)
    
    with col4:
        total_players = df['batter'].nunique()
        st.metric("Total Players", total_players)
    
    # Matches per season
    st.subheader("Matches per Season")
    matches_per_season = df.groupby('season')['match_id'].nunique().reset_index()
    fig = px.bar(matches_per_season, x='season', y='match_id', 
                 title="Number of Matches per Season")
    st.plotly_chart(fig, use_container_width=True)
    
    # Teams performance overview
    st.subheader("Team Performance Overview")
    col1, col2 = st.columns(2)
    
    with col1:
        # Total runs by team
        team_runs = df.groupby('batting_team')['runs_total'].sum().sort_values(ascending=False)
        fig = px.bar(team_runs, x=team_runs.index, y=team_runs.values,
                     title="Total Runs by Team")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Average score by team
        avg_score = df.groupby(['match_id', 'batting_team'])['runs_total'].sum().reset_index()
        avg_score = avg_score.groupby('batting_team')['runs_total'].mean().sort_values(ascending=False)
        fig = px.bar(avg_score, x=avg_score.index, y=avg_score.values,
                     title="Average Score per Match by Team")
        st.plotly_chart(fig, use_container_width=True)

def show_team_performance(df):
    st.header("🏆 Team Performance Analysis")
    
    team = st.selectbox("Select Team", df['batting_team'].unique())
    
    # Filter data for selected team
    team_df = df[df['batting_team'] == team]
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_matches = team_df['match_id'].nunique()
        st.metric("Matches Played", total_matches)
    
    with col2:
        total_runs = team_df['runs_total'].sum()
        st.metric("Total Runs", f"{total_runs:,}")
    
    with col3:
        avg_score = team_df.groupby('match_id')['runs_total'].sum().mean()
        st.metric("Average Score", f"{avg_score:.1f}")
    
    with col4:
        wins = df[df['win_outcome'] == team]['match_id'].nunique()
        st.metric("Matches Won", wins)
    
    # Performance over seasons
    st.subheader(f"{team} Performance Over Seasons")
    
    # Wins per season
    wins_per_season = df[df['win_outcome'] == team].groupby('season')['match_id'].nunique()
    matches_per_season = team_df.groupby('season')['match_id'].nunique()
    win_rate = (wins_per_season / matches_per_season * 100).fillna(0)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=win_rate.index, y=win_rate.values, 
                            mode='lines+markers', name='Win Rate %'))
    fig.update_layout(title=f"{team} Win Rate Over Seasons")
    st.plotly_chart(fig, use_container_width=True)
    
    # Batting performance by venue
    st.subheader("Batting Performance by Venue")
    venue_performance = team_df.groupby('venue').agg({
        'runs_total': 'sum',
        'match_id': 'nunique'
    }).reset_index()
    venue_performance['avg_score'] = venue_performance['runs_total'] / venue_performance['match_id']
    
    fig = px.bar(venue_performance.nlargest(10, 'avg_score'), 
                 x='venue', y='avg_score', title="Top 10 Venues by Average Score")
    st.plotly_chart(fig, use_container_width=True)

def show_player_performance(df):
    st.header("👤 Player Performance Analysis")
    
    analysis_type = st.radio("Select Player Type", ["Batters", "Bowlers"])
    
    if analysis_type == "Batters":
        # Batter analysis
        batters = df['batter'].unique()
        selected_batter = st.selectbox("Select Batter", batters)
        
        batter_df = df[df['batter'] == selected_batter]
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_runs = batter_df['runs_batter'].sum()
            st.metric("Total Runs", total_runs)
        
        with col2:
            total_balls = batter_df['balls_faced'].sum()
            strike_rate = (total_runs / total_balls * 100) if total_balls > 0 else 0
            st.metric("Strike Rate", f"{strike_rate:.2f}")
        
        with col3:
            total_matches = batter_df['match_id'].nunique()
            st.metric("Matches Played", total_matches)
        
        with col4:
            average = total_runs / total_matches if total_matches > 0 else 0
            st.metric("Average", f"{average:.2f}")
        
        # Runs over seasons
        runs_per_season = batter_df.groupby('season')['runs_batter'].sum()
        fig = px.line(runs_per_season, title=f"{selected_batter} - Runs per Season")
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance against different teams
        st.subheader("Performance Against Teams")
        vs_teams = batter_df.groupby('bowling_team')['runs_batter'].sum().sort_values(ascending=False)
        fig = px.bar(vs_teams, x=vs_teams.index, y=vs_teams.values,
                     title="Runs Against Different Teams")
        st.plotly_chart(fig, use_container_width=True)
    
    else:
        # Bowler analysis
        bowlers = df['bowler'].unique()
        selected_bowler = st.selectbox("Select Bowler", bowlers)
        
        bowler_df = df[df['bowler'] == selected_bowler]
        bowler_df = bowler_df[bowler_df['valid_ball'] == 1]  # Only valid deliveries
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_wickets = bowler_df['bowler_wicket'].sum()
            st.metric("Total Wickets", total_wickets)
        
        with col2:
            total_runs = bowler_df['runs_bowler'].sum()
            total_balls = bowler_df['valid_ball'].sum()
            economy = (total_runs / total_balls * 6) if total_balls > 0 else 0
            st.metric("Economy Rate", f"{economy:.2f}")
        
        with col3:
            total_matches = bowler_df['match_id'].nunique()
            st.metric("Matches Played", total_matches)
        
        with col4:
            average = total_runs / total_wickets if total_wickets > 0 else 0
            st.metric("Bowling Average", f"{average:.2f}")
        
        # Wickets over seasons
        wickets_per_season = bowler_df.groupby('season')['bowler_wicket'].sum()
        fig = px.line(wickets_per_season, title=f"{selected_bowler} - Wickets per Season")
        st.plotly_chart(fig, use_container_width=True)

def show_season_analysis(df):
    st.header("📈 Season-wise Analysis")
    
    selected_season = st.selectbox("Select Season", sorted(df['season'].unique()))
    
    season_df = df[df['season'] == selected_season]
    
    # Top run scorers
    st.subheader(f"Top 10 Run Scorers - {selected_season}")
    top_batters = season_df.groupby('batter').agg({
        'runs_batter': 'sum',
        'match_id': 'nunique',
        'balls_faced': 'sum'
    }).reset_index()
    top_batters = top_batters.nlargest(10, 'runs_batter')
    top_batters['strike_rate'] = (top_batters['runs_batter'] / top_batters['balls_faced'] * 100).round(2)
    
    fig = px.bar(top_batters, x='batter', y='runs_batter',
                 title=f"Top Run Scorers - {selected_season}")
    st.plotly_chart(fig, use_container_width=True)
    
    # Top wicket takers
    st.subheader(f"Top 10 Wicket Takers - {selected_season}")
    valid_balls = season_df[season_df['valid_ball'] == 1]
    top_bowlers = valid_balls.groupby('bowler').agg({
        'bowler_wicket': 'sum',
        'runs_bowler': 'sum',
        'valid_ball': 'sum'
    }).reset_index()
    top_bowlers = top_bowlers.nlargest(10, 'bowler_wicket')
    top_bowlers['economy'] = (top_bowlers['runs_bowler'] / top_bowlers['valid_ball'] * 6).round(2)
    
    fig = px.bar(top_bowlers, x='bowler', y='bowler_wicket',
                 title=f"Top Wicket Takers - {selected_season}")
    st.plotly_chart(fig, use_container_width=True)
    
    # Team standings
    st.subheader(f"Team Standings - {selected_season}")
    team_wins = season_df[season_df['win_outcome'].notna()].groupby('win_outcome')['match_id'].nunique()
    team_matches = season_df.groupby('batting_team')['match_id'].nunique()
    
    standings = pd.DataFrame({
        'team': team_matches.index,
        'matches': team_matches.values,
        'wins': team_wins.reindex(team_matches.index, fill_value=0).values
    })
    standings['win_percentage'] = (standings['wins'] / standings['matches'] * 100).round(2)
    standings = standings.sort_values('win_percentage', ascending=False)
    
    fig = px.bar(standings, x='team', y='win_percentage',
                 title=f"Team Win Percentage - {selected_season}")
    st.plotly_chart(fig, use_container_width=True)

def show_head_to_head(df):
    st.header("⚔️ Head-to-Head Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        team1 = st.selectbox("Select Team 1", df['batting_team'].unique())
    
    with col2:
        team2 = st.selectbox("Select Team 2", 
                            [team for team in df['batting_team'].unique() if team != team1])
    
    # Get matches between these teams
    matches_team1_batting = df[(df['batting_team'] == team1) & (df['bowling_team'] == team2)]
    matches_team2_batting = df[(df['batting_team'] == team2) & (df['bowling_team'] == team1)]
    
    all_matches = pd.concat([matches_team1_batting, matches_team2_batting])
    unique_matches = all_matches['match_id'].unique()
    
    # Win statistics
    matches_df = df[df['match_id'].isin(unique_matches)]
    team1_wins = matches_df[matches_df['win_outcome'] == team1]['match_id'].nunique()
    team2_wins = matches_df[matches_df['win_outcome'] == team2]['match_id'].nunique()
    total_matches = len(unique_matches)
    
    # Display head-to-head stats
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(f"{team1} Wins", team1_wins)
    
    with col2:
        st.metric(f"{team2} Wins", team2_wins)
    
    with col3:
        st.metric("Total Matches", total_matches)
    
    # Win comparison chart
    wins_data = pd.DataFrame({
        'Team': [team1, team2],
        'Wins': [team1_wins, team2_wins]
    })
    
    fig = px.pie(wins_data, values='Wins', names='Team', 
                 title=f"Head-to-Head Win Distribution")
    st.plotly_chart(fig, use_container_width=True)
    
    # Average scores in matches between these teams
    team1_avg = matches_team1_batting.groupby('match_id')['runs_total'].sum().mean()
    team2_avg = matches_team2_batting.groupby('match_id')['runs_total'].sum().mean()
    
    avg_scores = pd.DataFrame({
        'Team': [team1, team2],
        'Average Score': [team1_avg, team2_avg]
    })
    
    fig = px.bar(avg_scores, x='Team', y='Average Score',
                 title="Average Scores in Head-to-Head Matches")
    st.plotly_chart(fig, use_container_width=True)

def show_venue_analysis(df):
    st.header("🏟️ Venue Analysis")
    
    venue = st.selectbox("Select Venue", df['venue'].unique())
    
    venue_df = df[df['venue'] == venue]
    
    # Basic venue stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_matches = venue_df['match_id'].nunique()
        st.metric("Total Matches", total_matches)
    
    with col2:
        avg_runs = venue_df.groupby('match_id')['runs_total'].sum().mean()
        st.metric("Average Total Runs", f"{avg_runs:.1f}")
    
    with col3:
        avg_first_innings = venue_df[venue_df['innings'] == 1].groupby('match_id')['runs_total'].sum().mean()
        st.metric("Average 1st Innings Score", f"{avg_first_innings:.1f}")
    
    with col4:
        wins_batting_first = venue_df[venue_df['innings'] == 1]['match_id'].nunique()
        win_percentage = (venue_df[venue_df['win_outcome'] == venue_df['batting_team']]['match_id'].nunique() / total_matches * 100) if total_matches > 0 else 0
        st.metric("Batting First Win %", f"{win_percentage:.1f}%")
    
    # Team performance at venue
    st.subheader("Team Performance at this Venue")
    team_performance = venue_df.groupby('batting_team').agg({
        'runs_total': 'sum',
        'match_id': 'nunique'
    }).reset_index()
    team_performance['avg_score'] = team_performance['runs_total'] / team_performance['match_id']
    team_performance = team_performance.sort_values('avg_score', ascending=False)
    
    fig = px.bar(team_performance.head(10), x='batting_team', y='avg_score',
                 title="Top 10 Teams by Average Score at this Venue")
    st.plotly_chart(fig, use_container_width=True)
    
    # Highest totals at venue
    st.subheader("Highest Totals at this Venue")
    high_totals = venue_df.groupby(['match_id', 'batting_team'])['runs_total'].sum().reset_index()
    high_totals = high_totals.nlargest(10, 'runs_total')
    
    fig = px.bar(high_totals, x='batting_team', y='runs_total',
                 title="Highest Team Totals at this Venue")
    st.plotly_chart(fig, use_container_width=True)

def show_toss_analysis(df):
    st.header("🎯 Toss Analysis")
    
    # Toss impact on match results
    matches_with_toss = df.drop_duplicates('match_id')[['match_id', 'toss_winner', 'toss_decision', 'win_outcome']]
    matches_with_toss = matches_with_toss.dropna()
    
    matches_with_toss['toss_winner_won'] = matches_with_toss['toss_winner'] == matches_with_toss['win_outcome']
    
    toss_win_rate = matches_with_toss['toss_winner_won'].mean() * 100
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Toss Winner Win Rate", f"{toss_win_rate:.2f}%")
    
    with col2:
        total_matches = len(matches_with_toss)
        st.metric("Matches Analyzed", total_matches)
    
    with col3:
        bat_first_wins = matches_with_toss[
            (matches_with_toss['toss_decision'] == 'bat') & 
            (matches_with_toss['toss_winner_won'] == True)
        ].shape[0]
        bat_first_matches = matches_with_toss[matches_with_toss['toss_decision'] == 'bat'].shape[0]
        bat_first_win_rate = (bat_first_wins / bat_first_matches * 100) if bat_first_matches > 0 else 0
        st.metric("Bat First Win Rate", f"{bat_first_win_rate:.2f}%")
    
    # Toss decision preference by team
    st.subheader("Toss Decision Preference by Teams")
    toss_preference = matches_with_toss.groupby(['toss_winner', 'toss_decision']).size().unstack(fill_value=0)
    toss_preference['total'] = toss_preference.sum(axis=1)
    toss_preference['bat_percentage'] = (toss_preference.get('bat', 0) / toss_preference['total'] * 100).round(2)
    toss_preference = toss_preference.sort_values('bat_percentage', ascending=False)
    
    fig = px.bar(toss_preference.head(15), x=toss_preference.index[:15], y='bat_percentage',
                 title="Teams that Prefer to Bat First After Winning Toss (Top 15)")
    st.plotly_chart(fig, use_container_width=True)
    
    # Toss impact by venue
    st.subheader("Toss Impact by Venue")
    venue_toss_impact = df.drop_duplicates('match_id').groupby('venue').agg({
        'toss_winner': lambda x: (x == df[df['match_id'].isin(x.index)]['win_outcome'].values).mean() * 100
    }).reset_index()
    venue_toss_impact.columns = ['venue', 'toss_win_rate']
    venue_toss_impact = venue_toss_impact.sort_values('toss_win_rate', ascending=False)
    
    fig = px.bar(venue_toss_impact.head(15), x='venue', y='toss_win_rate',
                 title="Venues Where Toss Has Highest Impact (Top 15)")
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
