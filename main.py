import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import zipfile
# Set page configuration
st.set_page_config(page_title="IPL Data Analysis", layout="wide")

# Load data
@st.cache_data
def load_data():
    with zipfile.ZipFile('archive.zip', 'r') as zip_ref:
        zip_ref.extractall()
    df = pd.read_csv('IPL.csv')  # Update with your file path
    return df

# Load the data
df = load_data()

# Convert date to datetime
df['date'] = pd.to_datetime(df['date'])

# Add year column
df['year'] = df['date'].dt.year

# Sidebar for category selection
st.sidebar.title("IPL Data Analysis")
st.sidebar.markdown("Select a category for analysis:")

category = st.sidebar.selectbox(
    "Analysis Category",
    ["Team Performance", "Player Stats", "Match Analysis", "Venue Insights", "Seasonal Trends", "Toss Impact"]
)

# Main content based on selected category
st.title(f"IPL Data Analysis: {category}")

# Team Performance Analysis
if category == "Team Performance":
    st.header("Team Performance Analysis")
    
    # Team selection
    teams = sorted(df['batting_team'].unique())
    selected_team = st.selectbox("Select Team", teams)
    
    # Filter data for selected team
    team_data = df[(df['batting_team'] == selected_team) | (df['bowling_team'] == selected_team)]
    
    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate metrics
    matches_played = team_data['match_id'].nunique()
    matches_won = team_data[team_data['match_won_by'] == selected_team]['match_id'].nunique()
    win_percentage = (matches_won / matches_played) * 100 if matches_played > 0 else 0
    total_runs = team_data[team_data['batting_team'] == selected_team]['runs_total'].sum()
    
    # Display metrics
    col1.metric("Matches Played", matches_played)
    col2.metric("Matches Won", matches_won)
    col3.metric("Win Percentage", f"{win_percentage:.1f}%")
    col4.metric("Total Runs", total_runs)
    
    # Team performance chart
    st.subheader("Team Performance Over Seasons")
    team_season = team_data.groupby(['season', 'batting_team']).agg(
        total_runs=('runs_total', 'sum'),
        total_wickets=('wicket_kind', lambda x: x.notna().sum())
    ).reset_index()
    
    fig = px.bar(team_season[team_season['batting_team'] == selected_team], 
                 x='season', y='total_runs', 
                 title=f"Runs Scored by {selected_team} per Season",
                 color='season')
    st.plotly_chart(fig, use_container_width=True)
    
    # Opposition analysis
    st.subheader("Performance Against Opposition")
    opposition = sorted(list(set(teams) - {selected_team}))
    selected_opposition = st.selectbox("Select Opposition", opposition)
    
    vs_data = team_data[((team_data['batting_team'] == selected_team) & (team_data['bowling_team'] == selected_opposition)) |
                        ((team_data['batting_team'] == selected_opposition) & (team_data['bowling_team'] == selected_team))]
    
    vs_matches = vs_data['match_id'].nunique()
    vs_wins = vs_data[vs_data['match_won_by'] == selected_team]['match_id'].nunique()
    vs_win_pct = (vs_wins / vs_matches) * 100 if vs_matches > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    col1.metric(f"Matches vs {selected_opposition}", vs_matches)
    col2.metric(f"Wins vs {selected_opposition}", vs_wins)
    col3.metric(f"Win % vs {selected_opposition}", f"{vs_win_pct:.1f}%")

# Player Stats Analysis
elif category == "Player Stats":
    st.header("Player Statistics Analysis")
    
    # Player selection
    players = sorted(df['batter'].unique())
    selected_player = st.selectbox("Select Player", players)
    
    # Filter data for selected player
    player_data = df[(df['batter'] == selected_player) | (df['bowler'] == selected_player)]
    
    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate batting metrics
    batting_data = df[df['batter'] == selected_player]
    runs_scored = batting_data['runs_batter'].sum()
    balls_faced = batting_data['balls_faced'].sum()
    strike_rate = (runs_scored / balls_faced) * 100 if balls_faced > 0 else 0
    dismissals = batting_data['player_out'].notna().sum()
    batting_avg = runs_scored / dismissals if dismissals > 0 else runs_scored
    
    # Calculate bowling metrics
    bowling_data = df[df['bowler'] == selected_player]
    wickets_taken = bowling_data['wicket_kind'].notna().sum()
    balls_bowled = bowling_data['valid_ball'].sum()
    economy_rate = (bowling_data['runs_total'].sum() / balls_bowled) * 6 if balls_bowled > 0 else 0
    
    # Display metrics
    col1.metric("Runs Scored", runs_scored)
    col2.metric("Batting Average", f"{batting_avg:.2f}")
    col3.metric("Strike Rate", f"{strike_rate:.2f}")
    col4.metric("Wickets Taken", wickets_taken)
    
    # Player performance chart
    st.subheader("Performance Over Seasons")
    player_season = player_data.groupby(['season']).agg(
        runs=('runs_batter', 'sum'),
        wickets=('wicket_kind', lambda x: x.notna().sum())
    ).reset_index()
    
    fig = px.bar(player_season, x='season', y=['runs', 'wickets'], 
                 title=f"{selected_player}'s Performance per Season",
                 barmode='group')
    st.plotly_chart(fig, use_container_width=True)
    
    # Player vs Team analysis
    st.subheader("Performance Against Teams")
    opposition = sorted(list(set(teams) - {selected_player}))
    selected_opposition = st.selectbox("Select Opposition Team", opposition)
    
    vs_batting = df[(df['batter'] == selected_player) & (df['bowling_team'] == selected_opposition)]
    vs_bowling = df[(df['bowler'] == selected_player) & (df['batting_team'] == selected_opposition)]
    
    vs_runs = vs_batting['runs_batter'].sum()
    vs_wickets = vs_bowling['wicket_kind'].notna().sum()
    
    col1, col2 = st.columns(2)
    col1.metric(f"Runs vs {selected_opposition}", vs_runs)
    col2.metric(f"Wickets vs {selected_opposition}", vs_wickets)

# Match Analysis
elif category == "Match Analysis":
    st.header("Match Analysis")
    
    # Match type selection
    match_types = sorted(df['match_type'].unique())
    selected_type = st.selectbox("Select Match Type", match_types)
    
    # Filter data for selected match type
    match_data = df[df['match_type'] == selected_type]
    
    # Match outcome distribution
    st.subheader("Match Outcome Distribution")
    outcome_counts = match_data['win_outcome'].value_counts()
    
    fig = px.pie(values=outcome_counts.values, names=outcome_counts.index, 
                 title=f"Match Outcomes - {selected_type} Matches")
    st.plotly_chart(fig, use_container_width=True)
    
    # Runs distribution
    st.subheader("Team Runs Distribution")
    fig = px.histogram(match_data, x='team_runs', nbins=20, 
                       title=f"Distribution of Team Scores in {selected_type} Matches")
    st.plotly_chart(fig, use_container_width=True)
    
    # Wickets distribution
    st.subheader("Wickets Distribution")
    fig = px.histogram(match_data, x='team_wicket', nbins=11, 
                       title=f"Distribution of Wickets Fallen in {selected_type} Matches")
    st.plotly_chart(fig, use_container_width=True)

# Venue Insights
elif category == "Venue Insights":
    st.header("Venue Analysis")
    
    # Venue selection
    venues = sorted(df['venue'].unique())
    selected_venue = st.selectbox("Select Venue", venues)
    
    # Filter data for selected venue
    venue_data = df[df['venue'] == selected_venue]
    
    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate metrics
    matches_played = venue_data['match_id'].nunique()
    avg_runs = venue_data['team_runs'].mean()
    avg_wickets = venue_data['team_wicket'].mean()
    toss_wins = venue_data[venue_data['toss_winner'] == venue_data['match_won_by']]['match_id'].nunique()
    toss_win_pct = (toss_wins / matches_played) * 100 if matches_played > 0 else 0
    
    # Display metrics
    col1.metric("Matches Played", matches_played)
    col2.metric("Average Score", f"{avg_runs:.1f}")
    col3.metric("Average Wickets", f"{avg_wickets:.1f}")
    col4.metric("Toss Win %", f"{toss_win_pct:.1f}%")
    
    # Venue performance chart
    st.subheader("Team Performance at Venue")
    venue_teams = venue_data.groupby('batting_team').agg(
        avg_runs=('team_runs', 'mean'),
        matches=('match_id', 'nunique')
    ).reset_index()
    
    fig = px.bar(venue_teams, x='batting_team', y='avg_runs', 
                 title=f"Average Team Scores at {selected_venue}",
                 color='batting_team')
    st.plotly_chart(fig, use_container_width=True)
    
    # Toss impact at venue
    st.subheader("Toss Impact at Venue")
    toss_data = venue_data[['match_id', 'toss_winner', 'toss_decision', 'match_won_by']].drop_duplicates()
    toss_impact = toss_data.groupby(['toss_decision', 'match_won_by']).size().unstack().fillna(0)
    
    fig = px.imshow(toss_impact, text_auto=True, aspect="auto",
                    title=f"Toss Impact at {selected_venue}")
    st.plotly_chart(fig, use_container_width=True)

# Seasonal Trends
elif category == "Seasonal Trends":
    st.header("Seasonal Trends Analysis")
    
    # Season selection
    seasons = sorted(df['season'].unique())
    selected_season = st.selectbox("Select Season", seasons)
    
    # Filter data for selected season
    season_data = df[df['season'] == selected_season]
    
    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate metrics
    matches_played = season_data['match_id'].nunique()
    total_runs = season_data['runs_total'].sum()
    avg_score = season_data['team_runs'].mean()
    total_wickets = season_data['wicket_kind'].notna().sum()
    
    # Display metrics
    col1.metric("Matches Played", matches_played)
    col2.metric("Total Runs", total_runs)
    col3.metric("Average Score", f"{avg_score:.1f}")
    col4.metric("Total Wickets", total_wickets)
    
    # Team performance in season
    st.subheader("Team Performance in Season")
    team_season = season_data.groupby('batting_team').agg(
        total_runs=('runs_total', 'sum'),
        avg_runs=('team_runs', 'mean'),
        matches=('match_id', 'nunique')
    ).reset_index()
    
    fig = px.bar(team_season, x='batting_team', y='total_runs', 
                 title=f"Total Runs by Teams in {selected_season}",
                 color='batting_team')
    st.plotly_chart(fig, use_container_width=True)
    
    # Top performers in season
    st.subheader("Top Performers in Season")
    
    # Top batsmen
    top_batsmen = season_data[season_data['batter_runs'] > 0].groupby('batter').agg(
        runs=('batter_runs', 'sum'),
        matches=('match_id', 'nunique')
    ).sort_values('runs', ascending=False).head(5)
    
    # Top bowlers
    top_bowlers = season_data.groupby('bowler').agg(
        wickets=('bowler_wicket', 'sum'),
        matches=('match_id', 'nunique')
    ).sort_values('wickets', ascending=False).head(5)
    
    col1, col2 = st.columns(2)
    col1.subheader("Top Batsmen")
    col1.dataframe(top_batsmen)
    
    col2.subheader("Top Bowlers")
    col2.dataframe(top_bowlers)

# Toss Impact
elif category == "Toss Impact":
    st.header("Toss Impact Analysis")
    
    # Toss decision analysis
    st.subheader("Toss Decision Trends")
    toss_decisions = df.groupby(['season', 'toss_decision']).size().unstack().fillna(0)
    
    fig = px.bar(toss_decisions, barmode='group', 
                 title="Toss Decisions by Season")
    st.plotly_chart(fig, use_container_width=True)
    
    # Toss impact on match result
    st.subheader("Toss Impact on Match Result")
    toss_results = df[['match_id', 'toss_winner', 'toss_decision', 'match_won_by']].drop_duplicates()
    toss_results['toss_win_match_win'] = toss_results['toss_winner'] == toss_results['match_won_by']
    
    toss_impact = toss_results.groupby(['toss_decision', 'toss_win_match_win']).size().unstack().fillna(0)
    
    fig = px.bar(toss_impact, barmode='group', 
                 title="Impact of Toss on Match Result")
    st.plotly_chart(fig, use_container_width=True)
    
    # Toss impact by team
    st.subheader("Toss Impact by Team")
    teams = sorted(df['batting_team'].unique())
    selected_team = st.selectbox("Select Team", teams)
    
    team_toss = toss_results[toss_results['toss_winner'] == selected_team]
    team_toss_wins = team_toss['match_won_by'].value_counts()
    
    col1, col2 = st.columns(2)
    col1.metric(f"{selected_team} Won Toss", team_toss.shape[0])
    col2.metric(f"{selected_team} Won Match After Toss", team_toss_wins.get(selected_team, 0))
    
    fig = px.pie(values=team_toss_wins.values, names=team_toss_wins.index, 
                 title=f"Match Results After {selected_team} Won Toss")
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("IPL Data Analysis Dashboard")
st.sidebar.markdown("Built with Streamlit")
