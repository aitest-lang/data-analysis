import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import zipfile

with zipfile.ZipFile('archive.zip', 'r') as zip_ref:
    zip_ref.extractall()
# Page configuration
st.set_page_config(
    page_title="IPL Data Analysis Dashboard",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Team aliases mapping
TEAM_ALIASES = {
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
    "Rising Pune Supergiant": "Rising Pune Supergiant",
    "Kings XI Punjab": "Punjab Kings",
    "Delhi Daredevils": "Delhi Capitals",
    "Pune Warriors": "Pune Warriors India",
}

# Function to load and clean data
@st.cache_data(ttl=3600, show_spinner=True)
def load_data():
    # Load data
    df = pd.read_csv("IPL.csv")
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Standardize team names
    for col in ['batting_team', 'bowling_team', 'match_won_by', 'toss_winner', 'team_reviewed']:
        if col in df.columns:
            df[col] = df[col].replace(TEAM_ALIASES)
    
    # Extract year and month from date
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    
    # Create match result column
    df['result'] = np.where(df['match_won_by'] == df['batting_team'], 'Win', 
                           np.where(df['match_won_by'] == df['bowling_team'], 'Loss', 'Draw/Tie'))
    
    # Create phase of play (Powerplay: 1-6 overs, Middle: 7-15, Death: 16-20)
    df['phase'] = pd.cut(df['over'], bins=[0, 6, 15, 20], labels=['Powerplay', 'Middle', 'Death'])
    
    # Calculate run rate
    df['run_rate'] = df['runs_total'] / df['ball']
    
    # Calculate cumulative team score in innings
    df = df.sort_values(['match_id', 'innings', 'over', 'ball'])
    df['cumulative_runs'] = df.groupby(['match_id', 'innings'])['runs_total'].cumsum()
    df['cumulative_wickets'] = df.groupby(['match_id', 'innings'])['wicket_kind'].apply(lambda x: x.notna().cumsum())
    
    # Create dismissal type column
    df['dismissal_type'] = df['wicket_kind'].fillna('Not Out')
    
    return df

# Function to get unique teams
def get_unique_teams(df):
    teams = set(df['batting_team'].unique()) | set(df['bowling_team'].unique())
    return sorted(list(teams))

# Function to get unique seasons
def get_unique_seasons(df):
    return sorted(df['season'].unique())

# Function to get unique venues
def get_unique_venues(df):
    return sorted(df['venue'].unique())

# Function to get unique players
def get_unique_players(df):
    batters = set(df['batter'].unique())
    bowlers = set(df['bowler'].unique())
    return sorted(list(batters | bowlers))

# Load data
with st.spinner("Loading IPL data..."):
    df = load_data()

# Sidebar
st.sidebar.title("IPL Data Analysis")
st.sidebar.markdown("Select analysis category and filters to explore IPL data")

# Category selection
category = st.sidebar.selectbox(
    "Select Analysis Category",
    ["Match Overview", "Team Performance", "Player Analysis", "Match Dynamics", "Seasonal Insights"]
)

# Get unique values for filters
teams = get_unique_teams(df)
seasons = get_unique_seasons(df)
venues = get_unique_venues(df)
players = get_unique_players(df)

# Filters based on category
if category == "Match Overview":
    selected_seasons = st.sidebar.multiselect("Select Seasons", seasons, default=seasons[-3:])
    selected_venues = st.sidebar.multiselect("Select Venues", venues, default=venues[:5])
elif category == "Team Performance":
    selected_teams = st.sidebar.multiselect("Select Teams", teams, default=teams[:4])
    selected_seasons = st.sidebar.multiselect("Select Seasons", seasons, default=seasons[-3:])
elif category == "Player Analysis":
    selected_players = st.sidebar.multiselect("Select Players", players, default=players[:4])
    selected_seasons = st.sidebar.multiselect("Select Seasons", seasons, default=seasons[-3:])
elif category == "Match Dynamics":
    selected_phases = st.sidebar.multiselect(
        "Select Phases", 
        ["Powerplay", "Middle", "Death"], 
        default=["Powerplay", "Middle", "Death"]
    )
    selected_seasons = st.sidebar.multiselect("Select Seasons", seasons, default=seasons[-3:])
else:  # Seasonal Insights
    selected_seasons = st.sidebar.multiselect("Select Seasons", seasons, default=seasons[-5:])

# Main dashboard
st.title("🏏 IPL Data Analysis Dashboard")
st.markdown(f"### {category}")

# Apply filters
filtered_df = df.copy()

if category == "Match Overview":
    filtered_df = filtered_df[filtered_df['season'].isin(selected_seasons)]
    filtered_df = filtered_df[filtered_df['venue'].isin(selected_venues)]
elif category == "Team Performance":
    filtered_df = filtered_df[filtered_df['season'].isin(selected_seasons)]
    team_filter = (filtered_df['batting_team'].isin(selected_teams)) | (filtered_df['bowling_team'].isin(selected_teams))
    filtered_df = filtered_df[team_filter]
elif category == "Player Analysis":
    filtered_df = filtered_df[filtered_df['season'].isin(selected_seasons)]
    player_filter = (filtered_df['batter'].isin(selected_players)) | (filtered_df['bowler'].isin(selected_players))
    filtered_df = filtered_df[player_filter]
elif category == "Match Dynamics":
    filtered_df = filtered_df[filtered_df['season'].isin(selected_seasons)]
    filtered_df = filtered_df[filtered_df['phase'].isin(selected_phases)]
else:  # Seasonal Insights
    filtered_df = filtered_df[filtered_df['season'].isin(selected_seasons)]

# Display filtered data info
st.markdown(f"Showing data for **{len(filtered_df)} records** from **{filtered_df['match_id'].nunique()} matches**")

# Category-specific analysis
if category == "Match Overview":
    st.subheader("Match Overview Analysis")
    
    # Create columns for layout
    col1, col2 = st.columns(2)
    
    # Matches per season
    matches_per_season = filtered_df.groupby('season')['match_id'].nunique().reset_index()
    fig1 = px.bar(
        matches_per_season, 
        x='season', 
        y='match_id',
        title="Matches per Season",
        labels={'season': 'Season', 'match_id': 'Number of Matches'}
    )
    fig1.update_layout(xaxis_title="Season", yaxis_title="Number of Matches")
    col1.plotly_chart(fig1, use_container_width=True)
    
    # Matches per venue
    matches_per_venue = filtered_df.groupby('venue')['match_id'].nunique().reset_index().sort_values('match_id', ascending=False)
    fig2 = px.bar(
        matches_per_venue.head(10), 
        x='venue', 
        y='match_id',
        title="Top 10 Venues by Matches",
        labels={'venue': 'Venue', 'match_id': 'Number of Matches'}
    )
    fig2.update_layout(xaxis_title="Venue", yaxis_title="Number of Matches", xaxis={'categoryorder': 'total descending'})
    col2.plotly_chart(fig2, use_container_width=True)
    
    # Match outcomes
    col1, col2 = st.columns(2)
    
    # Win by runs vs wickets
    win_outcomes = filtered_df.drop_duplicates('match_id').groupby(['season', 'win_outcome']).size().reset_index(name='count')
    fig3 = px.bar(
        win_outcomes, 
        x='season', 
        y='count', 
        color='win_outcome',
        title="Match Outcomes by Season",
        labels={'season': 'Season', 'count': 'Number of Matches', 'win_outcome': 'Outcome'}
    )
    fig3.update_layout(xaxis_title="Season", yaxis_title="Number of Matches")
    col1.plotly_chart(fig3, use_container_width=True)
    
    # Toss impact
    toss_impact = filtered_df.drop_duplicates('match_id').copy()
    toss_impact['toss_match'] = toss_impact['toss_winner'] == toss_impact['match_won_by']
    toss_win_percentage = toss_impact.groupby('season')['toss_match'].mean().reset_index()
    toss_win_percentage['toss_match'] = toss_win_percentage['toss_match'] * 100
    
    fig4 = px.line(
        toss_win_percentage, 
        x='season', 
        y='toss_match',
        title="Toss Win to Match Win Percentage",
        labels={'season': 'Season', 'toss_match': 'Percentage (%)'},
        markers=True
    )
    fig4.update_layout(xaxis_title="Season", yaxis_title="Percentage (%)")
    col2.plotly_chart(fig4, use_container_width=True)

elif category == "Team Performance":
    st.subheader("Team Performance Analysis")
    
    # Create columns for layout
    col1, col2 = st.columns(2)
    
    # Team wins
    team_wins = filtered_df.drop_duplicates('match_id').groupby(['season', 'match_won_by']).size().reset_index(name='wins')
    fig1 = px.bar(
        team_wins, 
        x='season', 
        y='wins', 
        color='match_won_by',
        title="Team Wins by Season",
        labels={'season': 'Season', 'wins': 'Number of Wins', 'match_won_by': 'Team'}
    )
    fig1.update_layout(xaxis_title="Season", yaxis_title="Number of Wins")
    col1.plotly_chart(fig1, use_container_width=True)
    
    # Win percentage
    team_matches = filtered_df.drop_duplicates('match_id').groupby(['season', 'match_won_by']).size().reset_index(name='matches')
    team_win_pct = team_matches.copy()
    team_win_pct['win_pct'] = team_win_pct['matches'] / team_win_pct.groupby('season')['matches'].transform('sum') * 100
    
    fig2 = px.line(
        team_win_pct, 
        x='season', 
        y='win_pct', 
        color='match_won_by',
        title="Team Win Percentage by Season",
        labels={'season': 'Season', 'win_pct': 'Win Percentage (%)', 'match_won_by': 'Team'},
        markers=True
    )
    fig2.update_layout(xaxis_title="Season", yaxis_title="Win Percentage (%)")
    col2.plotly_chart(fig2, use_container_width=True)
    
    # Team scoring patterns
    col1, col2 = st.columns(2)
    
    # Average runs per match
    team_scores = filtered_df.groupby(['season', 'batting_team', 'match_id'])['cumulative_runs'].max().reset_index()
    team_avg_scores = team_scores.groupby(['season', 'batting_team'])['cumulative_runs'].mean().reset_index()
    
    fig3 = px.bar(
        team_avg_scores, 
        x='season', 
        y='cumulative_runs', 
        color='batting_team',
        title="Average Team Scores by Season",
        labels={'season': 'Season', 'cumulative_runs': 'Average Score', 'batting_team': 'Team'}
    )
    fig3.update_layout(xaxis_title="Season", yaxis_title="Average Score")
    col1.plotly_chart(fig3, use_container_width=True)
    
    # Head-to-head record
    head_to_head = filtered_df.drop_duplicates('match_id').copy()
    head_to_head['team1'] = head_to_head.apply(lambda x: x['batting_team'] if x['innings'] == 1 else x['bowling_team'], axis=1)
    head_to_head['team2'] = head_to_head.apply(lambda x: x['bowling_team'] if x['innings'] == 1 else x['batting_team'], axis=1)
    head_to_head['winner'] = head_to_head['match_won_by']
    
    # Create pairs
    head_to_head['pair'] = head_to_head.apply(lambda x: tuple(sorted([x['team1'], x['team2']])), axis=1)
    h2h_counts = head_to_head.groupby(['pair', 'winner']).size().reset_index(name='count')
    
    # Get top 5 most frequent matchups
    top_matchups = head_to_head['pair'].value_counts().head(5).index
    h2h_top = h2h_counts[h2h_counts['pair'].isin(top_matchups)]
    h2h_top['matchup'] = h2h_top['pair'].apply(lambda x: f"{x[0]} vs {x[1]}")
    
    fig4 = px.bar(
        h2h_top, 
        x='matchup', 
        y='count', 
        color='winner',
        title="Head-to-Head Record (Top 5 Matchups)",
        labels={'matchup': 'Matchup', 'count': 'Wins', 'winner': 'Winning Team'}
    )
    fig4.update_layout(xaxis_title="Matchup", yaxis_title="Number of Wins")
    col2.plotly_chart(fig4, use_container_width=True)

elif category == "Player Analysis":
    st.subheader("Player Performance Analysis")
    
    # Create columns for layout
    col1, col2 = st.columns(2)
    
    # Top run scorers
    batsman_stats = filtered_df.groupby(['season', 'batter']).agg(
        runs=('runs_batter', 'sum'),
        balls=('balls_faced', 'sum'),
        dismissals=('dismissal_type', lambda x: (x != 'Not Out').sum())
    ).reset_index()
    
    batsman_stats['strike_rate'] = (batsman_stats['runs'] / batsman_stats['balls']) * 100
    batsman_stats['average'] = batsman_stats['runs'] / batsman_stats['dismissals'].replace(0, np.nan)
    
    top_run_scorers = batsman_stats.groupby(['season', 'batter'])['runs'].sum().reset_index()
    top_run_scorers = top_run_scorers.sort_values(['season', 'runs'], ascending=[True, False])
    top_run_scorers = top_run_scorers.groupby('season').head(5)
    
    fig1 = px.bar(
        top_run_scorers, 
        x='season', 
        y='runs', 
        color='batter',
        title="Top Run Scorers by Season",
        labels={'season': 'Season', 'runs': 'Runs', 'batter': 'Batsman'}
    )
    fig1.update_layout(xaxis_title="Season", yaxis_title="Runs")
    col1.plotly_chart(fig1, use_container_width=True)
    
    # Top wicket takers
    bowler_stats = filtered_df.groupby(['season', 'bowler']).agg(
        wickets=('wicket_kind', lambda x: x.notna().sum()),
        runs_conceded=('runs_bowler', 'sum'),
        balls_bowled=('valid_ball', 'sum')
    ).reset_index()
    
    bowler_stats['economy'] = (bowler_stats['runs_conceded'] / bowler_stats['balls_bowled']) * 6
    bowler_stats['bowling_avg'] = bowler_stats['runs_conceded'] / bowler_stats['wickets'].replace(0, np.nan)
    
    top_wicket_takers = bowler_stats.groupby(['season', 'bowler'])['wickets'].sum().reset_index()
    top_wicket_takers = top_wicket_takers.sort_values(['season', 'wickets'], ascending=[True, False])
    top_wicket_takers = top_wicket_takers.groupby('season').head(5)
    
    fig2 = px.bar(
        top_wicket_takers, 
        x='season', 
        y='wickets', 
        color='bowler',
        title="Top Wicket Takers by Season",
        labels={'season': 'Season', 'wickets': 'Wickets', 'bowler': 'Bowler'}
    )
    fig2.update_layout(xaxis_title="Season", yaxis_title="Wickets")
    col2.plotly_chart(fig2, use_container_width=True)
    
    # Player comparison
    col1, col2 = st.columns(2)
    
    # Strike rate vs average for batsmen
    batsman_comp = batsman_stats[batsman_stats['runs'] > 100]  # Filter for significant contributions
    
    fig3 = px.scatter(
        batsman_comp, 
        x='average', 
        y='strike_rate', 
        color='batter',
        size='runs',
        title="Batsman Performance: Strike Rate vs Average",
        labels={'average': 'Batting Average', 'strike_rate': 'Strike Rate', 'batter': 'Batsman'},
        hover_data=['season', 'runs']
    )
    fig3.update_layout(xaxis_title="Batting Average", yaxis_title="Strike Rate")
    col1.plotly_chart(fig3, use_container_width=True)
    
    # Economy vs wickets for bowlers
    bowler_comp = bowler_stats[bowler_stats['wickets'] > 10]  # Filter for significant contributions
    
    fig4 = px.scatter(
        bowler_comp, 
        x='economy', 
        y='wickets', 
        color='bowler',
        size='balls_bowled',
        title="Bowler Performance: Economy vs Wickets",
        labels={'economy': 'Economy Rate', 'wickets': 'Wickets', 'bowler': 'Bowler'},
        hover_data=['season', 'bowling_avg']
    )
    fig4.update_layout(xaxis_title="Economy Rate", yaxis_title="Wickets")
    col2.plotly_chart(fig4, use_container_width=True)

elif category == "Match Dynamics":
    st.subheader("Match Dynamics Analysis")
    
    # Create columns for layout
    col1, col2 = st.columns(2)
    
    # Run rate by phase
    phase_run_rate = filtered_df.groupby(['season', 'phase'])['run_rate'].mean().reset_index()
    
    fig1 = px.bar(
        phase_run_rate, 
        x='season', 
        y='run_rate', 
        color='phase',
        title="Average Run Rate by Phase",
        labels={'season': 'Season', 'run_rate': 'Run Rate', 'phase': 'Phase'}
    )
    fig1.update_layout(xaxis_title="Season", yaxis_title="Run Rate (Runs per Ball)")
    col1.plotly_chart(fig1, use_container_width=True)
    
    # Wickets by phase
    phase_wickets = filtered_df.groupby(['season', 'phase'])['wicket_kind'].apply(lambda x: x.notna().sum()).reset_index(name='wickets')
    phase_matches = filtered_df.groupby(['season', 'phase'])['match_id'].nunique().reset_index(name='matches')
    phase_wickets = pd.merge(phase_wickets, phase_matches, on=['season', 'phase'])
    phase_wickets['wickets_per_match'] = phase_wickets['wickets'] / phase_wickets['matches']
    
    fig2 = px.bar(
        phase_wickets, 
        x='season', 
        y='wickets_per_match', 
        color='phase',
        title="Average Wickets per Match by Phase",
        labels={'season': 'Season', 'wickets_per_match': 'Wickets per Match', 'phase': 'Phase'}
    )
    fig2.update_layout(xaxis_title="Season", yaxis_title="Wickets per Match")
    col2.plotly_chart(fig2, use_container_width=True)
    
    # Powerplay analysis
    col1, col2 = st.columns(2)
    
    powerplay_df = filtered_df[filtered_df['phase'] == 'Powerplay']
    
    # Powerplay scores
    powerplay_scores = powerplay_df.groupby(['season', 'match_id', 'batting_team'])['cumulative_runs'].max().reset_index()
    powerplay_avg = powerplay_scores.groupby(['season', 'batting_team'])['cumulative_runs'].mean().reset_index()
    
    fig3 = px.line(
        powerplay_avg, 
        x='season', 
        y='cumulative_runs', 
        color='batting_team',
        title="Average Powerplay Scores by Team",
        labels={'season': 'Season', 'cumulative_runs': 'Average Score', 'batting_team': 'Team'},
        markers=True
    )
    fig3.update_layout(xaxis_title="Season", yaxis_title="Average Powerplay Score")
    col1.plotly_chart(fig3, use_container_width=True)
    
    # Powerplay wickets
    powerplay_wickets = powerplay_df.groupby(['season', 'match_id', 'bowling_team'])['wicket_kind'].apply(lambda x: x.notna().sum()).reset_index(name='wickets')
    powerplay_wickets_avg = powerplay_wickets.groupby(['season', 'bowling_team'])['wickets'].mean().reset_index()
    
    fig4 = px.line(
        powerplay_wickets_avg, 
        x='season', 
        y='wickets', 
        color='bowling_team',
        title="Average Powerplay Wickets by Team",
        labels={'season': 'Season', 'wickets': 'Average Wickets', 'bowling_team': 'Team'},
        markers=True
    )
    fig4.update_layout(xaxis_title="Season", yaxis_title="Average Powerplay Wickets")
    col2.plotly_chart(fig4, use_container_width=True)

else:  # Seasonal Insights
    st.subheader("Seasonal Insights Analysis")
    
    # Create columns for layout
    col1, col2 = st.columns(2)
    
    # Total runs per season
    season_runs = filtered_df.groupby('season')['runs_total'].sum().reset_index()
    
    fig1 = px.bar(
        season_runs, 
        x='season', 
        y='runs_total',
        title="Total Runs per Season",
        labels={'season': 'Season', 'runs_total': 'Total Runs'}
    )
    fig1.update_layout(xaxis_title="Season", yaxis_title="Total Runs")
    col1.plotly_chart(fig1, use_container_width=True)
    
    # Average score per match
    season_avg_scores = filtered_df.groupby(['season', 'match_id', 'innings'])['cumulative_runs'].max().reset_index()
    season_avg = season_avg_scores.groupby('season')['cumulative_runs'].mean().reset_index()
    
    fig2 = px.line(
        season_avg, 
        x='season', 
        y='cumulative_runs',
        title="Average Innings Score per Season",
        labels={'season': 'Season', 'cumulative_runs': 'Average Score'},
        markers=True
    )
    fig2.update_layout(xaxis_title="Season", yaxis_title="Average Score")
    col2.plotly_chart(fig2, use_container_width=True)
    
    # Team evolution over seasons
    col1, col2 = st.columns(2)
    
    # Team performance over seasons
    team_season = filtered_df.drop_duplicates('match_id').groupby(['season', 'match_won_by']).size().reset_index(name='wins')
    team_season_pivot = team_season.pivot(index='season', columns='match_won_by', values='wins').fillna(0)
    
    # Get top 5 teams by total wins
    top_teams = team_season.groupby('match_won_by')['wins'].sum().nlargest(5).index
    team_season_top = team_season[team_season['match_won_by'].isin(top_teams)]
    
    fig3 = px.area(
        team_season_top, 
        x='season', 
        y='wins', 
        color='match_won_by',
        title="Top Teams' Performance Over Seasons",
        labels={'season': 'Season', 'wins': 'Wins', 'match_won_by': 'Team'}
    )
    fig3.update_layout(xaxis_title="Season", yaxis_title="Wins")
    col1.plotly_chart(fig3, use_container_width=True)
    
    # Dismissal patterns over seasons
    dismissal_types = filtered_df[filtered_df['wicket_kind'].notna()].groupby(['season', 'wicket_kind']).size().reset_index(name='count')
    
    fig4 = px.bar(
        dismissal_types, 
        x='season', 
        y='count', 
        color='wicket_kind',
        title="Dismissal Types Over Seasons",
        labels={'season': 'Season', 'count': 'Count', 'wicket_kind': 'Dismissal Type'}
    )
    fig4.update_layout(xaxis_title="Season", yaxis_title="Number of Dismissals")
    col2.plotly_chart(fig4, use_container_width=True)

# Data table
st.subheader("Data Sample")
st.dataframe(filtered_df.head(100))

# Footer
st.markdown("---")
st.markdown("IPL Data Analysis Dashboard | Built with Streamlit")
