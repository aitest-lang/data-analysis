import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import zipfile
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="IPL Data Analysis Dashboard",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data
@st.cache_data
def load_data():
    # Replace 'your_data_file.csv' with your actual file path
    df = pd.read_csv('IPL.csv')  # Update with your file path
    return df

def main():
    st.title("🏏 IPL Data Analysis Dashboard")
    st.markdown("---")
    with zipfile.ZipFile('archive.zip', 'r') as zip_ref:
        zip_ref.extractall()
    # Load data
    try:
        df = load_data()
        st.success("Data loaded successfully!")
    except:
        st.error("Please upload your IPL data file or update the file path in the code")
        return
    
    # Sidebar for analysis selection
    st.sidebar.title("Analysis Options")
    analysis_category = st.sidebar.selectbox(
        "Select Analysis Category:",
        [
            "Overview",
            "Match Analysis",
            "Player Performance",
            "Team Performance", 
            "Bowling Analysis",
            "Batting Analysis",
            "Wickets Analysis",
            "Venue Analysis",
            "Toss Analysis",
            "Season Analysis"
        ]
    )
    
    # Data preprocessing
    df['date'] = pd.to_datetime(df['date'])
    
    if analysis_category == "Overview":
        show_overview(df)
    elif analysis_category == "Match Analysis":
        show_match_analysis(df)
    elif analysis_category == "Player Performance":
        show_player_performance(df)
    elif analysis_category == "Team Performance":
        show_team_performance(df)
    elif analysis_category == "Bowling Analysis":
        show_bowling_analysis(df)
    elif analysis_category == "Batting Analysis":
        show_batting_analysis(df)
    elif analysis_category == "Wickets Analysis":
        show_wickets_analysis(df)
    elif analysis_category == "Venue Analysis":
        show_venue_analysis(df)
    elif analysis_category == "Toss Analysis":
        show_toss_analysis(df)
    elif analysis_category == "Season Analysis":
        show_season_analysis(df)

def show_overview(df):
    st.header("📊 IPL Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Matches", df['match_id'].nunique())
    with col2:
        st.metric("Total Players", df['batter'].nunique())
    with col3:
        st.metric("Total Teams", df['batting_team'].nunique())
    with col4:
        st.metric("Total Seasons", df['season'].nunique())
    
    # Basic info
    st.subheader("Dataset Information")
    st.write(f"Total Records: {len(df)}")
    st.write(f"Columns: {len(df.columns)}")
    st.write(f"Date Range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
    
    # Sample data
    st.subheader("Sample Data")
    st.dataframe(df.head())

def show_match_analysis(df):
    st.header("🏆 Match Analysis")
    
    # Match type distribution
    col1, col2 = st.columns(2)
    
    with col1:
        match_type_dist = df['match_type'].value_counts()
        fig1 = px.pie(values=match_type_dist.values, names=match_type_dist.index, title="Match Type Distribution")
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        season_dist = df['season'].value_counts().sort_index()
        fig2 = px.bar(x=season_dist.index, y=season_dist.values, title="Matches per Season")
        st.plotly_chart(fig2, use_container_width=True)
    
    # Win outcomes
    st.subheader("Win Outcomes Distribution")
    win_outcomes = df['win_outcome'].value_counts()
    fig3 = px.bar(x=win_outcomes.index, y=win_outcomes.values, title="Win Outcomes")
    st.plotly_chart(fig3, use_container_width=True)
    
    # Top venues
    st.subheader("Top Venues by Matches")
    venue_counts = df['venue'].value_counts().head(10)
    fig4 = px.bar(x=venue_counts.values, y=venue_counts.index, orientation='h', title="Top 10 Venues")
    st.plotly_chart(fig4, use_container_width=True)

def show_player_performance(df):
    st.header("👤 Player Performance Analysis")
    
    # Top scorers
    st.subheader("Top Batters")
    batter_stats = df.groupby('batter').agg({
        'runs_batter': 'sum',
        'balls_faced': 'sum',
        'match_id': 'nunique'
    }).reset_index()
    batter_stats['strike_rate'] = (batter_stats['runs_batter'] / batter_stats['balls_faced']) * 100
    top_batters = batter_stats.sort_values('runs_batter', ascending=False).head(10)
    
    fig1 = px.bar(top_batters, x='batter', y='runs_batter', title="Top 10 Batters by Runs")
    st.plotly_chart(fig1, use_container_width=True)
    
    # Top wicket takers
    st.subheader("Top Bowlers")
    bowler_stats = df.groupby('bowler').agg({
        'bowler_wicket': 'sum',
        'runs_bowler': 'sum',
        'match_id': 'nunique'
    }).reset_index()
    bowler_stats['economy'] = bowler_stats['runs_bowler'] / (bowler_stats['match_id'] * 20)  # Approximate
    top_bowlers = bowler_stats.sort_values('bowler_wicket', ascending=False).head(10)
    
    fig2 = px.bar(top_bowlers, x='bowler', y='bowler_wicket', title="Top 10 Bowlers by Wickets")
    st.plotly_chart(fig2, use_container_width=True)
    
    # Player of the match
    st.subheader("Most Player of the Match Awards")
    potm = df['player_of_match'].value_counts().head(10)
    fig3 = px.bar(x=potm.values, y=potm.index, orientation='h', title="Top 10 Players of the Match")
    st.plotly_chart(fig3, use_container_width=True)

def show_team_performance(df):
    st.header("👥 Team Performance Analysis")
    
    # Team wins
    col1, col2 = st.columns(2)
    
    with col1:
        team_wins = df['match_won_by'].value_counts()
        fig1 = px.bar(x=team_wins.values, y=team_wins.index, orientation='h', title="Team Wins")
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        toss_wins = df['toss_winner'].value_counts()
        fig2 = px.pie(values=toss_wins.values, names=toss_wins.index, title="Toss Wins Distribution")
        st.plotly_chart(fig2, use_container_width=True)
    
    # Team vs Team performance
    st.subheader("Team vs Team Performance")
    team_performance = df.groupby(['batting_team', 'bowling_team', 'match_won_by']).size().reset_index(name='count')
    team_performance['winner'] = team_performance.apply(
        lambda x: x['batting_team'] if x['match_won_by'] == x['batting_team'] else x['bowling_team'], axis=1
    )
    
    fig3 = px.scatter(team_performance, x='batting_team', y='bowling_team', size='count', 
                     color='winner', title="Team vs Team Performance Matrix")
    st.plotly_chart(fig3, use_container_width=True)

def show_bowling_analysis(df):
    st.header("⚾ Bowling Analysis")
    
    # Bowling stats
    bowler_summary = df.groupby('bowler').agg({
        'runs_bowler': 'sum',
        'bowler_wicket': 'sum',
        'valid_ball': 'sum'
    }).reset_index()
    bowler_summary['overs_bowled'] = bowler_summary['valid_ball'] / 6
    bowler_summary['economy'] = bowler_summary['runs_bowler'] / bowler_summary['overs_bowled']
    bowler_summary['wickets_per_match'] = bowler_summary['bowler_wicket'] / df.groupby('bowler')['match_id'].nunique()
    
    col1, col2 = st.columns(2)
    
    with col1:
        top_economical = bowler_summary.nlargest(10, 'bowler_wicket').sort_values('economy')
        fig1 = px.bar(top_economical, x='bowler', y='economy', title="Top Wicket Takers Economy Rate")
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        bowler_wickets = bowler_summary.sort_values('bowler_wicket', ascending=False).head(10)
        fig2 = px.bar(bowler_wickets, x='bowler', y='bowler_wicket', title="Top Wicket Takers")
        st.plotly_chart(fig2, use_container_width=True)
    
    # Extras analysis
    st.subheader("Extras Analysis")
    extra_types = df[df['runs_extras'] > 0]['extra_type'].value_counts()
    fig3 = px.pie(values=extra_types.values, names=extra_types.index, title="Types of Extras")
    st.plotly_chart(fig3, use_container_width=True)

def show_batting_analysis(df):
    st.header("🏏 Batting Analysis")
    
    # Batting stats
    batter_summary = df.groupby('batter').agg({
        'runs_batter': 'sum',
        'balls_faced': 'sum',
        'match_id': 'nunique'
    }).reset_index()
    batter_summary['strike_rate'] = (batter_summary['runs_batter'] / batter_summary['balls_faced']) * 100
    batter_summary['average'] = batter_summary['runs_batter'] / batter_summary['match_id']
    
    col1, col2 = st.columns(2)
    
    with col1:
        top_strike_rate = batter_summary.nlargest(10, 'strike_rate')
        fig1 = px.bar(top_strike_rate, x='batter', y='strike_rate', title="Top Batters by Strike Rate")
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        top_average = batter_summary.nlargest(10, 'average')
        fig2 = px.bar(top_average, x='batter', y='average', title="Top Batters by Average")
        st.plotly_chart(fig2, use_container_width=True)
    
    # Runs distribution
    st.subheader("Runs Distribution by Innings")
    runs_by_innings = df.groupby('innings')['runs_total'].sum().reset_index()
    fig3 = px.bar(runs_by_innings, x='innings', y='runs_total', title="Total Runs by Innings")
    st.plotly_chart(fig3, use_container_width=True)
    
    # Boundaries analysis
    st.subheader("Boundaries Analysis")
    boundaries = df[df['runs_not_boundary'] == False].groupby('batter').size().reset_index(name='boundaries')
    top_boundaries = boundaries.nlargest(10, 'boundaries')
    fig4 = px.bar(top_boundaries, x='batter', y='boundaries', title="Top Batters by Boundaries")
    st.plotly_chart(fig4, use_container_width=True)

def show_wickets_analysis(df):
    st.header("🚫 Wickets Analysis")
    
    # Wicket types
    wicket_types = df['wicket_kind'].value_counts()
    fig1 = px.pie(values=wicket_types.values, names=wicket_types.index, title="Wicket Types Distribution")
    st.plotly_chart(fig1, use_container_width=True)
    
    # Wickets by innings
    wickets_by_innings = df[df['wicket_kind'].notna()].groupby('innings').size().reset_index(name='wickets')
    fig2 = px.bar(wickets_by_innings, x='innings', y='wickets', title="Wickets by Innings")
    st.plotly_chart(fig2, use_container_width=True)
    
    # Most dismissed players
    st.subheader("Most Dismissed Players")
    most_dismissed = df[df['wicket_kind'].notna()]['player_out'].value_counts().head(10)
    fig3 = px.bar(x=most_dismissed.values, y=most_dismissed.index, orientation='h', 
                  title="Top 10 Most Dismissed Players")
    st.plotly_chart(fig3, use_container_width=True)

def show_venue_analysis(df):
    st.header("🏟️ Venue Analysis")
    
    # Most matches at venues
    venue_matches = df['venue'].value_counts().head(10)
    fig1 = px.bar(x=venue_matches.values, y=venue_matches.index, orientation='h', 
                  title="Top 10 Venues by Matches")
    st.plotly_chart(fig1, use_container_width=True)
    
    # Average runs per venue
    venue_runs = df.groupby('venue')['runs_total'].mean().sort_values(ascending=False).head(10)
    fig2 = px.bar(x=venue_runs.values, y=venue_runs.index, orientation='h', 
                  title="Top 10 Venues by Average Runs per Match")
    st.plotly_chart(fig2, use_container_width=True)
    
    # Venue wins
    venue_wins = df.groupby(['venue', 'match_won_by']).size().reset_index(name='wins')
    fig3 = px.scatter(venue_wins, x='venue', y='match_won_by', size='wins', 
                      title="Venue vs Team Wins")
    st.plotly_chart(fig3, use_container_width=True)

def show_toss_analysis(df):
    st.header("🪙 Toss Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        toss_decisions = df['toss_decision'].value_counts()
        fig1 = px.pie(values=toss_decisions.values, names=toss_decisions.index, title="Toss Decisions")
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        toss_win_match_win = df[df['toss_winner'] == df['match_won_by']]
        toss_win_percentage = (len(toss_win_match_win) / len(df)) * 100
        st.metric("Toss Winner Wins Match (%)", f"{toss_win_percentage:.2f}%")
        
        # Toss win vs match win
        toss_match_result = pd.DataFrame({
            'Result': ['Toss Winner Wins', 'Toss Winner Loses'],
            'Count': [len(toss_win_match_win), len(df) - len(toss_win_match_win)]
        })
        fig2 = px.bar(toss_match_result, x='Result', y='Count', title="Toss Win vs Match Win")
        st.plotly_chart(fig2, use_container_width=True)
    
    # Toss decision impact
    st.subheader("Toss Decision Impact on Match Results")
    toss_impact = df.groupby(['toss_decision', 'match_won_by']).size().reset_index(name='count')
    fig3 = px.bar(toss_impact, x='toss_decision', y='count', color='match_won_by', 
                  title="Toss Decision vs Match Winner")
    st.plotly_chart(fig3, use_container_width=True)

def show_season_analysis(df):
    st.header("📅 Season Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        matches_per_season = df.groupby('season')['match_id'].nunique()
        fig1 = px.line(x=matches_per_season.index, y=matches_per_season.values, 
                       title="Matches per Season")
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        runs_per_season = df.groupby('season')['runs_total'].sum()
        fig2 = px.line(x=runs_per_season.index, y=runs_per_season.values, 
                       title="Total Runs per Season")
        st.plotly_chart(fig2, use_container_width=True)
    
    # Season wise team performance
    st.subheader("Season Wise Team Performance")
    season_team_wins = df.groupby(['season', 'match_won_by']).size().reset_index(name='wins')
    fig3 = px.bar(season_team_wins, x='season', y='wins', color='match_won_by', 
                  title="Team Wins per Season")
    st.plotly_chart(fig3, use_container_width=True)
    
    # Average runs per match by season
    st.subheader("Average Runs per Match by Season")
    avg_runs_per_match = df.groupby('season')['runs_total'].mean()
    fig4 = px.bar(x=avg_runs_per_match.index, y=avg_runs_per_match.values, 
                  title="Average Runs per Match by Season")
    st.plotly_chart(fig4, use_container_width=True)

if __name__ == "__main__":
    main()
