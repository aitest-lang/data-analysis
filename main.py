import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import zipfile

with zipfile.ZipFile('archive.zip', 'r') as zip_ref:
    zip_ref.extractall()
# Import utility functions
from utils import *
from analysis import *

# Page configuration
st.set_page_config(
    page_title="IPL Data Analysis Platform", 
    page_icon="🏏", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .sidebar .sidebar-content {
        background-color: #fafafa;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess IPL data"""
    df = pd.read_csv("IPL.csv")
    df = preprocess_data(df)
    return df

def main():
    st.markdown('<h1 class="main-header">🏏 IPL Data Analysis Platform</h1>', unsafe_allow_html=True)
    
    # Load data
    try:
        df = load_data()
        st.success(f"✅ Data loaded successfully! Total records: {len(df):,}")
    except FileNotFoundError:
        st.error("❌ IPL.csv file not found. Please ensure the file is in the correct directory.")
        return
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        return
    
    # Sidebar for navigation and filters
    st.sidebar.title("🎛️ Navigation & Filters")
    
    # Navigation
    page = st.sidebar.selectbox(
        "Choose Analysis Module",
        ["🏠 Overview", "🏏 Team Performance", "👤 Player Analytics", "🏟️ Match Insights", "📈 Strategic Analysis"]
    )
    
    # Global filters
    st.sidebar.markdown("### 📊 Global Filters")
    
    seasons = sorted(df['season'].unique())
    selected_seasons = st.sidebar.multiselect("Select Seasons", seasons, default=seasons[-3:])
    
    teams = sorted(df['batting_team'].unique())
    selected_teams = st.sidebar.multiselect("Select Teams", teams, default=teams)
    
    # Filter data based on selections
    filtered_df = df[
        (df['season'].isin(selected_seasons)) &
        (df['batting_team'].isin(selected_teams))
    ]
    
    # Route to appropriate page
    if page == "🏠 Overview":
        show_overview(filtered_df)
    elif page == "🏏 Team Performance":
        show_team_performance(filtered_df)
    elif page == "👤 Player Analytics":
        show_player_analytics(filtered_df)
    elif page == "🏟️ Match Insights":
        show_match_insights(filtered_df)
    elif page == "📈 Strategic Analysis":
        show_strategic_analysis(filtered_df)

def show_overview(df):
    """Display overview dashboard"""
    st.markdown("## 🏠 IPL Overview Dashboard")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_matches = df['match_id'].nunique()
        st.metric("Total Matches", f"{total_matches:,}")
    
    with col2:
        total_runs = df['runs_total'].sum()
        st.metric("Total Runs", f"{total_runs:,}")
    
    with col3:
        total_wickets = df[df['wicket_kind'].notna()].shape[0]
        st.metric("Total Wickets", f"{total_wickets:,}")
    
    with col4:
        avg_score = df.groupby('match_id')['team_runs'].first().mean()
        st.metric("Avg Score", f"{avg_score:.1f}")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Runs by Season")
        season_runs = get_season_runs_trend(df)
        fig = px.line(season_runs, x='season', y='total_runs', 
                     title="Total Runs Scored by Season")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🏆 Matches by Team")
        team_matches = get_team_match_counts(df)
        fig = px.bar(team_matches.head(8), x='team', y='matches',
                    title="Matches Played (Top 8 Teams)")
        fig.update_xaxis(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

def show_team_performance(df):
    """Display team performance analysis"""
    st.markdown("## 🏏 Team Performance Analysis")
    
    # Team selector
    selected_team = st.selectbox("Select Team for Detailed Analysis", 
                                sorted(df['batting_team'].unique()))
    
    team_df = df[df['batting_team'] == selected_team]
    
    # Team metrics
    col1, col2, col3, col4 = st.columns(4)
    
    team_stats = get_team_stats(df, selected_team)
    
    with col1:
        st.metric("Matches Played", team_stats['matches'])
    with col2:
        st.metric("Win Rate", f"{team_stats['win_rate']:.1f}%")
    with col3:
        st.metric("Avg Score", f"{team_stats['avg_runs']:.1f}")
    with col4:
        st.metric("Strike Rate", f"{team_stats['strike_rate']:.1f}")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Toss Impact")
        toss_impact = get_toss_impact(df, selected_team)
        fig = px.bar(toss_impact, x='toss_won', y='win_percentage',
                    title=f"{selected_team} - Win % by Toss Result")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🏟️ Venue Performance")
        venue_perf = get_venue_performance(df, selected_team)
        fig = px.scatter(venue_perf, x='matches', y='win_rate', 
                        hover_data=['venue'],
                        title=f"{selected_team} - Win Rate by Venue")
        st.plotly_chart(fig, use_container_width=True)

def show_player_analytics(df):
    """Display player analytics"""
    st.markdown("## 👤 Player Analytics")
    
    tab1, tab2 = st.tabs(["🏏 Batting", "⚾ Bowling"])
    
    with tab1:
        st.markdown("### 🏏 Top Batsmen")
        
        # Batting leaderboard
        batting_stats = get_batting_leaderboard(df)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.dataframe(batting_stats.head(15), use_container_width=True)
        
        with col2:
            # Top scorers chart
            fig = px.bar(batting_stats.head(10), x='total_runs', y='batter',
                        orientation='h', title="Top 10 Run Scorers")
            fig.update_yaxis(categoryorder="total ascending")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### ⚾ Top Bowlers")
        
        # Bowling leaderboard
        bowling_stats = get_bowling_leaderboard(df)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.dataframe(bowling_stats.head(15), use_container_width=True)
        
        with col2:
            # Top wicket takers chart
            fig = px.bar(bowling_stats.head(10), x='total_wickets', y='bowler',
                        orientation='h', title="Top 10 Wicket Takers")
            fig.update_yaxis(categoryorder="total ascending")
            st.plotly_chart(fig, use_container_width=True)

def show_match_insights(df):
    """Display match insights"""
    st.markdown("## 🏟️ Match Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎲 Toss vs Match Result")
        toss_analysis = get_overall_toss_impact(df)
        
        fig = px.pie(toss_analysis, values='matches', names='result',
                    title="Overall Toss Impact on Match Results")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🏟️ Venue Analysis")
        venue_stats = get_venue_analysis(df)
        
        fig = px.bar(venue_stats.head(10), x='venue', y='avg_runs',
                    title="Average Runs by Venue (Top 10)")
        fig.update_xaxis(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Season trends
    st.markdown("### 📅 Season Trends")
    season_trends = get_season_trends(df)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(x=season_trends['season'], y=season_trends['avg_runs'],
                  name="Avg Runs", mode='lines+markers'),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(x=season_trends['season'], y=season_trends['boundaries_per_match'],
                  name="Boundaries/Match", mode='lines+markers'),
        secondary_y=True,
    )
    
    fig.update_xaxes(title_text="Season")
    fig.update_yaxes(title_text="Average Runs", secondary_y=False)
    fig.update_yaxes(title_text="Boundaries per Match", secondary_y=True)
    fig.update_layout(title="Season-wise Scoring Trends")
    
    st.plotly_chart(fig, use_container_width=True)

def show_strategic_analysis(df):
    """Display strategic analysis"""
    st.markdown("## 📈 Strategic Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### ⚡ Powerplay Analysis")
        powerplay_stats = get_powerplay_analysis(df)
        
        fig = px.bar(powerplay_stats, x='team', y='pp_runs_per_match',
                    title="Powerplay Runs per Match by Team")
        fig.update_xaxis(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 💀 Death Overs Performance")
        death_stats = get_death_overs_analysis(df)
        
        fig = px.scatter(death_stats, x='death_runs_per_match', y='death_wickets_per_match',
                        hover_data=['team'], title="Death Overs: Runs vs Wickets")
        st.plotly_chart(fig, use_container_width=True)
    
    # Partnership analysis
    st.markdown("### 🤝 Partnership Analysis")
    partnership_stats = get_partnership_analysis(df)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(partnership_stats.head(10), use_container_width=True)
    
    with col2:
        fig = px.histogram(partnership_stats, x='avg_partnership',
                          title="Distribution of Average Partnerships")
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
