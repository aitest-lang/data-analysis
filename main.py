import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(
    page_title="IPL Data Analytics Dashboard",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)
with zipfile.ZipFile('archive.zip', 'r') as zip_ref:
    zip_ref.extractall()
# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_clean_data():
    """Load and clean IPL data"""
    try:
        df = pd.read_csv("IPL.csv")
        
        # Fix team name inconsistencies
        df['batting_team'] = df['batting_team'].replace('Royal Challengers Bangalore', 'Royal Challengers Bengaluru')
        df['bowling_team'] = df['bowling_team'].replace('Royal Challengers Bangalore', 'Royal Challengers Bengaluru')
        df['match_won_by'] = df['match_won_by'].replace('Royal Challengers Bangalore', 'Royal Challengers Bengaluru')
        df['toss_winner'] = df['toss_winner'].replace('Royal Challengers Bangalore', 'Royal Challengers Bengaluru')
        
        # Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Fill missing values
        df['extra_type'] = df['extra_type'].fillna('None')
        df['wicket_kind'] = df['wicket_kind'].fillna('Not Out')
        df['player_out'] = df['player_out'].fillna('None')
        
        # Create derived columns
        df['match_year'] = df['date'].dt.year
        df['is_boundary'] = ~df['runs_not_boundary']
        df['is_six'] = df['runs_batter'] == 6
        df['is_four'] = df['runs_batter'] == 4
        df['is_wicket'] = df['wicket_kind'] != 'Not Out'
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

def get_team_performance(df):
    """Calculate team performance metrics"""
    team_stats = df.groupby(['batting_team', 'match_id']).agg({
        'runs_total': 'sum',
        'is_wicket': 'sum',
        'is_boundary': 'sum',
        'is_six': 'sum',
        'is_four': 'sum'
    }).reset_index()
    
    team_summary = team_stats.groupby('batting_team').agg({
        'runs_total': ['mean', 'sum', 'max'],
        'is_wicket': 'mean',
        'is_boundary': 'mean',
        'is_six': 'sum',
        'is_four': 'sum',
        'match_id': 'count'
    }).round(2)
    
    team_summary.columns = ['Avg_Score', 'Total_Runs', 'Highest_Score', 
                          'Avg_Wickets', 'Boundaries_Per_Match', 'Total_Sixes', 
                          'Total_Fours', 'Matches_Played']
    
    return team_summary.reset_index()

def get_player_stats(df):
    """Calculate player batting statistics"""
    player_stats = df.groupby('batter').agg({
        'runs_batter': ['sum', 'mean', 'max'],
        'balls_faced': 'sum',
        'is_six': 'sum',
        'is_four': 'sum',
        'match_id': 'nunique'
    }).round(2)
    
    player_stats.columns = ['Total_Runs', 'Avg_Runs', 'Highest_Score', 
                          'Balls_Faced', 'Sixes', 'Fours', 'Matches']
    
    # Calculate Strike Rate
    player_stats['Strike_Rate'] = (player_stats['Total_Runs'] / player_stats['Balls_Faced'] * 100).round(2)
    
    # Filter players with minimum 100 runs
    player_stats = player_stats[player_stats['Total_Runs'] >= 100]
    
    return player_stats.reset_index()

def get_venue_analysis(df):
    """Analyze venue statistics"""
    venue_stats = df.groupby(['venue', 'match_id']).agg({
        'runs_total': 'sum'
    }).reset_index()
    
    venue_summary = venue_stats.groupby('venue').agg({
        'runs_total': ['mean', 'count', 'max', 'min']
    }).round(2)
    
    venue_summary.columns = ['Avg_Score', 'Matches_Played', 'Highest_Score', 'Lowest_Score']
    
    return venue_summary.reset_index()

def main():
    # Header
    st.markdown('<h1 class="main-header">🏏 IPL Data Analytics Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Load data
    df = load_and_clean_data()
    
    if df is None:
        st.error("Failed to load data. Please ensure IPL.csv is in the correct path.")
        return
    
    # Sidebar
    st.sidebar.header("📊 Analysis Categories")
    
    analysis_type = st.sidebar.selectbox(
        "Select Analysis Type:",
        ["Overview", "Team Performance", "Player Statistics", 
         "Venue Analysis", "Season Trends", "Match Insights"]
    )
    
    # Overview Section
    if analysis_type == "Overview":
        st.header("📈 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Matches", df['match_id'].nunique())
        with col2:
            st.metric("Total Teams", df['batting_team'].nunique())
        with col3:
            st.metric("Total Players", df['batter'].nunique())
        with col4:
            st.metric("Total Venues", df['venue'].nunique())
        
        st.subheader("📋 Data Sample")
        st.dataframe(df.head(10))
        
        st.subheader("🔍 Data Info")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Dataset Shape:**", df.shape)
            st.write("**Date Range:**", f"{df['date'].min().date()} to {df['date'].max().date()}")
        
        with col2:
            st.write("**Missing Values:**")
            missing_data = df.isnull().sum()
            st.write(missing_data[missing_data > 0])
    
    # Team Performance
    elif analysis_type == "Team Performance":
        st.header("🏆 Team Performance Analysis")
        
        team_stats = get_team_performance(df)
        
        # Team selector
        selected_teams = st.multiselect(
            "Select Teams (leave empty for all):",
            options=team_stats['batting_team'].tolist(),
            default=team_stats['batting_team'].tolist()[:8]
        )
        
        if selected_teams:
            filtered_stats = team_stats[team_stats['batting_team'].isin(selected_teams)]
        else:
            filtered_stats = team_stats
        
        # Display team statistics table
        st.subheader("📊 Team Statistics")
        st.dataframe(filtered_stats)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(filtered_stats, x='batting_team', y='Avg_Score',
                        title='Average Score by Team',
                        color='Avg_Score',
                        color_continuous_scale='viridis')
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(filtered_stats, x='Matches_Played', y='Highest_Score',
                           size='Total_Runs', color='batting_team',
                           title='Highest Score vs Matches Played',
                           hover_data=['Avg_Score'])
            st.plotly_chart(fig, use_container_width=True)
    
    # Player Statistics
    elif analysis_type == "Player Statistics":
        st.header("👨‍💼 Player Statistics")
        
        player_stats = get_player_stats(df)
        
        # Top players
        metric_options = ['Total_Runs', 'Avg_Runs', 'Strike_Rate', 'Sixes', 'Fours']
        selected_metric = st.selectbox("Select Metric:", metric_options)
        
        top_n = st.slider("Number of top players to display:", 5, 20, 10)
        
        top_players = player_stats.nlargest(top_n, selected_metric)
        
        st.subheader(f"🏅 Top {top_n} Players by {selected_metric}")
        st.dataframe(top_players)
        
        # Visualization
        fig = px.bar(top_players, x='batter', y=selected_metric,
                    title=f'Top {top_n} Players by {selected_metric}',
                    color=selected_metric,
                    color_continuous_scale='plasma')
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Strike Rate vs Average
        if len(player_stats) > 0:
            fig = px.scatter(player_stats, x='Avg_Runs', y='Strike_Rate',
                           size='Total_Runs', color='Matches',
                           hover_data=['batter', 'Sixes', 'Fours'],
                           title='Strike Rate vs Average (Bubble size = Total Runs)')
            st.plotly_chart(fig, use_container_width=True)
    
    # Venue Analysis
    elif analysis_type == "Venue Analysis":
        st.header("🏟️ Venue Analysis")
        
        venue_stats = get_venue_analysis(df)
        
        st.subheader("📍 Venue Statistics")
        st.dataframe(venue_stats)
        
        # Top venues by average score
        top_venues = venue_stats.nlargest(10, 'Avg_Score')
        
        fig = px.bar(top_venues, x='venue', y='Avg_Score',
                    title='Top 10 Venues by Average Score',
                    color='Matches_Played',
                    color_continuous_scale='blues')
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
        
        # Matches played at each venue
        fig = px.pie(venue_stats.nlargest(8, 'Matches_Played'), 
                    values='Matches_Played', names='venue',
                    title='Distribution of Matches by Venue (Top 8)')
        st.plotly_chart(fig, use_container_width=True)
    
    # Season Trends
    elif analysis_type == "Season Trends":
        st.header("📅 Season Trends")
        
        # Runs per season
        season_runs = df.groupby(['season', 'match_id'])['runs_total'].sum().reset_index()
        season_avg = season_runs.groupby('season')['runs_total'].mean().reset_index()
        
        fig = px.line(season_avg, x='season', y='runs_total',
                     title='Average Runs per Match by Season',
                     markers=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # Boundaries trend
        boundaries_trend = df.groupby('season').agg({
            'is_six': 'sum',
            'is_four': 'sum',
            'match_id': 'nunique'
        }).reset_index()
        
        boundaries_trend['Sixes_per_match'] = boundaries_trend['is_six'] / boundaries_trend['match_id']
        boundaries_trend['Fours_per_match'] = boundaries_trend['is_four'] / boundaries_trend['match_id']
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(x=boundaries_trend['season'], y=boundaries_trend['Sixes_per_match'],
                      name='Sixes per Match', mode='lines+markers'),
            secondary_y=False
        )
        fig.add_trace(
            go.Scatter(x=boundaries_trend['season'], y=boundaries_trend['Fours_per_match'],
                      name='Fours per Match', mode='lines+markers'),
            secondary_y=True
        )
        fig.update_xaxes(title_text="Season")
        fig.update_yaxes(title_text="Sixes per Match", secondary_y=False)
        fig.update_yaxes(title_text="Fours per Match", secondary_y=True)
        fig.update_layout(title_text="Boundaries Trend by Season")
        st.plotly_chart(fig, use_container_width=True)
    
    # Match Insights
    elif analysis_type == "Match Insights":
        st.header("⚾ Match Insights")
        
        # Toss analysis
        toss_impact = df.groupby(['match_id', 'toss_winner', 'match_won_by']).size().reset_index()
        toss_impact['toss_match'] = toss_impact['toss_winner'] == toss_impact['match_won_by']
        toss_win_rate = toss_impact['toss_match'].mean() * 100
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Toss Winner Match Win %", f"{toss_win_rate:.1f}%")
        
        # Win by runs vs wickets
        win_methods = df[df['match_won_by'].notna()].groupby('win_outcome').size()
        
        fig = px.pie(values=win_methods.values, names=win_methods.index,
                    title='Match Win Methods Distribution')
        st.plotly_chart(fig, use_container_width=True)
        
        # Most successful teams
        wins_by_team = df[df['match_won_by'].notna()]['match_won_by'].value_counts()
        
        fig = px.bar(x=wins_by_team.index[:10], y=wins_by_team.values[:10],
                    title='Most Successful Teams (Total Wins)',
                    labels={'x': 'Team', 'y': 'Wins'})
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
