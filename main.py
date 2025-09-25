import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import zipfile

with zipfile.ZipFile('archive.zip', 'r') as zip_ref:
    zip_ref.extractall()
# Set page configuration
st.set_page_config(
    page_title="IPL Data Analytics Dashboard",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #A23B72;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .sidebar-title {
        font-size: 1.2rem;
        font-weight: bold;
        color: #2E86AB;
    }
</style>
""", unsafe_allow_html=True)

def load_data():
    """Load the IPL data - Replace with your actual file path"""
    # For demonstration, I'll create a sample data loading function
    # In real scenario, replace this with: df = pd.read_csv('your_ipl_data.csv')
    
    # Since I don't have the actual file, I'll create a sample structure
    # You should replace this with your actual data loading code
    try:
        df = pd.read_csv('IPL.csv')  # Replace with your file path
    except:
        # Sample data creation for demonstration
        np.random.seed(42)
        n_rows = 1000
        df = pd.DataFrame({
            'match_id': np.random.randint(1, 100, n_rows),
            'date': pd.date_range('2020-01-01', periods=n_rows, freq='D'),
            'match_type': np.random.choice(['T20', 'ODI'], n_rows),
            'event_name': np.random.choice(['IPL 2020', 'IPL 2021', 'IPL 2022'], n_rows),
            'innings': np.random.choice([1, 2], n_rows),
            'batting_team': np.random.choice(['MI', 'CSK', 'RCB', 'KKR', 'SRH', 'RR', 'DC', 'PBKS'], n_rows),
            'bowling_team': np.random.choice(['MI', 'CSK', 'RCB', 'KKR', 'SRH', 'RR', 'DC', 'PBKS'], n_rows),
            'over': np.random.randint(0, 20, n_rows),
            'ball': np.random.randint(1, 7, n_rows),
            'batter': np.random.choice(['Player1', 'Player2', 'Player3', 'Player4', 'Player5'], n_rows),
            'runs_batter': np.random.randint(0, 7, n_rows),
            'bowler': np.random.choice(['Bowler1', 'Bowler2', 'Bowler3', 'Bowler4'], n_rows),
            'runs_total': np.random.randint(0, 10, n_rows),
            'wicket_kind': np.random.choice(['caught', 'bowled', 'lbw', 'run out', 'stumped', None], n_rows),
            'player_out': np.random.choice(['Player1', 'Player2', 'Player3', 'Player4', 'Player5', None], n_rows),
            'venue': np.random.choice(['Wankhede', 'Chinnaswamy', 'Eden Gardens', 'MCA'], n_rows),
            'city': np.random.choice(['Mumbai', 'Bangalore', 'Kolkata', 'Pune'], n_rows),
            'year': np.random.randint(2020, 2023, n_rows),
            'season': np.random.choice(['IPL 2020', 'IPL 2021', 'IPL 2022'], n_rows),
            'team_runs': np.random.randint(100, 200, n_rows),
            'team_wicket': np.random.randint(0, 10, n_rows),
        })
        # Ensure some teams don't bowl against themselves
        df.loc[df['batting_team'] == df['bowling_team'], 'bowling_team'] = 'PBKS'
    
    return df

def main():
    st.markdown('<h1 class="main-header">🏏 IPL Data Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    
    # Sidebar for analysis selection
    st.sidebar.markdown('<h2 class="sidebar-title">Analysis Categories</h2>', unsafe_allow_html=True)
    
    analysis_category = st.sidebar.selectbox(
        "Select Analysis Category:",
        ["Overview", "Team Analysis", "Player Analysis", "Match Analysis", "Venue Analysis", "Toss Analysis", "Wicket Analysis"]
    )
    
    # Additional filters in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Filters:**")
    
    seasons = ['All'] + sorted(df['season'].dropna().unique().tolist())
    selected_season = st.sidebar.selectbox("Select Season:", seasons)
    
    if selected_season != 'All':
        df = df[df['season'] == selected_season]
    
    teams = ['All'] + sorted(df['batting_team'].dropna().unique().tolist())
    selected_team = st.sidebar.selectbox("Select Team:", teams)
    
    if selected_team != 'All':
        df = df[(df['batting_team'] == selected_team) | (df['bowling_team'] == selected_team)]
    
    # Main content based on selected category
    if analysis_category == "Overview":
        show_overview(df)
    elif analysis_category == "Team Analysis":
        show_team_analysis(df)
    elif analysis_category == "Player Analysis":
        show_player_analysis(df)
    elif analysis_category == "Match Analysis":
        show_match_analysis(df)
    elif analysis_category == "Venue Analysis":
        show_venue_analysis(df)
    elif analysis_category == "Toss Analysis":
        show_toss_analysis(df)
    elif analysis_category == "Wicket Analysis":
        show_wicket_analysis(df)

def show_overview(df):
    st.markdown('<h2 class="sub-header">📊 IPL Overview</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Matches", len(df['match_id'].unique()))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Teams", len(df['batting_team'].unique()))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Players", len(df['batter'].unique()))
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Seasons", len(df['season'].unique()))
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Match distribution over seasons
    st.subheader("Match Distribution Over Seasons")
    season_counts = df['season'].value_counts().reset_index()
    season_counts.columns = ['Season', 'Matches']
    
    fig = px.bar(season_counts, x='Season', y='Matches', title="Matches per Season")
    st.plotly_chart(fig, use_container_width=True)
    
    # Runs distribution
    st.subheader("Runs Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df['runs_total'], bins=30, edgecolor='black')
    ax.set_xlabel('Runs')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Runs per Ball')
    st.pyplot(fig)
    
    # Top batting teams
    st.subheader("Top Batting Teams")
    team_runs = df.groupby('batting_team')['runs_total'].sum().sort_values(ascending=False).head(10)
    fig = px.bar(x=team_runs.index, y=team_runs.values, title="Total Runs by Team")
    fig.update_layout(xaxis_title="Team", yaxis_title="Total Runs")
    st.plotly_chart(fig, use_container_width=True)

def show_team_analysis(df):
    st.markdown('<h2 class="sub-header">🏆 Team Analysis</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Team performance by runs
        team_runs = df.groupby('batting_team')['runs_total'].sum().sort_values(ascending=False)
        fig = px.bar(x=team_runs.index, y=team_runs.values, title="Total Runs by Team")
        fig.update_layout(xaxis_title="Team", yaxis_title="Total Runs")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Team performance by wickets
        team_wickets = df[df['wicket_kind'].notna()].groupby('bowling_team').size().sort_values(ascending=False)
        fig = px.bar(x=team_wickets.index, y=team_wickets.values, title="Wickets Taken by Team")
        fig.update_layout(xaxis_title="Team", yaxis_title="Wickets")
        st.plotly_chart(fig, use_container_width=True)
    
    # Team vs Team comparison
    st.subheader("Team vs Team Performance")
    teams = df['batting_team'].unique()
    team1 = st.selectbox("Select Team 1:", teams, key='team1')
    team2 = st.selectbox("Select Team 2:", [t for t in teams if t != team1], key='team2')
    
    team1_runs = df[df['batting_team'] == team1]['runs_total'].sum()
    team2_runs = df[df['batting_team'] == team2]['runs_total'].sum()
    
    comparison_df = pd.DataFrame({
        'Team': [team1, team2],
        'Total Runs': [team1_runs, team2_runs]
    })
    
    fig = px.bar(comparison_df, x='Team', y='Total Runs', title=f"{team1} vs {team2} - Total Runs")
    st.plotly_chart(fig, use_container_width=True)

def show_player_analysis(df):
    st.markdown('<h2 class="sub-header">👨‍🦱 Player Analysis</h2>', unsafe_allow_html=True)
    
    # Top run scorers
    st.subheader("Top Run Scorers")
    player_runs = df.groupby('batter')['runs_batter'].sum().sort_values(ascending=False).head(10)
    fig = px.bar(x=player_runs.index, y=player_runs.values, title="Top 10 Run Scorers")
    fig.update_layout(xaxis_title="Player", yaxis_title="Total Runs")
    st.plotly_chart(fig, use_container_width=True)
    
    # Top wicket takers
    st.subheader("Top Wicket Takers")
    wicket_data = df[df['wicket_kind'].notna()]
    if not wicket_data.empty:
        bowler_wickets = wicket_data.groupby('bowler').size().sort_values(ascending=False).head(10)
        fig = px.bar(x=bowler_wickets.index, y=bowler_wickets.values, title="Top 10 Wicket Takers")
        fig.update_layout(xaxis_title="Bowler", yaxis_title="Wickets")
        st.plotly_chart(fig, use_container_width=True)
    
    # Player performance by team
    st.subheader("Player Performance by Team")
    selected_player = st.selectbox("Select Player:", df['batter'].unique())
    player_data = df[df['batter'] == selected_player]
    player_team_runs = player_data.groupby('batting_team')['runs_batter'].sum().sort_values(ascending=False)
    
    fig = px.bar(x=player_team_runs.index, y=player_team_runs.values, 
                 title=f"{selected_player} - Runs by Team")
    fig.update_layout(xaxis_title="Team", yaxis_title="Runs")
    st.plotly_chart(fig, use_container_width=True)

def show_match_analysis(df):
    st.markdown('<h2 class="sub-header">🏟️ Match Analysis</h2>', unsafe_allow_html=True)
    
    # Match results
    st.subheader("Match Results Distribution")
    if 'match_won_by' in df.columns:
        match_results = df['match_won_by'].value_counts()
        fig = px.pie(values=match_results.values, names=match_results.index, 
                     title="Match Results Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    # Runs per match
    st.subheader("Average Runs per Match")
    match_runs = df.groupby('match_id')['runs_total'].sum()
    avg_runs = match_runs.mean()
    st.metric("Average Runs per Match", f"{avg_runs:.2f}")
    
    # Runs progression in matches
    st.subheader("Runs Progression in Selected Match")
    selected_match = st.selectbox("Select Match ID:", df['match_id'].unique())
    match_data = df[df['match_id'] == selected_match].sort_values('ball_no')
    
    fig = px.line(match_data, x='ball_no', y='runs_total', 
                  title=f"Runs Progression in Match {selected_match}")
    fig.update_layout(xaxis_title="Ball Number", yaxis_title="Runs")
    st.plotly_chart(fig, use_container_width=True)

def show_venue_analysis(df):
    st.markdown('<h2 class="sub-header">🏟️ Venue Analysis</h2>', unsafe_allow_html=True)
    
    # Venue performance
    st.subheader("Top Venues by Total Runs")
    venue_runs = df.groupby('venue')['runs_total'].sum().sort_values(ascending=False).head(10)
    fig = px.bar(x=venue_runs.index, y=venue_runs.values, title="Top 10 Venues by Total Runs")
    fig.update_layout(xaxis_title="Venue", yaxis_title="Total Runs")
    st.plotly_chart(fig, use_container_width=True)
    
    # City analysis
    st.subheader("Top Cities by Total Runs")
    city_runs = df.groupby('city')['runs_total'].sum().sort_values(ascending=False).head(10)
    fig = px.bar(x=city_runs.index, y=city_runs.values, title="Top 10 Cities by Total Runs")
    fig.update_layout(xaxis_title="City", yaxis_title="Total Runs")
    st.plotly_chart(fig, use_container_width=True)

def show_toss_analysis(df):
    st.markdown('<h2 class="sub-header">🎯 Toss Analysis</h2>', unsafe_allow_html=True)
    
    if 'toss_winner' in df.columns and 'toss_decision' in df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            # Toss winner distribution
            toss_winners = df['toss_winner'].value_counts()
            fig = px.pie(values=toss_winners.values, names=toss_winners.index, 
                         title="Toss Winners Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Toss decision distribution
            toss_decisions = df['toss_decision'].value_counts()
            fig = px.pie(values=toss_decisions.values, names=toss_decisions.index, 
                         title="Toss Decision Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        # Toss winner vs match winner
        st.subheader("Toss Winner vs Match Winner")
        if 'match_won_by' in df.columns:
            toss_match_df = df.groupby(['toss_winner', 'match_won_by']).size().reset_index(name='count')
            fig = px.scatter(toss_match_df, x='toss_winner', y='match_won_by', size='count',
                             title="Toss Winner vs Match Winner")
            st.plotly_chart(fig, use_container_width=True)

def show_wicket_analysis(df):
    st.markdown('<h2 class="sub-header">⚾ Wicket Analysis</h2>', unsafe_allow_html=True)
    
    wicket_data = df[df['wicket_kind'].notna()]
    
    if not wicket_data.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # Wicket types distribution
            wicket_types = wicket_data['wicket_kind'].value_counts()
            fig = px.pie(values=wicket_types.values, names=wicket_types.index, 
                         title="Wicket Types Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Wickets per match
            wickets_per_match = wicket_data.groupby('match_id').size()
            fig = px.histogram(x=wickets_per_match, nbins=10, 
                               title="Distribution of Wickets per Match")
            fig.update_layout(xaxis_title="Wickets per Match", yaxis_title="Frequency")
            st.plotly_chart(fig, use_container_width=True)
        
        # Wickets by bowler
        st.subheader("Top Wicket Takers")
        bowler_wickets = wicket_data.groupby('bowler').size().sort_values(ascending=False).head(10)
        fig = px.bar(x=bowler_wickets.index, y=bowler_wickets.values, 
                     title="Top 10 Wicket Takers")
        fig.update_layout(xaxis_title="Bowler", yaxis_title="Wickets")
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
