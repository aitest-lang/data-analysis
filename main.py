import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import zipfile

with zipfile.ZipFile('archive.zip', 'r') as zip_ref:
    zip_ref.extractall()
# Team Aliases Dictionary
TEAM_ALIASES = {
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
    "Rising Pune Supergiant": "Rising Pune Supergiant",
    "Kings XI Punjab": "Punjab Kings",
    "Delhi Daredevils": "Delhi Capitals",
    "Pune Warriors": "Pune Warriors India",
}

# Load and Preprocess Data
@st.cache_data
def load_data():
    df = pd.read_csv("IPL.csv")
    
    # Apply team aliases
    df['batting_team'] = df['batting_team'].replace(TEAM_ALIASES)
    df['bowling_team'] = df['bowling_team'].replace(TEAM_ALIASES)
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Create derived columns
    df['is_powerplay'] = df['over'] < 6
    df['is_death'] = df['over'] >= 15
    df['is_boundary'] = df['runs_total'].isin([4, 6])
    df['batter_sr'] = (df['runs_batter'] / df['balls_faced']) * 100
    df['bowler_econ'] = (df['runs_bowler'] / (df['valid_ball'] / 6))
    
    # Fill missing values
    df['batter_sr'] = df['batter_sr'].fillna(0)
    df['bowler_econ'] = df['bowler_econ'].fillna(0)
    
    return df

# Create Aggregated Data
@st.cache_data
def create_aggregates(df):
    # Team Stats
    team_stats = df.groupby('batting_team').agg(
        matches=('match_id', 'nunique'),
        avg_runs=('runs_total', 'mean'),
        powerplay_avg=('runs_total', lambda x: x[df['is_powerplay']].mean()),
        death_avg=('runs_total', lambda x: x[df['is_death']].mean()),
        win_rate=('match_won_by', lambda x: (x == df['batting_team']).mean() * 100)
    ).reset_index()
    
    # Batter Stats
    batter_stats = df.groupby('batter').agg(
        total_runs=('runs_batter', 'sum'),
        balls_faced=('balls_faced', 'sum'),
        strike_rate=('batter_sr', 'mean'),
        boundaries=('is_boundary', 'sum'),
        matches=('match_id', 'nunique')
    ).reset_index()
    batter_stats = batter_stats[batter_stats['balls_faced'] > 50]
    
    # Bowler Stats
    bowler_stats = df.groupby('bowler').agg(
        wickets=('bowler_wicket', 'sum'),
        runs_conceded=('runs_bowler', 'sum'),
        balls_bowled=('valid_ball', 'sum'),
        economy=('bowler_econ', 'mean'),
        matches=('match_id', 'nunique')
    ).reset_index()
    bowler_stats = bowler_stats[bowler_stats['balls_bowled'] > 100]
    
    # Match Stats
    match_stats = df.groupby(['match_id', 'batting_team']).agg(
        total_runs=('runs_total', 'sum'),
        total_wickets=('bowler_wicket', 'sum'),
        venue=('venue', 'first'),
        toss_winner=('toss_winner', 'first'),
        match_won_by=('match_won_by', 'first'),
        date=('date', 'first')
    ).reset_index()
    
    # Seasonal Stats
    seasonal_stats = df.groupby('season').agg(
        avg_runs=('runs_total', 'mean'),
        avg_wickets=('bowler_wicket', 'mean'),
        matches=('match_id', 'nunique')
    ).reset_index()
    
    return team_stats, batter_stats, bowler_stats, match_stats, seasonal_stats

# Main App
def main():
    st.set_page_config(layout="wide", page_title="IPL Data Analysis Platform")
    st.title("🏏 IPL Data Analysis Platform")
    
    # Load data
    df = load_data()
    team_stats, batter_stats, bowler_stats, match_stats, seasonal_stats = create_aggregates(df)
    
    # Sidebar Filters
    st.sidebar.header("Filters")
    seasons = sorted(df['season'].unique())
    selected_season = st.sidebar.multiselect("Select Season", seasons, default=seasons)
    teams = sorted(df['batting_team'].unique())
    selected_teams = st.sidebar.multiselect("Select Teams", teams, default=teams[:5])
    
    # Filter data based on selections
    filtered_df = df[df['season'].isin(selected_season) & df['batting_team'].isin(selected_teams)]
    filtered_team_stats = team_stats[team_stats['batting_team'].isin(selected_teams)]
    filtered_batter_stats = batter_stats[batter_stats['batter'].isin(
        filtered_df[filtered_df['batter'].notnull()]['batter'].unique())]
    filtered_bowler_stats = bowler_stats[bowler_stats['bowler'].isin(
        filtered_df[filtered_df['bowler'].notnull()]['bowler'].unique())]
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Team Performance", "Player Stats", "Match Insights", "Seasonal Trends"])
    
    # Tab 1: Team Performance
    with tab1:
        st.header("Team Performance")
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = px.bar(filtered_team_stats, x='batting_team', y='avg_runs',
                         title="Average Runs per Match", color='batting_team')
            st.plotly_chart(fig1, use_container_width=True)
            
        with col2:
            fig2 = px.bar(filtered_team_stats, x='batting_team', y='win_rate',
                         title="Win Rate (%)", color='batting_team')
            st.plotly_chart(fig2, use_container_width=True)
        
        col3, col4 = st.columns(2)
        with col3:
            fig3 = px.bar(filtered_team_stats, x='batting_team', y='powerplay_avg',
                         title="Powerplay Average (Overs 1-6)", color='batting_team')
            st.plotly_chart(fig3, use_container_width=True)
            
        with col4:
            fig4 = px.bar(filtered_team_stats, x='batting_team', y='death_avg',
                         title="Death Overs Average (Overs 15-20)", color='batting_team')
            st.plotly_chart(fig4, use_container_width=True)
    
    # Tab 2: Player Stats
    with tab2:
        st.header("Player Statistics")
        player_type = st.radio("Select Player Type", ["Batsmen", "Bowlers"])
        
        if player_type == "Batsmen":
            st.subheader("Top Batsmen")
            top_batters = filtered_batter_stats.nlargest(10, 'total_runs')
            fig5 = px.scatter(top_batters, x='strike_rate', y='total_runs',
                             color='batter', size='boundaries',
                             hover_data=['matches', 'balls_faced'],
                             title="Top Batsmen: Strike Rate vs Total Runs")
            st.plotly_chart(fig5, use_container_width=True)
            
            st.dataframe(top_batters[['batter', 'total_runs', 'strike_rate', 'boundaries', 'matches']])
        
        else:
            st.subheader("Top Bowlers")
            top_bowlers = filtered_bowler_stats.nlargest(10, 'wickets')
            fig6 = px.scatter(top_bowlers, x='economy', y='wickets',
                             color='bowler', size='matches',
                             hover_data=['runs_conceded', 'balls_bowled'],
                             title="Top Bowlers: Economy vs Wickets")
            st.plotly_chart(fig6, use_container_width=True)
            
            st.dataframe(top_bowlers[['bowler', 'wickets', 'economy', 'matches']])
    
    # Tab 3: Match Insights
    with tab3:
        st.header("Match Insights")
        col5, col6 = st.columns(2)
        
        with col5:
            # Toss Impact
            toss_impact = match_stats.copy()
            toss_impact['toss_won'] = toss_impact['toss_winner'] == toss_impact['batting_team']
            toss_impact['match_won'] = toss_impact['match_won_by'] == toss_impact['batting_team']
            toss_stats = toss_impact.groupby('toss_won')['match_won'].mean() * 100
            
            fig7 = px.bar(x=['Lost Toss', 'Won Toss'], y=toss_stats.values,
                         title="Win % Based on Toss Result",
                         labels={'x': 'Toss Result', 'y': 'Win %'})
            st.plotly_chart(fig7, use_container_width=True)
        
        with col6:
            # Venue Analysis
            venue_stats = match_stats.groupby('venue')['total_runs'].mean().nlargest(10)
            fig8 = px.bar(x=venue_stats.index, y=venue_stats.values,
                         title="Top 10 High-Scoring Venues",
                         labels={'x': 'Venue', 'y': 'Average Runs'})
            fig8.update_xaxes(tickangle=45)
            st.plotly_chart(fig8, use_container_width=True)
        
        # Match Outcome Distribution
        outcome_dist = match_stats['match_won_by'].value_counts()
        fig9 = px.pie(values=outcome_dist.values, names=outcome_dist.index,
                     title="Match Outcome Distribution")
        st.plotly_chart(fig9, use_container_width=True)
    
    # Tab 4: Seasonal Trends
    with tab4:
        st.header("Seasonal Trends")
        filtered_seasonal = seasonal_stats[seasonal_stats['season'].isin(selected_season)]
        
        fig10 = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig10.add_trace(
            go.Scatter(x=filtered_seasonal['season'], y=filtered_seasonal['avg_runs'],
                      name="Avg Runs", line=dict(color='blue')),
            secondary_y=False
        )
        
        fig10.add_trace(
            go.Scatter(x=filtered_seasonal['season'], y=filtered_seasonal['avg_wickets'],
                      name="Avg Wickets", line=dict(color='red')),
            secondary_y=True
        )
        
        fig10.update_xaxes(title_text="Season")
        fig10.update_yaxes(title_text="Average Runs", secondary_y=False)
        fig10.update_yaxes(title_text="Average Wickets", secondary_y=True)
        fig10.update_layout(title_text="Seasonal Trends: Runs and Wickets")
        
        st.plotly_chart(fig10, use_container_width=True)
        
        # Team Performance Trajectory
        team_season = df.groupby(['season', 'batting_team']).agg(
            avg_runs=('runs_total', 'mean'),
            win_rate=('match_won_by', lambda x: (x == df['batting_team']).mean() * 100)
        ).reset_index()
        
        selected_team = st.selectbox("Select Team for Performance Trajectory", selected_teams)
        team_data = team_season[team_season['batting_team'] == selected_team]
        
        fig11 = make_subplots(specs=[[{"secondary_y": True}]])
        fig11.add_trace(
            go.Scatter(x=team_data['season'], y=team_data['avg_runs'],
                      name="Avg Runs", line=dict(color='green')),
            secondary_y=False
        )
        fig11.add_trace(
            go.Scatter(x=team_data['season'], y=team_data['win_rate'],
                      name="Win Rate", line=dict(color='purple')),
            secondary_y=True
        )
        fig11.update_xaxes(title_text="Season")
        fig11.update_yaxes(title_text="Average Runs", secondary_y=False)
        fig11.update_yaxes(title_text="Win Rate (%)", secondary_y=True)
        fig11.update_layout(title_text=f"{selected_team} Performance Trajectory")
        
        st.plotly_chart(fig11, use_container_width=True)

if __name__ == "__main__":
    main()
