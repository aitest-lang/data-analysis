import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import zipfile

with zipfile.ZipFile('archive.zip', 'r') as zip_ref:
    zip_ref.extractall()
# Configuration
st.set_page_config(page_title="IPL Data Analysis Platform", layout="wide")

# Team Aliases
TEAM_ALIASES = {
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
    "Rising Pune Supergiant": "Rising Pune Supergiant",
    "Kings XI Punjab": "Punjab Kings",
    "Delhi Daredevils": "Delhi Capitals",
    "Pune Warriors": "Pune Warriors India",
}

# Load and Process Data
@st.cache_data
def load_data():
    df = pd.read_csv("IPL.csv")
    
    # Apply team aliases
    df['batting_team'] = df['batting_team'].replace(TEAM_ALIASES)
    df['bowling_team'] = df['bowling_team'].replace(TEAM_ALIASES)
    
    # Create derived columns
    df['is_boundary'] = ~df['runs_not_boundary']
    df['phase'] = np.select(
        [df['over'] <= 6, df['over'] >= 16],
        ['Powerplay', 'Death'],
        default='Middle'
    )
    df['run_rate'] = df['runs_total'] / (df['over'] * 6 + df['ball'])
    
    # Convert date
    df['date'] = pd.to_datetime(df['date'])
    
    return df

# Compute Team Metrics
@st.cache_data
def compute_team_metrics(df):
    # Match results
    match_results = df[['match_id', 'match_won_by', 'toss_winner', 'toss_decision']].drop_duplicates()
    
    # Teams in each match
    teams_in_match = df.groupby('match_id')['batting_team'].apply(lambda x: list(set(x))).reset_index()
    teams_in_match = teams_in_match.explode('batting_team').rename(columns={'batting_team': 'team'})
    
    # Merge with match results
    team_matches = teams_in_match.merge(match_results, on='match_id')
    team_matches['won'] = team_matches['team'] == team_matches['match_won_by']
    team_matches['toss_won'] = team_matches['team'] == team_matches['toss_winner']
    
    # Team summary
    team_summary = team_matches.groupby('team').agg(
        matches_played=('won', 'count'),
        matches_won=('won', 'sum'),
        toss_won=('toss_won', 'sum'),
        toss_won_match_won=('toss_won', lambda x: (x & team_matches.loc[x.index, 'won']).sum())
    ).reset_index()
    
    team_summary['win_pct'] = (team_summary['matches_won'] / team_summary['matches_played'] * 100).round(1)
    team_summary['toss_win_match_win_pct'] = (team_summary['toss_won_match_won'] / team_summary['toss_won'] * 100).round(1)
    
    # Team scoring
    team_scores = df.groupby(['match_id', 'batting_team'])['runs_total'].sum().reset_index()
    team_avg_scores = team_scores.groupby('batting_team')['runs_total'].mean().round(1).reset_index()
    team_avg_scores.columns = ['team', 'avg_score']
    
    # Merge
    team_summary = team_summary.merge(team_avg_scores, on='team')
    
    return team_summary

# Compute Player Metrics
@st.cache_data
def compute_player_metrics(df):
    # Batsman stats
    batsman_stats = df.groupby('batter').agg(
        runs=('runs_batter', 'sum'),
        balls=('balls_faced', 'sum'),
        boundaries=('is_boundary', 'sum'),
        dismissals=('striker_out', 'sum')
    ).reset_index()
    
    batsman_stats['strike_rate'] = (batsman_stats['runs'] / batsman_stats['balls'] * 100).round(1)
    batsman_stats['boundary_pct'] = (batsman_stats['boundaries'] / batsman_stats['balls'] * 100).round(1)
    batsman_stats['average'] = (batsman_stats['runs'] / batsman_stats['dismissals']).replace([np.inf, -np.inf], 0).round(1)
    
    # Bowler stats
    bowler_stats = df.groupby('bowler').agg(
        wickets=('bowler_wicket', 'sum'),
        runs_conceded=('runs_bowler', 'sum'),
        balls_bowled=('valid_ball', 'sum')
    ).reset_index()
    
    bowler_stats['economy'] = (bowler_stats['runs_conceded'] / (bowler_stats['balls_bowled'] / 6)).round(2)
    bowler_stats['bowling_avg'] = (bowler_stats['runs_conceded'] / bowler_stats['wickets']).replace([np.inf, -np.inf], 0).round(1)
    bowler_stats['strike_rate'] = (bowler_stats['balls_bowled'] / bowler_stats['wickets']).replace([np.inf, -np.inf], 0).round(1)
    
    return batsman_stats, bowler_stats

# Compute Match Dynamics
@st.cache_data
def compute_match_dynamics(df):
    # Over progression
    over_stats = df.groupby('over').agg(
        avg_runs=('runs_total', 'mean'),
        avg_wickets=('bowler_wicket', 'mean')
    ).reset_index()
    
    # Venue stats
    venue_stats = df.groupby('venue').agg(
        avg_runs=('runs_total', 'mean'),
        matches=('match_id', lambda x: len(set(x)))
    ).reset_index()
    venue_stats = venue_stats[venue_stats['matches'] >= 5].sort_values('avg_runs', ascending=False)
    
    return over_stats, venue_stats

# Main App
def main():
    st.title("🏏 IPL Data Analysis Platform")
    st.markdown("Interactive dashboard for IPL ball-by-ball data analysis")
    
    # Load data
    df = load_data()
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Season filter
    seasons = sorted(df['season'].unique())
    selected_season = st.sidebar.selectbox("Select Season", ['All'] + list(seasons))
    
    # Team filter
    teams = sorted(df['batting_team'].unique())
    selected_team = st.sidebar.selectbox("Select Team", ['All'] + list(teams))
    
    # Apply filters
    df_filtered = df.copy()
    if selected_season != 'All':
        df_filtered = df_filtered[df_filtered['season'] == selected_season]
    if selected_team != 'All':
        df_filtered = df_filtered[(df_filtered['batting_team'] == selected_team) | 
                                  (df_filtered['bowling_team'] == selected_team)]
    
    # Compute metrics
    team_metrics = compute_team_metrics(df_filtered)
    batsman_stats, bowler_stats = compute_player_metrics(df_filtered)
    over_stats, venue_stats = compute_match_dynamics(df_filtered)
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["Team Performance", "Player Analytics", "Match Dynamics"])
    
    # Tab 1: Team Performance
    with tab1:
        st.header("Team Performance")
        
        # Key metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Teams", len(team_metrics))
        with col2:
            st.metric("Avg Win %", f"{team_metrics['win_pct'].mean():.1f}%")
        with col3:
            st.metric("Toss Win → Match Win", f"{team_metrics['toss_win_match_win_pct'].mean():.1f}%")
        
        # Win percentage chart
        st.subheader("Win Percentage")
        fig_win = px.bar(
            team_metrics.sort_values('win_pct', ascending=False),
            x='win_pct',
            y='team',
            orientation='h',
            color='win_pct',
            color_continuous_scale='RdYlGn',
            height=400
        )
        fig_win.update_layout(
            xaxis_title="Win Percentage",
            yaxis_title="Team",
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_win, use_container_width=True)
        
        # Average scores
        st.subheader("Average Scores per Match")
        fig_score = px.bar(
            team_metrics.sort_values('avg_score', ascending=False),
            x='avg_score',
            y='team',
            orientation='h',
            color='avg_score',
            color_continuous_scale='Blues',
            height=400
        )
        fig_score.update_layout(
            xaxis_title="Average Runs",
            yaxis_title="Team",
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_score, use_container_width=True)
        
        # Toss impact
        st.subheader("Toss Impact")
        fig_toss = px.bar(
            team_metrics.sort_values('toss_win_match_win_pct', ascending=False),
            x='toss_win_match_win_pct',
            y='team',
            orientation='h',
            color='toss_win_match_win_pct',
            color_continuous_scale='Teal',
            height=400
        )
        fig_toss.update_layout(
            xaxis_title="Toss Win → Match Win %",
            yaxis_title="Team",
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_toss, use_container_width=True)
    
    # Tab 2: Player Analytics
    with tab2:
        st.header("Player Analytics")
        
        # Top batsmen
        st.subheader("Top Batsmen")
        top_batsmen = batsman_stats.sort_values('runs', ascending=False).head(10)
        fig_batsmen = px.bar(
            top_batsmen,
            x='runs',
            y='batter',
            orientation='h',
            color='strike_rate',
            color_continuous_scale='Viridis',
            height=400
        )
        fig_batsmen.update_layout(
            xaxis_title="Total Runs",
            yaxis_title="Batsman",
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_batsmen, use_container_width=True)
        
        # Top bowlers
        st.subheader("Top Bowlers")
        top_bowlers = bowler_stats.sort_values('wickets', ascending=False).head(10)
        fig_bowlers = px.bar(
            top_bowlers,
            x='wickets',
            y='bowler',
            orientation='h',
            color='economy',
            color_continuous_scale='Plasma',
            height=400
        )
        fig_bowlers.update_layout(
            xaxis_title="Total Wickets",
            yaxis_title="Bowler",
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_bowlers, use_container_width=True)
        
        # Boundary percentage
        st.subheader("Boundary Percentage")
        top_boundary = batsman_stats[batsman_stats['balls'] >= 100].sort_values('boundary_pct', ascending=False).head(10)
        fig_boundary = px.bar(
            top_boundary,
            x='boundary_pct',
            y='batter',
            orientation='h',
            color='boundary_pct',
            color_continuous_scale='Hot',
            height=400
        )
        fig_boundary.update_layout(
            xaxis_title="Boundary %",
            yaxis_title="Batsman",
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_boundary, use_container_width=True)
    
    # Tab 3: Match Dynamics
    with tab3:
        st.header("Match Dynamics")
        
        # Over progression
        st.subheader("Run Rate Progression")
        fig_over = px.line(
            over_stats,
            x='over',
            y='avg_runs',
            markers=True,
            line_shape='linear'
        )
        fig_over.update_layout(
            xaxis_title="Over",
            yaxis_title="Average Runs per Ball",
            height=400
        )
        st.plotly_chart(fig_over, use_container_width=True)
        
        # Wicket timeline
        st.subheader("Wicket Timeline")
        fig_wicket = px.line(
            over_stats,
            x='over',
            y='avg_wickets',
            markers=True,
            line_shape='linear',
            color_discrete_sequence=['red']
        )
        fig_wicket.update_layout(
            xaxis_title="Over",
            yaxis_title="Average Wickets per Ball",
            height=400
        )
        st.plotly_chart(fig_wicket, use_container_width=True)
        
        # Venue comparison
        st.subheader("Venue Comparison")
        fig_venue = px.bar(
            venue_stats.head(15),
            x='avg_runs',
            y='venue',
            orientation='h',
            color='avg_runs',
            color_continuous_scale='Rainbow',
            height=500
        )
        fig_venue.update_layout(
            xaxis_title="Average Runs per Match",
            yaxis_title="Venue",
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_venue, use_container_width=True)

if __name__ == "__main__":
    main()
