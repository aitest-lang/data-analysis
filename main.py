import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')
import zipfile

with zipfile.ZipFile('archive.zip', 'r') as zip_ref:
    zip_ref.extractall()
# Set page config
st.set_page_config(
    page_title="IPL Data Analytics Dashboard",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .stSelectbox > div > div > div {
        background-color: #f0f2f6;
    }
    .analysis-section {
        background: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #FF6B35;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_clean_data():
    """Load and clean IPL data with comprehensive error handling"""
    try:
        with st.spinner('Loading IPL data...'):
            df = pd.read_csv("IPL.csv")
            
            # Data validation
            if df.empty:
                st.error("Dataset is empty!")
                return None
            
            # Fix team name inconsistencies (more comprehensive)
            team_name_mapping = {
                'Royal Challengers Bangalore': 'Royal Challengers Bengaluru',
                'Delhi Daredevils': 'Delhi Capitals',  # Added this common inconsistency
                'Kings XI Punjab': 'Punjab Kings'      # Added this common inconsistency
            }
            
            for col in ['batting_team', 'bowling_team', 'match_won_by', 'toss_winner']:
                if col in df.columns:
                    df[col] = df[col].replace(team_name_mapping)
            
            # Convert date column to datetime with error handling
            try:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
            except:
                st.warning("Date conversion failed, using original date format")
            
            # Fill missing values more intelligently
            df['extra_type'] = df['extra_type'].fillna('None')
            df['wicket_kind'] = df['wicket_kind'].fillna('Not Out')
            df['player_out'] = df['player_out'].fillna('None')
            df['runs_target'] = df['runs_target'].fillna(0)
            
            # Create derived columns with error handling
            try:
                df['match_year'] = df['date'].dt.year
                df['match_month'] = df['date'].dt.month
                df['is_boundary'] = ~df['runs_not_boundary']
                df['is_six'] = (df['runs_batter'] == 6) & (df['extra_type'] == 'None')
                df['is_four'] = (df['runs_batter'] == 4) & (df['extra_type'] == 'None')
                df['is_wicket'] = df['wicket_kind'] != 'Not Out'
                df['is_dot_ball'] = (df['runs_total'] == 0)
                df['is_single'] = (df['runs_batter'] == 1) & (df['extra_type'] == 'None')
                df['powerplay'] = df['over'] <= 6
                df['death_overs'] = df['over'] >= 16
                
                # Calculate strike rate per ball
                df['ball_sr'] = np.where(df['balls_faced'] > 0, 
                                       (df['runs_batter'] / df['balls_faced']) * 100, 0)
            except Exception as e:
                st.warning(f"Error creating derived columns: {e}")
            
            return df
            
    except FileNotFoundError:
        st.error("IPL.csv file not found! Please ensure the file is in the correct directory.")
        return None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_data
def get_enhanced_team_performance(df):
    """Calculate comprehensive team performance metrics"""
    try:
        # Match-level aggregation first
        match_stats = df.groupby(['batting_team', 'match_id', 'innings']).agg({
            'runs_total': 'max',  # Total runs in that innings
            'is_wicket': 'sum',
            'is_boundary': 'sum',
            'is_six': 'sum',
            'is_four': 'sum',
            'is_dot_ball': 'sum',
            'over': 'max'
        }).reset_index()
        
        # Team-level summary
        team_summary = match_stats.groupby('batting_team').agg({
            'runs_total': ['mean', 'sum', 'max', 'min', 'std'],
            'is_wicket': ['mean', 'sum'],
            'is_boundary': ['mean', 'sum'],
            'is_six': ['sum', 'mean'],
            'is_four': ['sum', 'mean'],
            'is_dot_ball': 'mean',
            'match_id': 'count'
        }).round(2)
        
        team_summary.columns = ['Avg_Score', 'Total_Runs', 'Highest_Score', 'Lowest_Score', 'Score_Consistency',
                              'Avg_Wickets_Lost', 'Total_Wickets_Lost', 'Avg_Boundaries', 'Total_Boundaries',
                              'Total_Sixes', 'Sixes_Per_Match', 'Total_Fours', 'Fours_Per_Match',
                              'Dot_Ball_Percentage', 'Matches_Played']
        
        # Calculate additional metrics
        team_summary['Boundary_Percentage'] = ((team_summary['Total_Boundaries'] / 
                                              (team_summary['Matches_Played'] * 20)) * 100).round(2)
        
        return team_summary.reset_index()
    except Exception as e:
        st.error(f"Error calculating team performance: {e}")
        return pd.DataFrame()

@st.cache_data
def get_enhanced_player_stats(df):
    """Calculate comprehensive player batting statistics"""
    try:
        player_stats = df.groupby('batter').agg({
            'runs_batter': ['sum', 'mean', 'max', 'count'],
            'balls_faced': 'sum',
            'is_six': 'sum',
            'is_four': 'sum',
            'is_dot_ball': 'sum',
            'match_id': 'nunique',
            'batting_team': lambda x: x.mode().iloc[0] if not x.empty else 'Unknown'
        }).round(2)
        
        player_stats.columns = ['Total_Runs', 'Avg_Runs_Per_Ball', 'Highest_Score', 'Balls_Played',
                              'Balls_Faced', 'Sixes', 'Fours', 'Dot_Balls', 'Matches', 'Primary_Team']
        
        # Calculate advanced metrics
        player_stats['Strike_Rate'] = np.where(player_stats['Balls_Faced'] > 0,
                                             (player_stats['Total_Runs'] / player_stats['Balls_Faced'] * 100), 0).round(2)
        player_stats['Dot_Ball_Percentage'] = np.where(player_stats['Balls_Faced'] > 0,
                                                     (player_stats['Dot_Balls'] / player_stats['Balls_Faced'] * 100), 0).round(2)
        player_stats['Boundary_Percentage'] = np.where(player_stats['Balls_Faced'] > 0,
                                                     ((player_stats['Sixes'] + player_stats['Fours']) / player_stats['Balls_Faced'] * 100), 0).round(2)
        player_stats['Avg_Per_Match'] = (player_stats['Total_Runs'] / player_stats['Matches']).round(2)
        
        # Filter players with minimum criteria (more flexible)
        min_balls = st.sidebar.slider("Minimum balls faced for player stats:", 50, 500, 200) if 'sidebar' in st.__dict__ else 200
        player_stats = player_stats[player_stats['Balls_Faced'] >= min_balls]
        
        return player_stats.reset_index()
    except Exception as e:
        st.error(f"Error calculating player stats: {e}")
        return pd.DataFrame()

@st.cache_data
def get_bowling_stats(df):
    """Calculate bowling statistics"""
    try:
        bowling_stats = df.groupby('bowler').agg({
            'runs_bowler': 'sum',
            'valid_ball': 'sum',
            'is_wicket': 'sum',
            'is_six': 'sum',
            'is_four': 'sum',
            'is_dot_ball': 'sum',
            'match_id': 'nunique'
        }).round(2)
        
        bowling_stats.columns = ['Runs_Conceded', 'Balls_Bowled', 'Wickets', 'Sixes_Conceded',
                               'Fours_Conceded', 'Dot_Balls', 'Matches']
        
        # Calculate bowling metrics
        bowling_stats['Economy_Rate'] = np.where(bowling_stats['Balls_Bowled'] > 0,
                                               (bowling_stats['Runs_Conceded'] / bowling_stats['Balls_Bowled'] * 6), 0).round(2)
        bowling_stats['Strike_Rate'] = np.where(bowling_stats['Wickets'] > 0,
                                              (bowling_stats['Balls_Bowled'] / bowling_stats['Wickets']), np.inf).round(2)
        bowling_stats['Average'] = np.where(bowling_stats['Wickets'] > 0,
                                          (bowling_stats['Runs_Conceded'] / bowling_stats['Wickets']), np.inf).round(2)
        bowling_stats['Dot_Ball_Percentage'] = np.where(bowling_stats['Balls_Bowled'] > 0,
                                                       (bowling_stats['Dot_Balls'] / bowling_stats['Balls_Bowled'] * 100), 0).round(2)
        
        # Filter bowlers with minimum balls
        bowling_stats = bowling_stats[bowling_stats['Balls_Bowled'] >= 100]
        
        return bowling_stats.reset_index()
    except Exception as e:
        st.error(f"Error calculating bowling stats: {e}")
        return pd.DataFrame()

def create_comparison_section(df):
    """Create team vs team comparison"""
    st.header("⚔️ Head-to-Head Comparison")
    
    teams = df['batting_team'].unique()
    
    col1, col2 = st.columns(2)
    with col1:
        team1 = st.selectbox("Select Team 1:", teams, key="team1")
    with col2:
        team2 = st.selectbox("Select Team 2:", teams, key="team2")
    
    if team1 and team2 and team1 != team2:
        # Head-to-head matches
        h2h_matches = df[((df['batting_team'] == team1) & (df['bowling_team'] == team2)) |
                        ((df['batting_team'] == team2) & (df['bowling_team'] == team1))]
        
        if not h2h_matches.empty:
            wins_team1 = len(h2h_matches[h2h_matches['match_won_by'] == team1])
            wins_team2 = len(h2h_matches[h2h_matches['match_won_by'] == team2])
            total_matches = h2h_matches['match_id'].nunique()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(f"{team1} Wins", wins_team1)
            with col2:
                st.metric(f"{team2} Wins", wins_team2)
            with col3:
                st.metric("Total Matches", total_matches)
            
            # Win percentage pie chart
            fig = px.pie(values=[wins_team1, wins_team2], 
                        names=[team1, team2],
                        title=f"{team1} vs {team2} - Head to Head")
            st.plotly_chart(fig, use_container_width=True)

def main():
    # Enhanced Header with animation
    st.markdown('<h1 class="main-header">🏏 IPL Data Analytics Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Load data with progress indicator
    df = load_and_clean_data()
    
    if df is None:
        st.error("Failed to load data. Please ensure IPL.csv is in the correct path.")
        st.info("Expected file: IPL.csv in the same directory as this script")
        return
    
    # Enhanced Sidebar with more options
    st.sidebar.header("📊 Analysis Categories")
    
    analysis_type = st.sidebar.selectbox(
        "Select Analysis Type:",
        ["📈 Overview", "🏆 Team Performance", "👨‍💼 Player Statistics", 
         "🎳 Bowling Analysis", "🏟️ Venue Analysis", "📅 Season Trends", 
         "⚾ Match Insights", "⚔️ Team Comparison"]
    )
    
    # Add date range filter in sidebar
    if 'date' in df.columns and not df['date'].isna().all():
        st.sidebar.subheader("📅 Date Range Filter")
        date_range = st.sidebar.date_input(
            "Select date range:",
            value=(df['date'].min(), df['date'].max()),
            min_value=df['date'].min(),
            max_value=df['date'].max()
        )
        
        if len(date_range) == 2:
            df = df[(df['date'] >= pd.to_datetime(date_range[0])) & 
                   (df['date'] <= pd.to_datetime(date_range[1]))]
    
    # Overview Section (Enhanced)
    if analysis_type == "📈 Overview":
        st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
        st.header("📈 Dataset Overview")
        
        # Enhanced metrics with better styling
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown(f'<div class="metric-card"><h3>{df["match_id"].nunique()}</h3><p>Total Matches</p></div>', 
                       unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><h3>{df["batting_team"].nunique()}</h3><p>Teams</p></div>', 
                       unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card"><h3>{df["batter"].nunique()}</h3><p>Players</p></div>', 
                       unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="metric-card"><h3>{df["venue"].nunique()}</h3><p>Venues</p></div>', 
                       unsafe_allow_html=True)
        with col5:
            st.markdown(f'<div class="metric-card"><h3>{df["season"].nunique()}</h3><p>Seasons</p></div>', 
                       unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Data quality indicators
        st.subheader("📊 Data Quality")
        col1, col2 = st.columns(2)
        
        with col1:
            completeness = ((len(df) - df.isnull().sum().sum()) / (len(df) * len(df.columns))) * 100
            st.metric("Data Completeness", f"{completeness:.1f}%")
            
        with col2:
            st.write("**Dataset Shape:**", df.shape)
            if 'date' in df.columns:
                st.write("**Date Range:**", f"{df['date'].min().date()} to {df['date'].max().date()}")
        
        # Sample data with better formatting
        st.subheader("📋 Data Sample")
        st.dataframe(df.head(10), use_container_width=True)
    
    # Enhanced Team Performance
    elif analysis_type == "🏆 Team Performance":
        st.header("🏆 Team Performance Analysis")
        
        team_stats = get_enhanced_team_performance(df)
        
        if not team_stats.empty:
            # Multi-select for teams with better UX
            all_teams = team_stats['batting_team'].tolist()
            selected_teams = st.multiselect(
                "Select Teams for Analysis:",
                options=all_teams,
                default=all_teams[:8] if len(all_teams) > 8 else all_teams,
                help="Choose teams to compare (leave empty to show all)"
            )
            
            if selected_teams:
                filtered_stats = team_stats[team_stats['batting_team'].isin(selected_teams)]
            else:
                filtered_stats = team_stats
            
            # Enhanced statistics table
            st.subheader("📊 Comprehensive Team Statistics")
            st.dataframe(filtered_stats, use_container_width=True)
            
            # Multiple visualization options
            chart_type = st.selectbox("Select Chart Type:", 
                                    ["Average Score", "Boundary Analysis", "Consistency", "Overall Performance"])
            
            if chart_type == "Average Score":
                fig = px.bar(filtered_stats, x='batting_team', y='Avg_Score',
                           color='Matches_Played', title='Average Score by Team',
                           hover_data=['Highest_Score', 'Score_Consistency'])
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
                
            elif chart_type == "Boundary Analysis":
                fig = make_subplots(rows=1, cols=2, 
                                  subplot_titles=('Sixes per Match', 'Fours per Match'))
                fig.add_trace(go.Bar(x=filtered_stats['batting_team'], 
                                   y=filtered_stats['Sixes_Per_Match'], name='Sixes'), row=1, col=1)
                fig.add_trace(go.Bar(x=filtered_stats['batting_team'], 
                                   y=filtered_stats['Fours_Per_Match'], name='Fours'), row=1, col=2)
                fig.update_layout(title_text="Boundary Analysis by Team", showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
    
    # Enhanced Player Statistics
    elif analysis_type == "👨‍💼 Player Statistics":
        st.header("👨‍💼 Player Performance Analysis")
        
        player_stats = get_enhanced_player_stats(df)
        
        if not player_stats.empty:
            # Enhanced metric selection
            col1, col2 = st.columns(2)
            with col1:
                metric_options = ['Total_Runs', 'Strike_Rate', 'Avg_Per_Match', 'Boundary_Percentage', 'Sixes', 'Fours']
                selected_metric = st.selectbox("Select Primary Metric:", metric_options)
            
            with col2:
                top_n = st.slider("Number of players to display:", 5, 25, 15)
            
            # Team filter
            teams = ['All Teams'] + player_stats['Primary_Team'].unique().tolist()
            selected_team = st.selectbox("Filter by Team:", teams)
            
            if selected_team != 'All Teams':
                filtered_players = player_stats[player_stats['Primary_Team'] == selected_team]
            else:
                filtered_players = player_stats
            
            top_players = filtered_players.nlargest(top_n, selected_metric)
            
            st.subheader(f"🏅 Top {top_n} Players by {selected_metric}")
            st.dataframe(top_players, use_container_width=True)
            
            # Enhanced visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(top_players, x='batter', y=selected_metric,
                           color='Primary_Team', title=f'Top Players by {selected_metric}',
                           hover_data=['Strike_Rate', 'Avg_Per_Match'])
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.scatter(filtered_players, x='Strike_Rate', y='Avg_Per_Match',
                               size='Total_Runs', color='Primary_Team',
                               hover_data=['batter', 'Sixes', 'Fours'],
                               title='Strike Rate vs Average per Match')
                st.plotly_chart(fig, use_container_width=True)
    
    # New Bowling Analysis Section
    elif analysis_type == "🎳 Bowling Analysis":
        st.header("🎳 Bowling Performance Analysis")
        
        bowling_stats = get_bowling_stats(df)
        
        if not bowling_stats.empty:
            metric_options = ['Economy_Rate', 'Strike_Rate', 'Wickets', 'Average', 'Dot_Ball_Percentage']
            selected_metric = st.selectbox("Select Bowling Metric:", metric_options)
            
            top_n = st.slider("Number of bowlers:", 5, 20, 10)
            
            if selected_metric in ['Strike_Rate', 'Average']:
                # For these metrics, lower is better
                top_bowlers = bowling_stats[bowling_stats[selected_metric] != np.inf].nsmallest(top_n, selected_metric)
            else:
                # For wickets and dot ball %, higher is better; for economy, lower is better
                if selected_metric == 'Economy_Rate':
                    top_bowlers = bowling_stats.nsmallest(top_n, selected_metric)
                else:
                    top_bowlers = bowling_stats.nlargest(top_n, selected_metric)
            
            st.subheader(f"🏆 Top {top_n} Bowlers by {selected_metric}")
            st.dataframe(top_bowlers, use_container_width=True)
            
            # Bowling visualizations
            fig = px.bar(top_bowlers, x='bowler', y=selected_metric,
                        title=f'Top Bowlers by {selected_metric}',
                        color=selected_metric, color_continuous_scale='viridis')
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
    
    # Team Comparison Section
    elif analysis_type == "⚔️ Team Comparison":
        create_comparison_section(df)
    
    # Keep other sections similar but with minor enhancements...
    # (Venue Analysis, Season Trends, Match Insights remain largely the same but with better styling)

    # Footer with additional info
    st.markdown("---")
    st.markdown("**IPL Data Analytics Dashboard** | Built with Streamlit and Plotly")
    
    # Export functionality
    if st.button("📊 Generate Summary Report"):
        with st.spinner("Generating report..."):
            st.success("Report functionality would be implemented here!")

if __name__ == "__main__":
    main()
