import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
from datetime import datetime

# Team aliases mapping
TEAM_ALIASES = {
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
    "Rising Pune Supergiant": "Rising Pune Supergiant",
    "Kings XI Punjab": "Punjab Kings",
    "Delhi Daredevils": "Delhi Capitals",
    "Pune Warriors": "Pune Warriors India",
}

def load_and_preprocess_data():
    """Load and preprocess IPL data"""
    # Load data
    df = pd.read_csv("IPL.csv")
    
    # Standardize team names
    df['batting_team'] = df['batting_team'].replace(TEAM_ALIASES)
    df['bowling_team'] = df['bowling_team'].replace(TEAM_ALIASES)
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Extract additional date features
    df['day_of_week'] = df['date'].dt.day_name()
    df['is_weekend'] = df['date'].dt.dayofweek >= 5
    
    # Handle missing values
    df = df.fillna({
        'wicket_kind': 'None',
        'player_out': 'None',
        'fielders': 'None',
        'extra_type': 'None',
        'review_batter': 'None',
        'team_reviewed': 'None',
        'review_decision': 'None',
        'umpire': 'None',
        'superover_winner': 'None',
        'method': 'Normal'
    })
    
    # Create match phase feature
    df['phase'] = np.select(
        [df['over'] < 6, (df['over'] >= 6) & (df['over'] < 16), df['over'] >= 16],
        ['Powerplay', 'Middle', 'Death'],
        default='Unknown'
    )
    
    # Create partnership identifier
    df['partnership'] = df.groupby(['match_id', 'innings', 'batting_team']).ngroup()
    
    return df

def create_derived_features(df):
    """Create derived features for analysis"""
    # Batting features
    batting_df = df.groupby(['match_id', 'innings', 'batting_team', 'batter']).agg(
        runs_batter=('runs_batter', 'sum'),
        balls_faced=('valid_ball', 'sum'),
        fours=('runs_batter', lambda x: (x == 4).sum()),
        sixes=('runs_batter', lambda x: (x == 6).sum()),
        dismissals=('player_out', lambda x: (x != 'None').sum())
    ).reset_index()
    
    # Calculate strike rate
    batting_df['strike_rate'] = np.where(
        batting_df['balls_faced'] > 0,
        (batting_df['runs_batter'] / batting_df['balls_faced']) * 100,
        0
    )
    
    # Bowling features
    bowling_df = df.groupby(['match_id', 'innings', 'bowling_team', 'bowler']).agg(
        runs_conceded=('runs_bowler', 'sum'),
        balls_bowled=('valid_ball', 'sum'),
        wickets=('wicket_kind', lambda x: (x != 'None').sum()),
        dots=('runs_total', lambda x: (x == 0).sum()),
        extras=('runs_extras', 'sum')
    ).reset_index()
    
    # Calculate economy rate
    bowling_df['economy_rate'] = np.where(
        bowling_df['balls_bowled'] > 0,
        (bowling_df['runs_conceded'] / bowling_df['balls_bowled']) * 6,
        0
    )
    
    # Team match features
    team_match_df = df.groupby(['match_id', 'innings', 'batting_team']).agg(
        total_runs=('runs_total', 'sum'),
        total_wickets=('wicket_kind', lambda x: (x != 'None').sum()),
        total_balls=('valid_ball', 'sum'),
        extras=('runs_extras', 'sum')
    ).reset_index()
    
    # Calculate run rate
    team_match_df['run_rate'] = np.where(
        team_match_df['total_balls'] > 0,
        (team_match_df['total_runs'] / team_match_df['total_balls']) * 6,
        0
    )
    
    return batting_df, bowling_df, team_match_df

def get_team_performance_data(df, season=None):
    """Get team performance data aggregated by season"""
    # Filter by season if specified
    if season:
        df = df[df['season'] == season]
    
    # Team wins
    team_wins = df[df['innings'] == 2].groupby(['season', 'batting_team', 'match_won_by']).size().reset_index(name='matches')
    team_wins = team_wins[team_wins['batting_team'] == team_wins['match_won_by']]
    team_wins = team_wins.groupby(['season', 'batting_team'])['matches'].sum().reset_index(name='wins')
    
    # Team matches
    team_matches = df.groupby(['season', 'batting_team'])['match_id'].nunique().reset_index(name='matches')
    
    # Merge and calculate win percentage
    team_perf = pd.merge(team_matches, team_wins, on=['season', 'batting_team'], how='left')
    team_perf['win_percentage'] = (team_perf['wins'] / team_perf['matches']) * 100
    team_perf['win_percentage'] = team_perf['win_percentage'].fillna(0)
    
    # Team runs and wickets
    team_stats = df.groupby(['season', 'batting_team']).agg(
        avg_runs=('runs_total', 'mean'),
        avg_wickets=('wicket_kind', lambda x: (x != 'None').mean()),
        avg_run_rate=('runs_total', lambda x: (x.sum() / x.count()) * 6)
    ).reset_index()
    
    # Merge all team stats
    team_perf = pd.merge(team_perf, team_stats, on=['season', 'batting_team'])
    
    return team_perf

def get_player_performance_data(df, role='batter', season=None):
    """Get player performance data"""
    if role == 'batter':
        player_df = batting_df.copy()
        if season:
            player_df = player_df[player_df['match_id'].isin(df[df['season'] == season]['match_id'])]
        
        player_stats = player_df.groupby('batter').agg(
            matches=('match_id', 'nunique'),
            innings=('innings', 'nunique'),
            runs=('runs_batter', 'sum'),
            balls=('balls_faced', 'sum'),
            dismissals=('dismissals', 'sum'),
            fours=('fours', 'sum'),
            sixes=('sixes', 'sum')
        ).reset_index()
        
        # Calculate averages and strike rate
        player_stats['average'] = np.where(
            player_stats['dismissals'] > 0,
            player_stats['runs'] / player_stats['dismissals'],
            player_stats['runs']
        )
        player_stats['strike_rate'] = np.where(
            player_stats['balls'] > 0,
            (player_stats['runs'] / player_stats['balls']) * 100,
            0
        )
        player_stats['fours_per_match'] = player_stats['fours'] / player_stats['matches']
        player_stats['sixes_per_match'] = player_stats['sixes'] / player_stats['matches']
        
    else:  # bowler
        player_df = bowling_df.copy()
        if season:
            player_df = player_df[player_df['match_id'].isin(df[df['season'] == season]['match_id'])]
        
        player_stats = player_df.groupby('bowler').agg(
            matches=('match_id', 'nunique'),
            innings=('innings', 'nunique'),
            wickets=('wickets', 'sum'),
            runs_conceded=('runs_conceded', 'sum'),
            balls_bowled=('balls_bowled', 'sum'),
            dots=('dots', 'sum'),
            extras=('extras', 'sum')
        ).reset_index()
        
        # Calculate economy and bowling average
        player_stats['economy_rate'] = np.where(
            player_stats['balls_bowled'] > 0,
            (player_stats['runs_conceded'] / player_stats['balls_bowled']) * 6,
            0
        )
        player_stats['bowling_average'] = np.where(
            player_stats['wickets'] > 0,
            player_stats['runs_conceded'] / player_stats['wickets'],
            player_stats['runs_conceded']
        )
        player_stats['dot_percentage'] = np.where(
            player_stats['balls_bowled'] > 0,
            (player_stats['dots'] / player_stats['balls_bowled']) * 100,
            0
        )
    
    return player_stats

def get_venue_analysis_data(df):
    """Get venue analysis data"""
    venue_stats = df.groupby('venue').agg(
        matches=('match_id', 'nunique'),
        avg_runs=('runs_total', 'mean'),
        avg_wickets=('wicket_kind', lambda x: (x != 'None').mean()),
        avg_first_innings_score=('runs_total', lambda x: x[df[df['match_id'].isin(x.index)]['innings'] == 1].mean()),
        avg_second_innings_score=('runs_total', lambda x: x[df[df['match_id'].isin(x.index)]['innings'] == 2].mean())
    ).reset_index()
    
    # Calculate win percentage for teams batting first
    first_innings_wins = df[df['innings'] == 1].groupby(['venue', 'batting_team', 'match_won_by']).size().reset_index(name='matches')
    first_innings_wins = first_innings_wins[first_innings_wins['batting_team'] == first_innings_wins['match_won_by']]
    first_innings_wins = first_innings_wins.groupby('venue')['matches'].sum().reset_index(name='first_innings_wins')
    
    # Total first innings matches
    first_innings_matches = df[df['innings'] == 1].groupby('venue')['match_id'].nunique().reset_index(name='first_innings_matches')
    
    # Merge and calculate win percentage
    venue_stats = pd.merge(venue_stats, first_innings_matches, on='venue')
    venue_stats = pd.merge(venue_stats, first_innings_wins, on='venue', how='left')
    venue_stats['first_innings_win_percentage'] = (venue_stats['first_innings_wins'] / venue_stats['first_innings_matches']) * 100
    venue_stats['first_innings_win_percentage'] = venue_stats['first_innings_win_percentage'].fillna(0)
    
    return venue_stats

def get_phase_analysis_data(df):
    """Get phase analysis data"""
    phase_stats = df.groupby(['batting_team', 'phase']).agg(
        runs=('runs_total', 'sum'),
        balls=('valid_ball', 'sum'),
        wickets=('wicket_kind', lambda x: (x != 'None').sum()),
        dots=('runs_total', lambda x: (x == 0).sum()),
        boundaries=('runs_batter', lambda x: ((x == 4) | (x == 6)).sum())
    ).reset_index()
    
    # Calculate metrics
    phase_stats['run_rate'] = (phase_stats['runs'] / phase_stats['balls']) * 6
    phase_stats['wicket_rate'] = (phase_stats['wickets'] / phase_stats['balls']) * 6
    phase_stats['dot_percentage'] = (phase_stats['dots'] / phase_stats['balls']) * 100
    phase_stats['boundary_percentage'] = (phase_stats['boundaries'] / phase_stats['balls']) * 100
    
    return phase_stats

# Load and preprocess data
df = load_and_preprocess_data()
batting_df, bowling_df, team_match_df = create_derived_features(df)

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], suppress_callback_exceptions=True)
server = app.server

# Get unique values for filters
seasons = sorted(df['season'].unique())
teams = sorted(df['batting_team'].unique())
venues = sorted(df['venue'].unique())
players_batting = sorted(batting_df['batter'].unique())
players_bowling = sorted(bowling_df['bowler'].unique())

# Define the layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("IPL Data Analysis Platform", className="text-center text-primary mb-4"), width=12)
    ]),
    
    dbc.Tabs([
        dbc.Tab(label="Team Performance", tab_id="team-tab"),
        dbc.Tab(label="Player Performance", tab_id="player-tab"),
        dbc.Tab(label="Match Analysis", tab_id="match-tab"),
        dbc.Tab(label="Historical Trends", tab_id="historical-tab"),
        dbc.Tab(label="Advanced Analytics", tab_id="advanced-tab"),
    ], id="tabs", active_tab="team-tab", className="mb-4"),
    
    # Tab content
    html.Div(id="tab-content", className="mt-4"),
    
], fluid=True)

# Team Performance Tab Layout
team_tab_layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Filters"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Season"),
                            dcc.Dropdown(
                                id="team-season-dropdown",
                                options=[{'label': s, 'value': s} for s in seasons],
                                value=seasons[-1],
                                clearable=False
                            )
                        ], width=6),
                        dbc.Col([
                            html.Label("Team"),
                            dcc.Dropdown(
                                id="team-dropdown",
                                options=[{'label': t, 'value': t} for t in teams],
                                value=teams[0],
                                clearable=False
                            )
                        ], width=6),
                    ]),
                ])
            ])
        ], width=12),
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Team Performance Overview"),
                dbc.CardBody([
                    dcc.Graph(id="team-performance-chart")
                ])
            ])
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Win Percentage by Season"),
                dbc.CardBody([
                    dcc.Graph(id="team-win-percentage-chart")
                ])
            ])
        ], width=6),
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Phase-wise Performance"),
                dbc.CardBody([
                    dcc.Graph(id="team-phase-chart")
                ])
            ])
        ], width=12),
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Head-to-Head Record"),
                dbc.CardBody([
                    dcc.Graph(id="team-h2h-chart")
                ])
            ])
        ], width=12),
    ], className="mb-4"),
])

# Player Performance Tab Layout
player_tab_layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Filters"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Player Type"),
                            dcc.RadioItems(
                                id="player-type-radio",
                                options=[
                                    {'label': 'Batter', 'value': 'batter'},
                                    {'label': 'Bowler', 'value': 'bowler'}
                                ],
                                value='batter',
                                inline=True
                            )
                        ], width=6),
                        dbc.Col([
                            html.Label("Season"),
                            dcc.Dropdown(
                                id="player-season-dropdown",
                                options=[{'label': s, 'value': s} for s in seasons],
                                value=seasons[-1],
                                clearable=False
                            )
                        ], width=6),
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Player"),
                            dcc.Dropdown(
                                id="player-dropdown",
                                options=[{'label': p, 'value': p} for p in players_batting],
                                value=players_batting[0],
                                clearable=False
                            )
                        ], width=12),
                    ], className="mt-3"),
                ])
            ])
        ], width=12),
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Player Performance Overview"),
                dbc.CardBody([
                    dcc.Graph(id="player-performance-chart")
                ])
            ])
        ], width=12),
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Season-wise Performance"),
                dbc.CardBody([
                    dcc.Graph(id="player-season-chart")
                ])
            ])
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Performance Against Teams"),
                dbc.CardBody([
                    dcc.Graph(id="player-vs-team-chart")
                ])
            ])
        ], width=6),
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Performance by Phase"),
                dbc.CardBody([
                    dcc.Graph(id="player-phase-chart")
                ])
            ])
        ], width=12),
    ], className="mb-4"),
])

# Match Analysis Tab Layout
match_tab_layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Filters"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Season"),
                            dcc.Dropdown(
                                id="match-season-dropdown",
                                options=[{'label': s, 'value': s} for s in seasons],
                                value=seasons[-1],
                                clearable=False
                            )
                        ], width=6),
                        dbc.Col([
                            html.Label("Venue"),
                            dcc.Dropdown(
                                id="match-venue-dropdown",
                                options=[{'label': v, 'value': v} for v in venues],
                                value=venues[0],
                                clearable=False
                            )
                        ], width=6),
                    ]),
                ])
            ])
        ], width=12),
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Venue Analysis"),
                dbc.CardBody([
                    dcc.Graph(id="venue-analysis-chart")
                ])
            ])
        ], width=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Toss Impact"),
                dbc.CardBody([
                    dcc.Graph(id="toss-impact-chart")
                ])
            ])
        ], width=6),
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Match Outcome Distribution"),
                dbc.CardBody([
                    dcc.Graph(id="match-outcome-chart")
                ])
            ])
        ], width=12),
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Score Distribution"),
                dbc.CardBody([
                    dcc.Graph(id="score-distribution-chart")
                ])
            ])
        ], width=12),
    ], className="mb-4"),
])

# Historical Trends Tab Layout
historical_tab_layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Filters"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Metric"),
                            dcc.Dropdown(
                                id="historical-metric-dropdown",
                                options=[
                                    {'label': 'Average Runs per Match', 'value': 'avg_runs'},
                                    {'label': 'Average Wickets per Match', 'value': 'avg_wickets'},
                                    {'label': 'Run Rate', 'value': 'run_rate'},
                                    {'label': 'Boundary Percentage', 'value': 'boundary_percentage'}
                                ],
                                value='avg_runs',
                                clearable=False
                            )
                        ], width=6),
                        dbc.Col([
                            html.Label("Team"),
                            dcc.Dropdown(
                                id="historical-team-dropdown",
                                options=[{'label': t, 'value': t} for t in teams],
                                value=None,
                                placeholder="All Teams",
                                clearable=True
                            )
                        ], width=6),
                    ]),
                ])
            ])
        ], width=12),
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Historical Trends"),
                dbc.CardBody([
                    dcc.Graph(id="historical-trends-chart")
                ])
            ])
        ], width=12),
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Season Comparison"),
                dbc.CardBody([
                    dcc.Graph(id="season-comparison-chart")
                ])
            ])
        ], width=12),
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Team Evolution"),
                dbc.CardBody([
                    dcc.Graph(id="team-evolution-chart")
                ])
            ])
        ], width=12),
    ], className="mb-4"),
])

# Advanced Analytics Tab Layout
advanced_tab_layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Filters"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Analysis Type"),
                            dcc.Dropdown(
                                id="advanced-analysis-dropdown",
                                options=[
                                    {'label': 'Partnership Analysis', 'value': 'partnership'},
                                    {'label': 'Pressure Performance', 'value': 'pressure'},
                                    {'label': 'Momentum Shifts', 'value': 'momentum'},
                                    {'label': 'Player Impact', 'value': 'impact'}
                                ],
                                value='partnership',
                                clearable=False
                            )
                        ], width=6),
                        dbc.Col([
                            html.Label("Season"),
                            dcc.Dropdown(
                                id="advanced-season-dropdown",
                                options=[{'label': s, 'value': s} for s in seasons],
                                value=seasons[-1],
                                clearable=False
                            )
                        ], width=6),
                    ]),
                ])
            ])
        ], width=12),
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Advanced Analysis"),
                dbc.CardBody([
                    dcc.Graph(id="advanced-analysis-chart")
                ])
            ])
        ], width=12),
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Key Insights"),
                dbc.CardBody([
                    html.Div(id="advanced-insights")
                ])
            ])
        ], width=12),
    ], className="mb-4"),
])

# Callback for tab content
@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "active_tab")
)
def render_tab_content(active_tab):
    if active_tab == "team-tab":
        return team_tab_layout
    elif active_tab == "player-tab":
        return player_tab_layout
    elif active_tab == "match-tab":
        return match_tab_layout
    elif active_tab == "historical-tab":
        return historical_tab_layout
    elif active_tab == "advanced-tab":
        return advanced_tab_layout
    return html.P("This shouldn't ever be displayed...")

# Team Performance Tab Callbacks
@app.callback(
    Output("team-performance-chart", "figure"),
    Input("team-season-dropdown", "value"),
    Input("team-dropdown", "value")
)
def update_team_performance_chart(season, team):
    team_perf = get_team_performance_data(df, season)
    team_data = team_perf[team_perf['batting_team'] == team]
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Win Percentage', 'Average Runs', 'Average Wickets', 'Run Rate'),
        specs=[[{"type": "indicator"}, {"type": "indicator"}],
               [{"type": "indicator"}, {"type": "indicator"}]]
    )
    
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=team_data['win_percentage'].values[0],
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Win %"},
        delta={'reference': 50},
        gauge={'axis': {'range': [None, 100]},
               'bar': {'color': "darkblue"},
               'steps': [
                   {'range': [0, 50], 'color': "lightgray"},
                   {'range': [50, 80], 'color': "gray"},
                   {'range': [80, 100], 'color': "lightgreen"}],
               'threshold': {'line': {'color': "red", 'width': 4},
                            'thickness': 0.75, 'value': 90}}
    ), row=1, col=1)
    
    fig.add_trace(go.Indicator(
        mode="number+delta",
        value=team_data['avg_runs'].values[0],
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Avg Runs"},
        delta={'reference': team_perf['avg_runs'].mean()}
    ), row=1, col=2)
    
    fig.add_trace(go.Indicator(
        mode="number+delta",
        value=team_data['avg_wickets'].values[0],
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Avg Wickets"},
        delta={'reference': team_perf['avg_wickets'].mean()}
    ), row=2, col=1)
    
    fig.add_trace(go.Indicator(
        mode="number+delta",
        value=team_data['avg_run_rate'].values[0],
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Run Rate"},
        delta={'reference': team_perf['avg_run_rate'].mean()}
    ), row=2, col=2)
    
    fig.update_layout(height=400)
    return fig

@app.callback(
    Output("team-win-percentage-chart", "figure"),
    Input("team-dropdown", "value")
)
def update_team_win_percentage_chart(team):
    team_perf = get_team_performance_data(df)
    team_data = team_perf[team_perf['batting_team'] == team]
    
    fig = px.line(
        team_data,
        x='season',
        y='win_percentage',
        title=f'{team} Win Percentage by Season',
        labels={'win_percentage': 'Win %', 'season': 'Season'},
        markers=True
    )
    
    fig.update_layout(
        yaxis_title="Win Percentage",
        xaxis_title="Season",
        hovermode='x unified'
    )
    
    return fig

@app.callback(
    Output("team-phase-chart", "figure"),
    Input("team-season-dropdown", "value"),
    Input("team-dropdown", "value")
)
def update_team_phase_chart(season, team):
    phase_data = get_phase_analysis_data(df)
    if season:
        phase_data = phase_data[phase_data['match_id'].isin(df[df['season'] == season]['match_id'])]
    
    team_phase_data = phase_data[phase_data['batting_team'] == team]
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Run Rate', 'Wicket Rate', 'Dot Ball %', 'Boundary %'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    phases = ['Powerplay', 'Middle', 'Death']
    colors = ['blue', 'green', 'red']
    
    for i, phase in enumerate(phases):
        phase_df = team_phase_data[team_phase_data['phase'] == phase]
        if not phase_df.empty:
            fig.add_trace(
                go.Bar(x=[phase], y=[phase_df['run_rate'].values[0]], name=phase, marker_color=colors[i]),
                row=1, col=1
            )
            fig.add_trace(
                go.Bar(x=[phase], y=[phase_df['wicket_rate'].values[0]], name=phase, marker_color=colors[i]),
                row=1, col=2
            )
            fig.add_trace(
                go.Bar(x=[phase], y=[phase_df['dot_percentage'].values[0]], name=phase, marker_color=colors[i]),
                row=2, col=1
            )
            fig.add_trace(
                go.Bar(x=[phase], y=[phase_df['boundary_percentage'].values[0]], name=phase, marker_color=colors[i]),
                row=2, col=2
            )
    
    fig.update_layout(height=500, showlegend=False)
    return fig

@app.callback(
    Output("team-h2h-chart", "figure"),
    Input("team-season-dropdown", "value"),
    Input("team-dropdown", "value")
)
def update_team_h2h_chart(season, team):
    # Filter data by season if specified
    if season:
        match_df = df[df['season'] == season]
    else:
        match_df = df.copy()
    
    # Get matches involving the selected team
    team_matches = match_df[(match_df['batting_team'] == team) | (match_df['bowling_team'] == team)]
    
    # Get unique match IDs
    match_ids = team_matches['match_id'].unique()
    
    # Get match results
    match_results = match_df[match_df['match_id'].isin(match_ids)][['match_id', 'match_won_by']].drop_duplicates()
    
    # Get opponent teams
    team_matches_summary = team_matches.groupby('match_id').agg(
        team1=('batting_team', 'first'),
        team2=('bowling_team', 'first')
    ).reset_index()
    
    # Merge with results
    match_summary = pd.merge(team_matches_summary, match_results, on='match_id')
    
    # Determine opponent and result
    match_summary['opponent'] = np.where(
        match_summary['team1'] == team,
        match_summary['team2'],
        match_summary['team1']
    )
    
    match_summary['result'] = np.where(
        match_summary['match_won_by'] == team,
        'Win',
        'Loss'
    )
    
    # Count wins and losses against each opponent
    h2h = match_summary.groupby(['opponent', 'result']).size().unstack().fillna(0)
    h2h = h2h.reset_index()
    
    # Create bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=h2h['opponent'],
        y=h2h['Win'],
        name='Wins',
        marker_color='green'
    ))
    
    fig.add_trace(go.Bar(
        x=h2h['opponent'],
        y=h2h['Loss'],
        name='Losses',
        marker_color='red'
    ))
    
    fig.update_layout(
        title=f'{team} Head-to-Head Record',
        xaxis_title='Opponent',
        yaxis_title='Matches',
        barmode='stack'
    )
    
    return fig

# Player Performance Tab Callbacks
@app.callback(
    Output("player-dropdown", "options"),
    Input("player-type-radio", "value")
)
def update_player_dropdown(player_type):
    if player_type == 'batter':
        return [{'label': p, 'value': p} for p in players_batting]
    else:
        return [{'label': p, 'value': p} for p in players_bowling]

@app.callback(
    Output("player-performance-chart", "figure"),
    Input("player-type-radio", "value"),
    Input("player-season-dropdown", "value"),
    Input("player-dropdown", "value")
)
def update_player_performance_chart(player_type, season, player):
    if player_type == 'batter':
        player_data = get_player_performance_data(df, 'batter', season)
        player_stats = player_data[player_data['batter'] == player]
        
        if player_stats.empty:
            return go.Figure()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Runs', 'Average', 'Strike Rate', 'Boundary %'),
            specs=[[{"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "indicator"}, {"type": "indicator"}]]
        )
        
        fig.add_trace(go.Indicator(
            mode="number+delta",
            value=player_stats['runs'].values[0],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Runs"},
            delta={'reference': player_data['runs'].quantile(0.75)}
        ), row=1, col=1)
        
        fig.add_trace(go.Indicator(
            mode="number+delta",
            value=player_stats['average'].values[0],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Average"},
            delta={'reference': player_data['average'].quantile(0.75)}
        ), row=1, col=2)
        
        fig.add_trace(go.Indicator(
            mode="number+delta",
            value=player_stats['strike_rate'].values[0],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Strike Rate"},
            delta={'reference': player_data['strike_rate'].quantile(0.75)}
        ), row=2, col=1)
        
        boundary_pct = ((player_stats['fours'].values[0] + player_stats['sixes'].values[0]) / player_stats['balls'].values[0]) * 100
        fig.add_trace(go.Indicator(
            mode="number+delta",
            value=boundary_pct,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Boundary %"},
            delta={'reference': player_data.apply(lambda x: ((x['fours'] + x['sixes']) / x['balls']) * 100, axis=1).quantile(0.75)}
        ), row=2, col=2)
        
        fig.update_layout(height=400)
        return fig
    
    else:  # bowler
        player_data = get_player_performance_data(df, 'bowler', season)
        player_stats = player_data[player_data['bowler'] == player]
        
        if player_stats.empty:
            return go.Figure()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Wickets', 'Economy Rate', 'Bowling Average', 'Dot Ball %'),
            specs=[[{"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "indicator"}, {"type": "indicator"}]]
        )
        
        fig.add_trace(go.Indicator(
            mode="number+delta",
            value=player_stats['wickets'].values[0],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Wickets"},
            delta={'reference': player_data['wickets'].quantile(0.75)}
        ), row=1, col=1)
        
        fig.add_trace(go.Indicator(
            mode="number+delta",
            value=player_stats['economy_rate'].values[0],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Economy Rate"},
            delta={'reference': player_data['economy_rate'].quantile(0.25)}
        ), row=1, col=2)
        
        fig.add_trace(go.Indicator(
            mode="number+delta",
            value=player_stats['bowling_average'].values[0],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Bowling Average"},
            delta={'reference': player_data['bowling_average'].quantile(0.25)}
        ), row=2, col=1)
        
        fig.add_trace(go.Indicator(
            mode="number+delta",
            value=player_stats['dot_percentage'].values[0],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Dot Ball %"},
            delta={'reference': player_data['dot_percentage'].quantile(0.75)}
        ), row=2, col=2)
        
        fig.update_layout(height=400)
        return fig

@app.callback(
    Output("player-season-chart", "figure"),
    Input("player-type-radio", "value"),
    Input("player-dropdown", "value")
)
def update_player_season_chart(player_type, player):
    if player_type == 'batter':
        player_data = batting_df[batting_df['batter'] == player]
        season_stats = player_data.groupby('season').agg(
            runs=('runs_batter', 'sum'),
            matches=('match_id', 'nunique'),
            average=('runs_batter', lambda x: x.sum() / player_data[player_data['season'] == x.index[0]]['dismissals'].sum() if player_data[player_data['season'] == x.index[0]]['dismissals'].sum() > 0 else x.sum()),
            strike_rate=('runs_batter', lambda x: (x.sum() / player_data[player_data['season'] == x.index[0]]['balls_faced'].sum()) * 100 if player_data[player_data['season'] == x.index[0]]['balls_faced'].sum() > 0 else 0)
        ).reset_index()
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            subplot_titles=('Runs by Season', 'Strike Rate by Season'),
            vertical_spacing=0.1
        )
        
        fig.add_trace(
            go.Bar(x=season_stats['season'], y=season_stats['runs'], name='Runs'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=season_stats['season'], y=season_stats['strike_rate'], name='Strike Rate', mode='lines+markers'),
            row=2, col=1
        )
        
        fig.update_layout(height=500)
        return fig
    
    else:  # bowler
        player_data = bowling_df[bowling_df['bowler'] == player]
        season_stats = player_data.groupby('season').agg(
            wickets=('wickets', 'sum'),
            matches=('match_id', 'nunique'),
            economy_rate=('runs_conceded', lambda x: (x.sum() / player_data[player_data['season'] == x.index[0]]['balls_bowled'].sum()) * 6 if player_data[player_data['season'] == x.index[0]]['balls_bowled'].sum() > 0 else 0),
            bowling_average=('runs_conceded', lambda x: x.sum() / player_data[player_data['season'] == x.index[0]]['wickets'].sum() if player_data[player_data['season'] == x.index[0]]['wickets'].sum() > 0 else x.sum())
        ).reset_index()
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            subplot_titles=('Wickets by Season', 'Economy Rate by Season'),
            vertical_spacing=0.1
        )
        
        fig.add_trace(
            go.Bar(x=season_stats['season'], y=season_stats['wickets'], name='Wickets'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=season_stats['season'], y=season_stats['economy_rate'], name='Economy Rate', mode='lines+markers'),
            row=2, col=1
        )
        
        fig.update_layout(height=500)
        return fig

@app.callback(
    Output("player-vs-team-chart", "figure"),
    Input("player-type-radio", "value"),
    Input("player-season-dropdown", "value"),
    Input("player-dropdown", "value")
)
def update_player_vs_team_chart(player_type, season, player):
    if player_type == 'batter':
        player_data = batting_df[batting_df['batter'] == player]
        if season:
            player_data = player_data[player_data['match_id'].isin(df[df['season'] == season]['match_id'])]
        
        # Get opponent team (bowling team)
        player_data = player_data.merge(
            df[['match_id', 'bowling_team']].drop_duplicates(),
            on='match_id'
        )
        
        vs_team_stats = player_data.groupby('bowling_team').agg(
            runs=('runs_batter', 'sum'),
            balls=('balls_faced', 'sum'),
            dismissals=('dismissals', 'sum')
        ).reset_index()
        
        vs_team_stats['average'] = np.where(
            vs_team_stats['dismissals'] > 0,
            vs_team_stats['runs'] / vs_team_stats['dismissals'],
            vs_team_stats['runs']
        )
        
        vs_team_stats['strike_rate'] = np.where(
            vs_team_stats['balls'] > 0,
            (vs_team_stats['runs'] / vs_team_stats['balls']) * 100,
            0
        )
        
        fig = px.bar(
            vs_team_stats,
            x='bowling_team',
            y='runs',
            title=f'{player} - Runs Against Teams',
            labels={'runs': 'Runs', 'bowling_team': 'Opponent Team'},
            hover_data=['average', 'strike_rate']
        )
        
        return fig
    
    else:  # bowler
        player_data = bowling_df[bowling_df['bowler'] == player]
        if season:
            player_data = player_data[player_data['match_id'].isin(df[df['season'] == season]['match_id'])]
        
        # Get opponent team (batting team)
        player_data = player_data.merge(
            df[['match_id', 'batting_team']].drop_duplicates(),
            on='match_id'
        )
        
        vs_team_stats = player_data.groupby('batting_team').agg(
            wickets=('wickets', 'sum'),
            runs_conceded=('runs_conceded', 'sum'),
            balls_bowled=('balls_bowled', 'sum')
        ).reset_index()
        
        vs_team_stats['economy_rate'] = np.where(
            vs_team_stats['balls_bowled'] > 0,
            (vs_team_stats['runs_conceded'] / vs_team_stats['balls_bowled']) * 6,
            0
        )
        
        vs_team_stats['bowling_average'] = np.where(
            vs_team_stats['wickets'] > 0,
            vs_team_stats['runs_conceded'] / vs_team_stats['wickets'],
            vs_team_stats['runs_conceded']
        )
        
        fig = px.bar(
            vs_team_stats,
            x='batting_team',
            y='wickets',
            title=f'{player} - Wickets Against Teams',
            labels={'wickets': 'Wickets', 'batting_team': 'Opponent Team'},
            hover_data=['economy_rate', 'bowling_average']
        )
        
        return fig

@app.callback(
    Output("player-phase-chart", "figure"),
    Input("player-type-radio", "value"),
    Input("player-season-dropdown", "value"),
    Input("player-dropdown", "value")
)
def update_player_phase_chart(player_type, season, player):
    if player_type == 'batter':
        player_data = df[df['batter'] == player]
        if season:
            player_data = player_data[player_data['season'] == season]
        
        phase_stats = player_data.groupby('phase').agg(
            runs=('runs_batter', 'sum'),
            balls=('valid_ball', 'sum'),
            dismissals=('player_out', lambda x: (x != 'None').sum())
        ).reset_index()
        
        phase_stats['strike_rate'] = np.where(
            phase_stats['balls'] > 0,
            (phase_stats['runs'] / phase_stats['balls']) * 100,
            0
        )
        
        phase_stats['average'] = np.where(
            phase_stats['dismissals'] > 0,
            phase_stats['runs'] / phase_stats['dismissals'],
            phase_stats['runs']
        )
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            subplot_titles=('Strike Rate by Phase', 'Average by Phase'),
            vertical_spacing=0.1
        )
        
        fig.add_trace(
            go.Bar(x=phase_stats['phase'], y=phase_stats['strike_rate'], name='Strike Rate'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=phase_stats['phase'], y=phase_stats['average'], name='Average'),
            row=2, col=1
        )
        
        fig.update_layout(height=500)
        return fig
    
    else:  # bowler
        player_data = df[df['bowler'] == player]
        if season:
            player_data = player_data[player_data['season'] == season]
        
        phase_stats = player_data.groupby('phase').agg(
            wickets=('wicket_kind', lambda x: (x != 'None').sum()),
            runs_conceded=('runs_bowler', 'sum'),
            balls_bowled=('valid_ball', 'sum')
        ).reset_index()
        
        phase_stats['economy_rate'] = np.where(
            phase_stats['balls_bowled'] > 0,
            (phase_stats['runs_conceded'] / phase_stats['balls_bowled']) * 6,
            0
        )
        
        phase_stats['bowling_average'] = np.where(
            phase_stats['wickets'] > 0,
            phase_stats['runs_conceded'] / phase_stats['wickets'],
            phase_stats['runs_conceded']
        )
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            subplot_titles=('Economy Rate by Phase', 'Bowling Average by Phase'),
            vertical_spacing=0.1
        )
        
        fig.add_trace(
            go.Bar(x=phase_stats['phase'], y=phase_stats['economy_rate'], name='Economy Rate'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=phase_stats['phase'], y=phase_stats['bowling_average'], name='Bowling Average'),
            row=2, col=1
        )
        
        fig.update_layout(height=500)
        return fig

# Match Analysis Tab Callbacks
@app.callback(
    Output("venue-analysis-chart", "figure"),
    Input("match-season-dropdown", "value"),
    Input("match-venue-dropdown", "value")
)
def update_venue_analysis_chart(season, venue):
    venue_data = get_venue_analysis_data(df)
    if season:
        venue_data = venue_data[venue_data['match_id'].isin(df[df['season'] == season]['match_id'])]
    
    venue_stats = venue_data[venue_data['venue'] == venue]
    
    if venue_stats.empty:
        return go.Figure()
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Matches', 'Avg First Innings Score', 'Avg Second Innings Score', 'First Innings Win %'),
        specs=[[{"type": "indicator"}, {"type": "indicator"}],
               [{"type": "indicator"}, {"type": "indicator"}]]
    )
    
    fig.add_trace(go.Indicator(
        mode="number",
        value=venue_stats['matches'].values[0],
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Matches"}
    ), row=1, col=1)
    
    fig.add_trace(go.Indicator(
        mode="number",
        value=venue_stats['avg_first_innings_score'].values[0],
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Avg 1st Innings Score"}
    ), row=1, col=2)
    
    fig.add_trace(go.Indicator(
        mode="number",
        value=venue_stats['avg_second_innings_score'].values[0],
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Avg 2nd Innings Score"}
    ), row=2, col=1)
    
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=venue_stats['first_innings_win_percentage'].values[0],
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "1st Innings Win %"},
        gauge={'axis': {'range': [None, 100]},
               'bar': {'color': "darkblue"},
               'steps': [
                   {'range': [0, 40], 'color': "lightgray"},
                   {'range': [40, 60], 'color': "gray"},
                   {'range': [60, 100], 'color': "lightgreen"}],
               'threshold': {'line': {'color': "red", 'width': 4},
                            'thickness': 0.75, 'value': 80}}
    ), row=2, col=2)
    
    fig.update_layout(height=400)
    return fig

@app.callback(
    Output("toss-impact-chart", "figure"),
    Input("match-season-dropdown", "value"),
    Input("match-venue-dropdown", "value")
)
def update_toss_impact_chart(season, venue):
    match_df = df.copy()
    if season:
        match_df = match_df[match_df['season'] == season]
    if venue:
        match_df = match_df[match_df['venue'] == venue]
    
    # Get unique matches
    matches = match_df[['match_id', 'toss_winner', 'toss_decision', 'match_won_by']].drop_duplicates()
    
    # Determine if toss winner won the match
    matches['toss_winner_won'] = matches['toss_winner'] == matches['match_won_by']
    
    # Group by toss decision
    toss_impact = matches.groupby('toss_decision')['toss_winner_won'].agg(['count', 'sum']).reset_index()
    toss_impact['win_percentage'] = (toss_impact['sum'] / toss_impact['count']) * 100
    
    fig = px.bar(
        toss_impact,
        x='toss_decision',
        y='win_percentage',
        title='Toss Impact on Match Outcome',
        labels={'win_percentage': 'Win %', 'toss_decision': 'Toss Decision'},
        text=toss_impact['win_percentage'].apply(lambda x: f'{x:.1f}%')
    )
    
    fig.update_traces(texttemplate='%{text}', textposition='outside')
    fig.update_layout(yaxis_title="Win Percentage")
    
    return fig

@app.callback(
    Output("match-outcome-chart", "figure"),
    Input("match-season-dropdown", "value"),
    Input("match-venue-dropdown", "value")
)
def update_match_outcome_chart(season, venue):
    match_df = df.copy()
    if season:
        match_df = match_df[match_df['season'] == season]
    if venue:
        match_df = match_df[match_df['venue'] == venue]
    
    # Get unique matches
    matches = match_df[['match_id', 'win_outcome']].drop_duplicates()
    
    # Count outcomes
    outcomes = matches['win_outcome'].value_counts().reset_index()
    outcomes.columns = ['win_outcome', 'count']
    
    fig = px.pie(
        outcomes,
        values='count',
        names='win_outcome',
        title='Match Outcome Distribution',
        hole=0.3
    )
    
    return fig

@app.callback(
    Output("score-distribution-chart", "figure"),
    Input("match-season-dropdown", "value"),
    Input("match-venue-dropdown", "value")
)
def update_score_distribution_chart(season, venue):
    match_df = df.copy()
    if season:
        match_df = match_df[match_df['season'] == season]
    if venue:
        match_df = match_df[match_df['venue'] == venue]
    
    # Get first innings scores
    first_innings = match_df[match_df['innings'] == 1].groupby('match_id')['team_runs'].max().reset_index()
    
    fig = px.histogram(
        first_innings,
        x='team_runs',
        title='First Innings Score Distribution',
        nbins=20,
        labels={'team_runs': 'Score', 'count': 'Frequency'}
    )
    
    fig.update_layout(bargap=0.1)
    return fig

# Historical Trends Tab Callbacks
@app.callback(
    Output("historical-trends-chart", "figure"),
    Input("historical-metric-dropdown", "value"),
    Input("historical-team-dropdown", "value")
)
def update_historical_trends_chart(metric, team):
    team_perf = get_team_performance_data(df)
    
    if team:
        team_perf = team_perf[team_perf['batting_team'] == team]
    
    fig = px.line(
        team_perf,
        x='season',
        y=metric,
        title=f'{metric.replace("_", " ").title()} Over Time',
        labels={metric: metric.replace("_", " ").title(), 'season': 'Season'},
        color='batting_team' if not team else None,
        markers=True
    )
    
    fig.update_layout(
        yaxis_title=metric.replace("_", " ").title(),
        xaxis_title="Season",
        hovermode='x unified'
    )
    
    return fig

@app.callback(
    Output("season-comparison-chart", "figure"),
    Input("historical-metric-dropdown", "value")
)
def update_season_comparison_chart(metric):
    team_perf = get_team_performance_data(df)
    
    # Get top 5 teams by the selected metric in the latest season
    latest_season = team_perf['season'].max()
    top_teams = team_perf[team_perf['season'] == latest_season].nlargest(5, metric)['batting_team'].tolist()
    
    # Filter data for top teams
    top_team_data = team_perf[team_perf['batting_team'].isin(top_teams)]
    
    fig = px.bar(
        top_team_data,
        x='season',
        y=metric,
        color='batting_team',
        title=f'Top 5 Teams by {metric.replace("_", " ").title()}',
        labels={metric: metric.replace("_", " ").title(), 'season': 'Season'},
        barmode='group'
    )
    
    return fig

@app.callback(
    Output("team-evolution-chart", "figure"),
    Input("historical-team-dropdown", "value")
)
def update_team_evolution_chart(team):
    if not team:
        return go.Figure()
    
    team_perf = get_team_performance_data(df)
    team_data = team_perf[team_perf['batting_team'] == team]
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Win Percentage', 'Average Runs', 'Average Wickets', 'Run Rate'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    fig.add_trace(
        go.Scatter(x=team_data['season'], y=team_data['win_percentage'], name='Win %', mode='lines+markers'),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=team_data['season'], y=team_data['avg_runs'], name='Avg Runs', mode='lines+markers'),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(x=team_data['season'], y=team_data['avg_wickets'], name='Avg Wickets', mode='lines+markers'),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=team_data['season'], y=team_data['avg_run_rate'], name='Run Rate', mode='lines+markers'),
        row=2, col=2
    )
    
    fig.update_layout(height=600, title_text=f"{team} Team Evolution")
    return fig

# Advanced Analytics Tab Callbacks
@app.callback(
    Output("advanced-analysis-chart", "figure"),
    Input("advanced-analysis-dropdown", "value"),
    Input("advanced-season-dropdown", "value")
)
def update_advanced_analysis_chart(analysis_type, season):
    match_df = df.copy()
    if season:
        match_df = match_df[match_df['season'] == season]
    
    if analysis_type == 'partnership':
        # Get partnership data
        partnership_data = match_df.groupby(['match_id', 'innings', 'partnership', 'batting_team']).agg(
            runs=('runs_total', 'sum'),
            balls=('valid_ball', 'sum'),
            wickets=('wicket_kind', lambda x: (x != 'None').sum())
        ).reset_index()
        
        # Filter partnerships with at least 20 runs
        partnership_data = partnership_data[partnership_data['runs'] >= 20]
        
        # Get top partnerships
        top_partnerships = partnership_data.nlargest(20, 'runs')
        
        # Create partnership identifier
        top_partnerships['partnership_id'] = top_partnerships.apply(
            lambda x: f"{x['match_id']} - {x['batting_team']} - Inn {x['innings']}", axis=1
        )
        
        fig = px.bar(
            top_partnerships,
            x='partnership_id',
            y='runs',
            title='Top Partnerships',
            labels={'runs': 'Runs', 'partnership_id': 'Partnership'},
            hover_data=['balls', 'wickets']
        )
        
        fig.update_layout(xaxis_title="Partnership", yaxis_title="Runs")
        return fig
    
    elif analysis_type == 'pressure':
        # Define pressure situations (last 4 overs of chase with close game)
        chase_matches = match_df[match_df['innings'] == 2].copy()
        chase_matches['target'] = chase_matches.groupby('match_id')['runs_target'].transform('first')
        chase_matches['remaining_runs'] = chase_matches['target'] - chase_matches.groupby('match_id')['team_runs'].transform('cumsum')
        chase_matches['remaining_balls'] = chase_matches.groupby('match_id').cumcount(ascending=False)
        
        # Pressure situations: last 4 overs with less than 40 runs to win
        pressure_situations = chase_matches[
            (chase_matches['over'] >= 16) & 
            (chase_matches['remaining_runs'] < 40) & 
            (chase_matches['remaining_runs'] > 0)
        ]
        
        # Get batter performance in pressure
        pressure_batters = pressure_situations.groupby('batter').agg(
            runs=('runs_batter', 'sum'),
            balls=('valid_ball', 'sum'),
            dismissals=('player_out', lambda x: (x != 'None').sum())
        ).reset_index()
        
        # Calculate strike rate
        pressure_batters['strike_rate'] = np.where(
            pressure_batters['balls'] > 0,
            (pressure_batters['runs'] / pressure_batters['balls']) * 100,
            0
        )
        
        # Filter batters with at least 20 balls faced
        pressure_batters = pressure_batters[pressure_batters['balls'] >= 20]
        
        # Get top performers
        top_pressure_batters = pressure_batters.nlargest(10, 'strike_rate')
        
        fig = px.scatter(
            top_pressure_batters,
            x='runs',
            y='strike_rate',
            size='balls',
            color='dismissals',
            title='Batter Performance in Pressure Situations',
            labels={'runs': 'Runs', 'strike_rate': 'Strike Rate', 'dismissals': 'Dismissals'},
            hover_name='batter'
        )
        
        return fig
    
    elif analysis_type == 'momentum':
        # Calculate momentum shifts (wickets in quick succession)
        match_df['over_ball'] = match_df['over'] + (match_df['ball'] / 10)
        
        # Get wicket events
        wickets = match_df[match_df['wicket_kind'] != 'None'].copy()
        
        # Group by match and innings
        wickets['wicket_number'] = wickets.groupby(['match_id', 'innings']).cumcount() + 1
        
        # Find quick wickets (within 12 balls)
        quick_wickets = wickets.copy()
        quick_wickets['prev_wicket_time'] = quick_wickets.groupby(['match_id', 'innings'])['over_ball'].shift(1)
        quick_wickets['balls_since_prev'] = (quick_wickets['over_ball'] - quick_wickets['prev_wicket_time']) * 10
        
        # Filter wickets within 12 balls of previous wicket
        quick_wickets = quick_wickets[
            (quick_wickets['wicket_number'] > 1) & 
            (quick_wickets['balls_since_prev'] <= 12)
        ]
        
        # Count quick wicket clusters by team
        momentum_shifts = quick_wickets.groupby('bowling_team').size().reset_index(name='quick_wicket_clusters')
        
        fig = px.bar(
            momentum_shifts,
            x='bowling_team',
            y='quick_wicket_clusters',
            title='Momentum Shifts (Quick Wicket Clusters)',
            labels={'quick_wicket_clusters': 'Quick Wicket Clusters', 'bowling_team': 'Team'}
        )
        
        return fig
    
    else:  # impact
        # Calculate player impact score (simplified version)
        # Impact = (runs + wickets*25) / matches
        
        # Get batter impact
        batters = get_player_performance_data(match_df, 'batter')
        batters['impact'] = (batters['runs'] + batters['dismissals']*10) / batters['matches']
        
        # Get bowler impact
        bowlers = get_player_performance_data(match_df, 'bowler')
        bowlers['impact'] = (bowlers['wickets']*25 + bowlers['dots']*0.5) / bowlers['matches']
        
        # Combine and get top impact players
        batters['role'] = 'Batter'
        bowlers['role'] = 'Bowler'
        
        batters = batters.rename(columns={'batter': 'player'})
        bowlers = bowlers.rename(columns={'bowler': 'player'})
        
        impact_players = pd.concat([
            batters[['player', 'impact', 'role']],
            bowlers[['player', 'impact', 'role']]
        ])
        
        # Get top 20 impact players
        top_impact = impact_players.nlargest(20, 'impact')
        
        fig = px.bar(
            top_impact,
            x='player',
            y='impact',
            color='role',
            title='Player Impact Scores',
            labels={'impact': 'Impact Score', 'player': 'Player'}
        )
        
        return fig

@app.callback(
    Output("advanced-insights", "children"),
    Input("advanced-analysis-dropdown", "value"),
    Input("advanced-season-dropdown", "value")
)
def update_advanced_insights(analysis_type, season):
    match_df = df.copy()
    if season:
        match_df = match_df[match_df['season'] == season]
    
    if analysis_type == 'partnership':
        # Get top partnership
        partnership_data = match_df.groupby(['match_id', 'innings', 'partnership', 'batting_team']).agg(
            runs=('runs_total', 'sum'),
            balls=('valid_ball', 'sum')
        ).reset_index()
        
        top_partnership = partnership_data.nlargest(1, 'runs')
        
        return html.Div([
            html.H4("Key Partnership Insights"),
            html.P(f"Highest partnership: {top_partnership['runs'].values[0]} runs in {top_partnership['balls'].values[0]} balls"),
            html.P(f"Team: {top_partnership['batting_team'].values[0]}"),
            html.P(f"Match ID: {top_partnership['match_id'].values[0]}, Innings: {top_partnership['innings'].values[0]}")
        ])
    
    elif analysis_type == 'pressure':
        # Calculate pressure performance stats
        chase_matches = match_df[match_df['innings'] == 2].copy()
        chase_matches['target'] = chase_matches.groupby('match_id')['runs_target'].transform('first')
        chase_matches['remaining_runs'] = chase_matches['target'] - chase_matches.groupby('match_id')['team_runs'].transform('cumsum')
        chase_matches['remaining_balls'] = chase_matches.groupby('match_id').cumcount(ascending=False)
        
        pressure_situations = chase_matches[
            (chase_matches['over'] >= 16) & 
            (chase_matches['remaining_runs'] < 40) & 
            (chase_matches['remaining_runs'] > 0)
        ]
        
        total_pressure_balls = pressure_situations['valid_ball'].sum()
        total_pressure_runs = pressure_situations['runs_batter'].sum()
        pressure_run_rate = (total_pressure_runs / total_pressure_balls) * 6 if total_pressure_balls > 0 else 0
        
        return html.Div([
            html.H4("Pressure Performance Insights"),
            html.P(f"Total pressure situations balls: {total_pressure_balls}"),
            html.P(f"Total runs scored in pressure: {total_pressure_runs}"),
            html.P(f"Pressure run rate: {pressure_run_rate:.2f} runs per over")
        ])
    
    elif analysis_type == 'momentum':
        # Calculate momentum shift stats
        wickets = match_df[match_df['wicket_kind'] != 'None'].copy()
        wickets['wicket_number'] = wickets.groupby(['match_id', 'innings']).cumcount() + 1
        
        quick_wickets = wickets.copy()
        quick_wickets['prev_wicket_time'] = quick_wickets.groupby(['match_id', 'innings'])['over_ball'].shift(1)
        quick_wickets['balls_since_prev'] = (quick_wickets['over_ball'] - quick_wickets['prev_wicket_time']) * 10
        
        quick_wickets = quick_wickets[
            (quick_wickets['wicket_number'] > 1) & 
            (quick_wickets['balls_since_prev'] <= 12)
        ]
        
        total_quick_wicket_clusters = len(quick_wickets.groupby(['match_id', 'innings']))
        total_matches = match_df['match_id'].nunique()
        
        return html.Div([
            html.H4("Momentum Shift Insights"),
            html.P(f"Total quick wicket clusters: {total_quick_wicket_clusters}"),
            html.P(f"Percentage of matches with momentum shifts: {(total_quick_wicket_clusters / total_matches) * 100:.1f}%")
        ])
    
    else:  # impact
        # Calculate top impact players
        batters = get_player_performance_data(match_df, 'batter')
        batters['impact'] = (batters['runs'] + batters['dismissals']*10) / batters['matches']
        
        bowlers = get_player_performance_data(match_df, 'bowler')
        bowlers['impact'] = (bowlers['wickets']*25 + bowlers['dots']*0.5) / bowlers['matches']
        
        top_batter = batters.nlargest(1, 'impact')
        top_bowler = bowlers.nlargest(1, 'impact')
        
        return html.Div([
            html.H4("Player Impact Insights"),
            html.P(f"Highest impact batter: {top_batter['batter'].values[0]} (Impact: {top_batter['impact'].values[0]:.2f})"),
            html.P(f"Highest impact bowler: {top_bowler['bowler'].values[0]} (Impact: {top_bowler['impact'].values[0]:.2f})")
        ])

if __name__ == '__main__':
    app.run_server(debug=True)
