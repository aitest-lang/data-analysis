import pandas as pd
import streamlit as st
import plotly.express as px
import zipfile

with zipfile.ZipFile('archive.zip', 'r') as zip_ref:
    zip_ref.extractall()
# Team aliases
TEAM_ALIASES = {
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
    "Rising Pune Supergiant": "Rising Pune Supergiant",
    "Kings XI Punjab": "Punjab Kings",
    "Delhi Daredevils": "Delhi Capitals",
    "Pune Warriors": "Pune Warriors India",
}

@st.cache_data
def load_data():
    df = pd.read_csv("IPL.csv")
    # Preprocess
    df = df.drop(columns=['Unnamed: 0'])
    df['date'] = pd.to_datetime(df['date'])
    for old, new in TEAM_ALIASES.items():
        df['batting_team'] = df['batting_team'].replace(old, new)
        df['bowling_team'] = df['bowling_team'].replace(old, new)
        df['match_won_by'] = df['match_won_by'].replace(old, new)
        df['toss_winner'] = df['toss_winner'].replace(old, new)
    df = df.fillna({'runs_extras': 0, 'wicket_kind': 'None', 'runs_target': 0})
    return df

def main():
    st.title("IPL Data Analysis (2008-2025)")
    df = load_data()

    # Sidebar filters
    years = sorted(df['year'].unique())  # Use 'year' instead of 'season'
    teams = sorted(df['batting_team'].unique())
    selected_years = st.sidebar.multiselect("Select Years", years, default=years)
    selected_teams = st.sidebar.multiselect("Select Teams", teams, default=teams)

    # Filter data
    filtered_df = df[df['year'].isin(selected_years) & 
                    (df['batting_team'].isin(selected_teams) | df['bowling_team'].isin(selected_teams))]

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Teams", "Players", "Matches", "Venues", "Trends"])

    with tab1:
        st.header("Team Performance")
        wins = filtered_df.groupby(['year', 'match_won_by']).size().reset_index(name='wins')  # Use 'year'
        fig = px.bar(wins, x='year', y='wins', color='match_won_by', title="Wins by Team per Year")
        st.plotly_chart(fig)
        team_runs = filtered_df.groupby('batting_team')['team_runs'].sum().reset_index()
        st.dataframe(team_runs.sort_values('team_runs', ascending=False))

    with tab2:
        st.header("Player Stats")
        top_batters = filtered_df.groupby('batter')['runs_batter'].sum().reset_index().sort_values('runs_batter', ascending=False).head(10)
        fig = px.bar(top_batters, x='batter', y='runs_batter', title="Top 10 Batters by Runs")
        st.plotly_chart(fig)
        st.dataframe(top_batters)

    with tab3:
        st.header("Match Insights")
        fig = px.histogram(filtered_df, x='runs_total', nbins=50, title="Runs Distribution per Ball")
        st.plotly_chart(fig)
        superovers = filtered_df[filtered_df['superover_winner'].notnull()].groupby('superover_winner').size().reset_index(name='count')
        st.dataframe(superovers)

    with tab4:
        st.header("Venue Trends")
        venue_wins = filtered_df.groupby('venue')['match_won_by'].value_counts().unstack().fillna(0)
        fig = px.pie(values=venue_wins.sum(), names=venue_wins.columns, title="Wins by Team across Venues")
        st.plotly_chart(fig)
        st.dataframe(venue_wins)

    with tab5:
        st.header("Year Trends")
        avg_runs = filtered_df.groupby('year')['team_runs'].mean().reset_index()
        fig = px.line(avg_runs, x='year', y='team_runs', title="Average Team Runs per Year")
        st.plotly_chart(fig)
        st.dataframe(avg_runs)

    # Raw data viewer
    with st.expander("View Raw Data"):
        st.dataframe(filtered_df.head(100))

if __name__ == "__main__":
    main()
