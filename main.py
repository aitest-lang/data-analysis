import pandas as pd
import streamlit as st
import altair as alt

# ---------- CONFIG ----------
st.set_page_config(page_title="IPL Insights", layout="centered")
st.title("🏏 IPL Insights – Quick View")
alt.themes.enable("streamlit")
import zipfile

with zipfile.ZipFile('archive.zip', 'r') as zip_ref:
    zip_ref.extractall()
# ---------- DATA ----------
@st.cache_data
def load_data():
    df = pd.read_csv("IPL.csv")
    # canonical team names
    aliases = {
        "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
        "Rising Pune Supergiants": "Rising Pune Supergiant",
        "Kings XI Punjab": "Punjab Kings",
        "Delhi Daredevils": "Delhi Capitals",
        "Pune Warriors": "Pune Warriors India",
    }
    df["batting_team"] = df["batting_team"].replace(aliases)
    df["bowling_team"] = df["bowling_team"].replace(aliases)
    df["toss_winner"] = df["toss_winner"].replace(aliases)
    df["match_won_by"] = df["match_won_by"].replace(aliases)
    # date
    df["date"] = pd.to_datetime(df["date"])
    df["season"] = df["season"].astype(str)
    return df

df = load_data()

# ---------- SIDEBAR ----------
with st.sidebar:
    st.header("Filters")
    seasons = st.multiselect("Season", sorted(df["season"].unique()), default=sorted(df["season"].unique()))
    teams = st.multiselect("Team", sorted(df["batting_team"].unique()), default=sorted(df["batting_team"].unique()))
    venues = st.selectbox("Venue", ["All"] + sorted(df["venue"].unique()))
    match_types = st.multiselect("Match Type", sorted(df["match_type"].unique()), default=sorted(df["match_type"].unique()))
    over_range = st.slider("Over range", 1, 20, (1, 20))

# ---------- FILTER ----------
def filtered(df):
    f = df[
        (df["season"].isin(seasons)) &
        (df["batting_team"].isin(teams) | df["bowling_team"].isin(teams)) &
        (df["match_type"].isin(match_types)) &
        (df["over"].between(over_range[0], over_range[1]))
    ]
    if venues != "All":
        f = f[f["venue"] == venues]
    return f

df_f = filtered(df)

# ---------- TABS ----------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Scorecard", "Batting", "Bowling", "Head-to-Head", "Story"])

with tab1:
    st.subheader("Team Scorecard")
    score = df_f.groupby("batting_team").agg(
        runs=pd.NamedAgg("runs_total", "sum"),
        wickets=pd.NamedAgg("wicket_kind", lambda x: x.notna().sum()),
        extras=pd.NamedAgg("runs_extras", "sum")
    ).reset_index().sort_values("runs", ascending=False)
    st.dataframe(score, use_container_width=True)
    st.download_button("Download CSV", score.to_csv(index=False), file_name="scorecard.csv")

with tab2:
    st.subheader("Top Batters")
    bat = df_f.groupby(["batter", "batting_team"]).agg(
        runs=pd.NamedAgg("runs_batter", "sum"),
        balls=pd.NamedAgg("balls_faced", "sum"),
        fours=pd.NamedAgg("runs_batter", lambda x: (x == 4).sum()),
        sixes=pd.NamedAgg("runs_batter", lambda x: (x == 6).sum())
    ).reset_index()
    bat["SR"] = (bat["runs"] / bat["balls"] * 100).round(1)
    bat = bat[bat["balls"] >= 5].sort_values("runs", ascending=False).head(10)
    st.dataframe(bat, use_container_width=True)
    c = alt.Chart(bat).mark_bar().encode(
        x="runs:Q", y=alt.Y("batter:N", sort="-x"), tooltip=["runs", "balls", "SR"]
    ).properties(height=300)
    st.altair_chart(c, use_container_width=True)
    st.download_button("Download CSV", bat.to_csv(index=False), file_name="batting.csv", key="bat")

with tab3:
    st.subheader("Top Bowlers")
    bowl = df_f.groupby(["bowler", "bowling_team"]).agg(
        runs=pd.NamedAgg("runs_bowler", "sum"),
        balls=pd.NamedAgg("valid_ball", "sum"),
        wickets=pd.NamedAgg("wicket_kind", lambda x: x.notna().sum())
    ).reset_index()
    bowl["Econ"] = (bowl["runs"] / (bowl["balls"] / 6)).round(2)
    bowl = bowl[bowl["balls"] >= 6].sort_values("wickets", ascending=False).head(10)
    st.dataframe(bowl, use_container_width=True)
    c = alt.Chart(bowl).mark_bar().encode(
        x="wickets:Q", y=alt.Y("bowler:N", sort="-x"), tooltip=["wickets", "Econ"]
    ).properties(height=300)
    st.altair_chart(c, use_container_width=True)
    st.download_button("Download CSV", bowl.to_csv(index=False), file_name="bowling.csv", key="bowl")

with tab4:
    st.subheader("Head-to-Head")
    col1, col2 = st.columns(2)
    t1 = col1.selectbox("Team A", sorted(df_f["batting_team"].unique()))
    t2 = col2.selectbox("Team B", sorted(df_f["batting_team"].unique()))
    h2h_df = df_f[df_f["batting_team"].isin([t1, t2]) & df_f["bowling_team"].isin([t1, t2])]
    wins = h2h_df["match_won_by"].value_counts().reset_index()
    wins.columns = ["team", "wins"]
    wins = wins[wins["team"].isin([t1, t2])]
    st.dataframe(wins, use_container_width=True)
    if not wins.empty:
        c = alt.Chart(wins).mark_bar().encode(
            x="wins:Q", y=alt.Y("team:N", sort="-x"), tooltip=["wins"]
        ).properties(height=200)
        st.altair_chart(c, use_container_width=True)

with tab5:
    st.subheader("Quick Story")
    total_runs = df_f["runs_total"].sum()
    max_total = df_f.groupby(["match_id", "batting_team"])["runs_total"].sum().max()
    max_sixes = (df_f["runs_batter"] == 6).sum()
    st.write(f"- Filtered data covers **{len(df_f['match_id'].unique())}** matches.")
    st.write(f"- **{total_runs}** total runs scored.")
    st.write(f"- Highest team total in selection: **{max_total}** runs.")
    st.write(f"- **{max_sixes}** sixes hit.")

# ---------- FOOTER ----------
st.sidebar.write("---")
st.sidebar.download_button("Download Filtered Ball-by-Ball CSV", df_f.to_csv(index=False), file_name="ipl_filtered.csv")
