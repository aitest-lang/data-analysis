import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import zipfile

with zipfile.ZipFile('archive.zip', 'r') as zip_ref:
    zip_ref.extractall()
# -------------------------------------------------
# 0. CONFIG & CONSTANTS
# -------------------------------------------------
st.set_page_config(page_title="IPL Analytics Hub", layout="wide")
st.sidebar.title("🏏 IPL Analytics Hub")

TEAM_ALIASES = {
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
    "Kings XI Punjab": "Punjab Kings",
    "Delhi Daredevils": "Delhi Capitals",
    "Pune Warriors": "Pune Warriors India",
}

# -------------------------------------------------
# 1. DATA LOADING & CLEANING (CACHED)
# -------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv("IPL.csv")

    # canonicalise team names
    for old, new in TEAM_ALIASES.items():
        df["batting_team"] = df["batting_team"].replace(old, new)
        df["bowling_team"] = df["bowling_team"].replace(old, new)
        df["toss_winner"] = df["toss_winner"].replace(old, new)

    # parse date
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = df["date"].dt.year

    # drop the unnamed index column if it exists
    df = df.drop(columns=[c for c in df.columns if "unnamed" in c.lower()])

    return df

df = load_data()

# -------------------------------------------------
# 2. SIDEBAR FILTERS (GLOBAL)
# -------------------------------------------------
with st.sidebar:
    st.subheader("Global Filters")
    years = st.slider("Seasons", int(df["year"].min()), int(df["year"].max()),
                      (int(df["year"].min()), int(df["year"].max())))
    teams = st.multiselect("Teams", sorted(df["batting_team"].unique()),
                           default=sorted(df["batting_team"].unique()))

# -------------------------------------------------
# 3. APPLY FILTERS ONCE
# -------------------------------------------------
filtered = df[(df["year"].between(*years)) &
              (df["batting_team"].isin(teams))]

# -------------------------------------------------
# 4. ANALYSIS CATEGORIES
# -------------------------------------------------
CATEGORIES = {
    "Overview": "High-level season & team summary",
    "Batting": "Runs, SR, boundaries, milestones…",
    "Bowling": "Eco, wickets, dot %…",
    "Partnerships": "Stand, avg, SR by pair",
    "Venue & Toss": "Home advantage, toss impact",
    "Wickets": "Kind, timing, fielders",
}

choice = st.sidebar.radio("Pick analysis", list(CATEGORIES.keys()),
                          format_func=lambda x: f"{x} – {CATEGORIES[x]}")

# -------------------------------------------------
# 5. CATEGORY-SPECIFIC PLOTS
# -------------------------------------------------
def overview():
    st.header("Season & Team Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Matches", filtered["match_id"].nunique())
    with col2:
        st.metric("Total Runs", filtered["runs_total"].sum())
    with col3:
        st.metric("Total Wickets", filtered["wicket_kind"].notna().sum())

    # runs per season
    season_runs = filtered.groupby("year")["runs_total"].sum().reset_index()
    fig = px.bar(season_runs, x="year", y="runs_total", title="Runs per season")
    st.plotly_chart(fig, use_container_width=True)

def batting():
    st.header("Batting Analytics")
    # top run scorers
    batter_stats = (filtered.groupby("batter")
                    .agg(runs=("runs_batter", "sum"),
                         balls=("balls_faced", "sum"))
                    .reset_index())
    batter_stats["sr"] = (batter_stats["runs"] / batter_stats["balls"] * 100).round(2)
    batter_stats = batter_stats.sort_values("runs", ascending=False).head(30)

    fig = px.scatter(batter_stats, x="balls", y="runs", size="sr", color="sr",
                     hover_name="batter", title="Top 30 run scorers (size = SR)")
    st.plotly_chart(fig, use_container_width=True)

def bowling():
    st.header("Bowling Analytics")
    bowl_stats = (filtered.groupby("bowler")
                  .agg(runs=("runs_bowler", "sum"),
                       balls=("valid_ball", "sum"),
                       wkt=("bowler_wicket", "sum"))
                  .reset_index())
    bowl_stats["eco"] = (bowl_stats["runs"] / (bowl_stats["balls"] / 6)).round(2)
    bowl_stats = bowl_stats[bowl_stats["balls"] >= 60].sort_values("eco").head(30)

    fig = px.scatter(bowl_stats, x="wkt", y="eco", hover_name="bowler",
                     title="Economy vs Wickets (≥10 overs)")
    st.plotly_chart(fig, use_container_width=True)

def partnerships():
    st.header("Batting Partnerships")
    # create stand identifier
    temp = filtered.copy()
    temp["stand"] = temp["batting_partners"].astype(str) + " vs " + temp["bowling_team"]
    stand_stats = (temp.groupby(["batting_team", "batting_partners"])
                   .agg(runs=("runs_total", "sum"),
                        balls=("valid_ball", "sum"))
                   .reset_index())
    stand_stats["sr"] = (stand_stats["runs"] / stand_stats["balls"] * 100).round(2)
    stand_stats = stand_stats.sort_values("runs", ascending=False).head(30)

    fig = px.bar(stand_stats, x="runs", y="batting_partners", color="batting_team",
                 title="Top 30 partnerships by runs")
    st.plotly_chart(fig, use_container_width=True)

def venue_toss():
    st.header("Venue & Toss Impact")
    toss_win = (filtered.groupby(["venue", "toss_winner", "toss_decision"])
                .size().reset_index(name="count"))
    fig = px.treemap(toss_win, path=["venue", "toss_winner", "toss_decision"],
                     values="count", title="Toss decision distribution by venue")
    st.plotly_chart(fig, use_container_width=True)

def wickets():
    st.header("Wicket Analytics")
    wkt_kind = filtered["wicket_kind"].value_counts().reset_index()
    fig = px.pie(wkt_kind, names="wicket_kind", values="count",
                 title="Dismissal types")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------
# 6. ROUTER
# -------------------------------------------------
if choice == "Overview":
    overview()
elif choice == "Batting":
    batting()
elif choice == "Bowling":
    bowling()
elif choice == "Partnerships":
    partnerships()
elif choice == "Venue & Toss":
    venue_toss()
elif choice == "Wickets":
    wickets()
