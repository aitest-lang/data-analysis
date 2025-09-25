import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import zipfile

with zipfile.ZipFile('archive.zip', 'r') as zip_ref:
    zip_ref.extractall()
# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="IPL Data Analytics Dashboard",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------
# Helpers
# -------------------------
TEAM_ALIASES = {
    # Canonicalize team names (includes your RCB fix)
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Rising Pune Supergiants": "Rising Pune Supergiant",
    "Rising Pune Supergiant": "Rising Pune Supergiant",
    "Kings XI Punjab": "Punjab Kings",
    "Delhi Daredevils": "Delhi Capitals",
    "Pune Warriors": "Pune Warriors India",
    # Leave historical teams as-is if already canonical (Deccan Chargers, Gujarat Lions, Kochi Tuskers Kerala)
}

TEAM_COLUMNS = [
    "batting_team", "bowling_team", "match_won_by", "toss_winner",
    "superover_winner", "team_reviewed"
]

TEXT_COLUMNS_STRIP = [
    "match_type", "event_name", "batter", "bowler", "extra_type", "non_striker",
    "wicket_kind", "player_out", "fielders", "review_batter", "team_reviewed",
    "review_decision", "umpire", "player_of_match", "match_won_by",
    "win_outcome", "toss_winner", "toss_decision", "venue", "city", "season",
    "gender", "team_type", "superover_winner", "result_type", "method",
    "event_match_no", "stage", "match_number", "batting_team", "bowling_team",
    "new_batter", "batting_partners", "next_batter"
]

BOOL_COLUMNS = ["runs_not_boundary", "umpires_call", "striker_out"]

PHASE_LABELS = {
    "Powerplay": (1, 6),
    "Middle": (7, 15),
    "Death": (16, 50),  # safe upper bound
}

def normalize_team_name(x: str) -> str:
    if pd.isna(x):
        return x
    x = x.strip()
    return TEAM_ALIASES.get(x, x)

def assign_phase(over_num: int) -> str:
    for phase, (lo, hi) in PHASE_LABELS.items():
        if lo <= over_num <= hi:
            return phase
    return "Unknown"

# -------------------------
# Data Loading & Cleaning
# -------------------------
@st.cache_data(show_spinner=True)
def load_and_clean_data() -> pd.DataFrame:
    df = pd.read_csv("IPL.csv")

    # Trim text columns
    for col in TEXT_COLUMNS_STRIP:
        if col in df.columns and df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip().replace({"nan": np.nan})

    # Normalize team names across all relevant columns
    for col in TEAM_COLUMNS:
        if col in df.columns:
            df[col] = df[col].apply(normalize_team_name)

    # Parse date; preserve original if parsing fails
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Booleans: ensure no NaNs
    for col in BOOL_COLUMNS:
        if col in df.columns:
            if df[col].dtype != bool:
                # Convert common string booleans to real bools if present
                df[col] = df[col].replace({"True": True, "False": False})
            df[col] = df[col].fillna(False).astype(bool)

    # Fill common nulls with safe defaults
    fill_defaults = {
        "extra_type": "None",
        "wicket_kind": "Not Out",
        "player_out": "None",
        "win_outcome": "Unknown",
        "toss_decision": "Unknown",
        "result_type": "Unknown",
        "season": "Unknown",
        "city": "Unknown",
        "venue": "Unknown",
    }
    for col, val in fill_defaults.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)

    # Ensure core numeric columns exist and are numeric
    numeric_cols = [
        "innings", "over", "ball", "ball_no", "runs_batter", "balls_faced",
        "valid_ball", "runs_extras", "runs_total", "runs_bowler", "non_striker_pos",
        "day", "month", "year", "balls_per_over", "overs", "team_runs", "team_balls",
        "team_wicket", "batter_runs", "batter_balls", "bowler_wicket", "match_id"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Derived columns
    if "date" in df.columns:
        df["match_year"] = df["date"].dt.year
    else:
        df["match_year"] = df.get("year", np.nan)

    df["is_boundary"] = (~df["runs_not_boundary"]) & (df["runs_batter"].isin([4, 6]))
    df["is_six"] = df["runs_batter"] == 6
    df["is_four"] = df["runs_batter"] == 4
    df["is_dot"] = df["runs_total"] == 0
    df["is_wicket"] = df["wicket_kind"].fillna("Not Out").ne("Not Out")

    # Phase tagging by over (1-based expected)
    if "over" in df.columns:
        df["phase"] = df["over"].fillna(0).astype(int).apply(assign_phase)
    else:
        df["phase"] = "Unknown"

    # Legal balls for bowling metrics (use provided valid_ball)
    df["legal_ball"] = df["valid_ball"].fillna(0).astype(int) == 1

    # Over-ball index within innings
    if {"over", "ball", "balls_per_over"}.issubset(df.columns):
        bpo = df["balls_per_over"].replace(0, np.nan).fillna(6).astype(int)
        df["ball_idx_in_innings"] = (df["over"].fillna(0).astype(int) - 1) * bpo + df["ball"].fillna(0).astype(int)
    else:
        df["ball_idx_in_innings"] = np.nan

    # Ensure categories for memory/performance
    cat_cols = [
        "season", "venue", "city", "match_type", "event_name",
        "batting_team", "bowling_team", "batter", "bowler",
        "toss_winner", "toss_decision", "match_won_by", "phase",
        "result_type"
    ]
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype("category")

    # Winner table (one row per match)
    winners = (
        df.loc[df["match_won_by"].notna(), ["match_id", "match_won_by"]]
        .drop_duplicates(subset=["match_id"], keep="last")
        .rename(columns={"match_won_by": "winner"})
    )
    df = df.merge(winners, on="match_id", how="left")

    return df

# -------------------------
# Cached aggregations
# -------------------------
@st.cache_data(show_spinner=False)
def compute_team_innings(df: pd.DataFrame) -> pd.DataFrame:
    # Per-innings aggregates
    g = df.groupby(["match_id", "innings", "batting_team"], observed=True)
    agg = g.agg(
        runs=("runs_total", "sum"),
        wkts=("is_wicket", "sum"),
        balls=("legal_ball", "sum"),
        sixes=("is_six", "sum"),
        fours=("is_four", "sum"),
        dots=("is_dot", "sum"),
        boundaries=("is_boundary", "sum"),
        venue=("venue", "first"),
        season=("season", "first"),
        date=("date", "first")
    ).reset_index()
    agg["overs_faced"] = (agg["balls"] / 6).replace(0, np.nan)
    agg["run_rate"] = (agg["runs"] / agg["overs_faced"]).round(2)
    return agg

@st.cache_data(show_spinner=False)
def compute_batting_stats(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("batter", observed=True).agg(
        runs=("runs_batter", "sum"),
        balls=("balls_faced", "sum"),
        matches=("match_id", "nunique"),
        sixes=("is_six", "sum"),
        fours=("is_four", "sum"),
    )
    g = g[g["runs"] > 0].copy()
    g["strike_rate"] = (g["runs"] / g["balls"] * 100).replace([np.inf, -np.inf], np.nan).round(2)
    return g.reset_index()

@st.cache_data(show_spinner=False)
def compute_bowling_stats(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("bowler", observed=True).agg(
        balls=("legal_ball", "sum"),
        runs_conceded=("runs_bowler", "sum"),
        wkts=("bowler_wicket", "sum"),
        matches=("match_id", "nunique"),
    )
    g = g[g["balls"] > 0].copy()
    g["overs"] = g["balls"] / 6
    g["economy"] = (g["runs_conceded"] / g["overs"]).replace([np.inf, -np.inf], np.nan).round(2)
    g["strike_rate"] = (g["balls"] / g["wkts"]).replace([np.inf, -np.inf], np.nan).round(2)
    g["avg"] = (g["runs_conceded"] / g["wkts"]).replace([np.inf, -np.inf], np.nan).round(2)
    return g.reset_index()

@st.cache_data(show_spinner=False)
def compute_venue_summary(team_innings: pd.DataFrame) -> pd.DataFrame:
    g = team_innings.groupby("venue", observed=True).agg(
        matches=("match_id", "nunique"),
        avg_score=("runs", "mean"),
        max_score=("runs", "max"),
        min_score=("runs", "min"),
        avg_rr=("run_rate", "mean")
    )
    return g.reset_index()

@st.cache_data(show_spinner=False)
def compute_win_table(df: pd.DataFrame) -> pd.DataFrame:
    wins = (
        df.loc[df["winner"].notna(), ["match_id", "winner", "win_outcome", "toss_winner", "toss_decision", "venue", "season"]]
        .drop_duplicates(subset=["match_id"], keep="last")
    )
    return wins

# -------------------------
# Global Filters
# -------------------------
def apply_filters(df: pd.DataFrame):
    st.sidebar.header("Filters")

    seasons = sorted([s for s in df["season"].dropna().unique().tolist() if s != "Unknown"])
    venues = sorted([v for v in df["venue"].dropna().unique().tolist() if v != "Unknown"])
    teams = sorted([t for t in df["batting_team"].dropna().unique().tolist() if t != "Unknown"])

    sel_seasons = st.sidebar.multiselect("Season", seasons, default=seasons)
    sel_teams = st.sidebar.multiselect("Team (either side)", teams, default=teams)
    sel_venues = st.sidebar.multiselect("Venue", venues, default=venues)

    df_f = df.copy()
    if sel_seasons:
        df_f = df_f[df_f["season"].isin(sel_seasons)]
    if sel_venues:
        df_f = df_f[df_f["venue"].isin(sel_venues)]
    if sel_teams:
        df_f = df_f[(df_f["batting_team"].isin(sel_teams)) | (df_f["bowling_team"].isin(sel_teams))]

    return df_f, sel_seasons, sel_teams, sel_venues

# -------------------------
# UI Sections
# -------------------------
def overview_page(df: pd.DataFrame, team_innings: pd.DataFrame):
    st.header("Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Matches", df["match_id"].nunique())
    with col2:
        st.metric("Teams", df["batting_team"].nunique())
    with col3:
        st.metric("Players", df["batter"].nunique())
    with col4:
        st.metric("Venues", df["venue"].nunique())

    st.subheader("Sample")
    st.dataframe(df.head(10), use_container_width=True)

    st.subheader("Date Range")
    min_d = df["date"].min()
    max_d = df["date"].max()
    if pd.notna(min_d) and pd.notna(max_d):
        st.write(f"{min_d.date()} to {max_d.date()}")

    st.subheader("Missing Values")
    miss = df.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    if not miss.empty:
        st.dataframe(miss)

    st.subheader("Season-wise Average Runs per Innings")
    season_avg = (
        team_innings.groupby("season", observed=True)["runs"].mean().reset_index().sort_values("season")
    )
    fig = px.line(season_avg, x="season", y="runs", markers=True, title="Average Runs per Innings by Season")
    st.plotly_chart(fig, use_container_width=True)

def team_performance_page(df: pd.DataFrame, team_innings: pd.DataFrame, wins: pd.DataFrame):
    st.header("Team Performance")

    # Summaries per team (batting innings aggregation)
    team_sum = team_innings.groupby("batting_team", observed=True).agg(
        matches=("match_id", "nunique"),
        avg_score=("runs", "mean"),
        max_score=("runs", "max"),
        avg_rr=("run_rate", "mean"),
        sixes=("sixes", "sum"),
        fours=("fours", "sum"),
        wkts_per_innings=("wkts", "mean"),
        boundaries_per_innings=("boundaries", "mean"),
        dots_per_innings=("dots", "mean"),
    ).reset_index()

    # Win counts
    win_counts = wins["winner"].value_counts().rename_axis("batting_team").reset_index(name="wins")
    team_sum = team_sum.merge(win_counts, on="batting_team", how="left").fillna({"wins": 0})
    team_sum["wins"] = team_sum["wins"].astype(int)

    st.subheader("Team Table")
    st.dataframe(team_sum.sort_values("avg_score", ascending=False), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            team_sum.sort_values("avg_score", ascending=False),
            x="batting_team", y="avg_score", color="avg_score",
            title="Average Score by Team", color_continuous_scale="viridis"
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.scatter(
            team_sum, x="matches", y="max_score", size="wins", color="batting_team",
            title="Max Score vs Matches (Bubble = Wins)", hover_data=["avg_rr"]
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Phase-wise Scoring (Selected Teams)")
    teams = sorted(team_sum["batting_team"].unique().tolist())
    selected = st.multiselect("Select teams", teams, default=teams[:6])
    phase_df = (
        df[df["batting_team"].isin(selected)]
        .groupby(["batting_team", "phase"], observed=True)["runs_total"]
        .sum().reset_index()
    )
    fig = px.bar(phase_df, x="batting_team", y="runs_total", color="phase", barmode="group",
                 title="Runs by Phase")
    st.plotly_chart(fig, use_container_width=True)

def player_stats_page(df: pd.DataFrame):
    st.header("Player Statistics")

    batting = compute_batting_stats(df)
    bowling = compute_bowling_stats(df)

    st.subheader("Top Batters")
    metric = st.selectbox("Batting metric", ["runs", "strike_rate", "sixes", "fours"])
    top_n = st.slider("Top N", 5, 25, 10)
    st.dataframe(batting.sort_values(metric, ascending=False).head(top_n), use_container_width=True)
    fig = px.bar(batting.sort_values(metric, ascending=False).head(top_n), x="batter", y=metric,
                 color=metric, title=f"Top {top_n} batters by {metric}")
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Strike Rate vs Runs (Batting)")
    fig = px.scatter(
        batting, x="runs", y="strike_rate", size="matches",
        hover_name="batter", color="runs", title="Batting Strike Rate vs Runs"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top Bowlers")
    bowl_metric = st.selectbox("Bowling metric", ["wkts", "economy", "avg", "strike_rate", "runs_conceded"])
    st.dataframe(bowling.sort_values(
        bowl_metric,
        ascending=(bowl_metric in ["economy", "avg", "strike_rate"])
    ).head(top_n), use_container_width=True)
    fig = px.bar(
        bowling.sort_values(
            bowl_metric,
            ascending=(bowl_metric in ["economy", "avg", "strike_rate"])
        ).head(top_n),
        x="bowler", y=bowl_metric, color=bowl_metric, title=f"Top {top_n} bowlers by {bowl_metric}"
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

def venue_analysis_page(df: pd.DataFrame, team_innings: pd.DataFrame, wins: pd.DataFrame):
    st.header("Venue Analysis")

    venue_sum = compute_venue_summary(team_innings)
    st.subheader("Venue Table")
    st.dataframe(venue_sum.sort_values("avg_score", ascending=False), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        top10 = venue_sum.sort_values("avg_score", ascending=False).head(10)
        fig = px.bar(top10, x="venue", y="avg_score", color="matches", title="Top 10 Venues by Avg Score")
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.pie(venue_sum.sort_values("matches", ascending=False).head(10),
                     values="matches", names="venue", title="Matches by Venue (Top 10)")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Chase vs Defend at Venue")
    # Winner vs innings 2 inference: compute batting teams per match innings
    inns = team_innings.copy()
    inns1 = inns[inns["innings"] == 1][["match_id", "batting_team"]].rename(columns={"batting_team": "team_inn1"})
    inns2 = inns[inns["innings"] == 2][["match_id", "batting_team"]].rename(columns={"batting_team": "team_inn2"})
    w = wins.merge(inns1, on="match_id", how="left").merge(inns2, on="match_id", how="left")
    w["won_chasing"] = (w["winner"] == w["team_inn2"]).astype(int)
    venue_win = w.groupby("venue", observed=True)["won_chasing"].agg(["mean", "count"]).reset_index()
    venue_win.rename(columns={"mean": "chase_win_rate", "count": "matches"}, inplace=True)

    show_v = st.selectbox("Select venue", sorted(venue_win["venue"].unique().tolist()))
    row = venue_win[venue_win["venue"] == show_v].iloc[0]
    st.metric("Chase Win Rate", f"{row['chase_win_rate']*100:.1f}%")
    st.metric("Matches Considered", int(row["matches"]))

def season_trends_page(df: pd.DataFrame):
    st.header("Season Trends")

    # Runs and boundaries per match (approx via innings)
    inns = compute_team_innings(df)
    season_avg = inns.groupby("season", observed=True).agg(
        avg_runs=("runs", "mean"),
        avg_rr=("run_rate", "mean"),
        matches=("match_id", "nunique"),
        sixes=("sixes", "sum"),
        fours=("fours", "sum")
    ).reset_index()

    col1, col2 = st.columns(2)
    with col1:
        fig = px.line(season_avg.sort_values("season"), x="season", y="avg_runs",
                      markers=True, title="Avg Runs per Innings by Season")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=season_avg["season"], y=season_avg["sixes"], name="Sixes", mode="lines+markers"), secondary_y=False)
        fig.add_trace(go.Scatter(x=season_avg["season"], y=season_avg["fours"], name="Fours", mode="lines+markers"), secondary_y=True)
        fig.update_yaxes(title_text="Sixes", secondary_y=False)
        fig.update_yaxes(title_text="Fours", secondary_y=True)
        fig.update_layout(title_text="Season Boundary Totals")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Toss Impact by Season")
    wins = compute_win_table(df)
    toss = wins.copy()
    toss["won_after_toss_win"] = (toss["winner"] == toss["toss_winner"]).astype(int)
    toss_agg = toss.groupby("season", observed=True)["won_after_toss_win"].mean().reset_index()
    toss_agg["won_after_toss_win"] *= 100
    fig = px.bar(toss_agg, x="season", y="won_after_toss_win", title="Toss Winner Match Win % by Season")
    st.plotly_chart(fig, use_container_width=True)

def match_insights_page(df: pd.DataFrame):
    st.header("Match Insights")

    # Select a match
    match_ids = sorted(df["match_id"].unique().tolist())
    mid = st.selectbox("Select match_id", match_ids)

    dff = df[df["match_id"] == mid].copy()
    inns_labels = sorted(dff["innings"].dropna().unique().tolist())

    st.subheader("Worm (Over-by-over Runs)")
    # Compute over runs per innings
    over_runs = (
        dff.groupby(["innings", "over"], observed=True)["runs_total"]
        .sum().reset_index().sort_values(["innings", "over"])
    )
    fig = px.line(over_runs, x="over", y="runs_total", color="innings", markers=True,
                  title=f"Match {mid}: Runs per Over")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Phase Breakdown")
    phase_df = (
        dff.groupby(["innings", "phase"], observed=True)["runs_total"]
        .sum().reset_index()
    )
    fig = px.bar(phase_df, x="phase", y="runs_total", color="innings", barmode="group",
                 title=f"Match {mid}: Runs by Phase")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Key Facts")
    winners = dff.loc[dff["winner"].notna(), ["winner"]].drop_duplicates()
    if not winners.empty:
        st.metric("Winner", winners.iloc[0]["winner"])
    st.write("Venue:", dff["venue"].iloc[0])
    st.write("Season:", dff["season"].iloc[0])

    # Super Over insight if present
    if "superover_winner" in dff.columns:
        sow = dff["superover_winner"].dropna().unique().tolist()
        if sow:
            st.metric("Super Over Winner", sow[0])

def head_to_head_page(df: pd.DataFrame):
    st.header("Head-to-Head")

    teams = sorted(df["batting_team"].dropna().unique().tolist())
    c1, c2 = st.columns(2)
    with c1:
        team_a = st.selectbox("Team A", teams, index=0)
    with c2:
        team_b = st.selectbox("Team B", teams, index=1 if len(teams) > 1 else 0)

    # Consider matches where both appear
    matches_both = df[df["batting_team"].isin([team_a, team_b]) | df["bowling_team"].isin([team_a, team_b])]
    mids = matches_both["match_id"].unique()
    dff = df[df["match_id"].isin(mids)].copy()

    wins = compute_win_table(dff)
    total = wins["match_id"].nunique()
    wins_a = (wins["winner"] == team_a).sum()
    wins_b = (wins["winner"] == team_b).sum()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Matches", total)
    col2.metric(f"Wins - {team_a}", wins_a)
    col3.metric(f"Wins - {team_b}", wins_b)

    # Avg scores against each other (per innings for the two teams)
    inns = compute_team_innings(dff)
    inns_ab = inns[inns["batting_team"].isin([team_a, team_b])]
    avg_scores = inns_ab.groupby("batting_team", observed=True)["runs"].mean().reset_index()
    fig = px.bar(avg_scores, x="batting_team", y="runs", color="batting_team",
                 title="Average Innings Score (H2H)")
    st.plotly_chart(fig, use_container_width=True)

    # Boundary comparison
    bndry = inns_ab.groupby("batting_team", observed=True)[["sixes", "fours"]].sum().reset_index()
    fig = go.Figure(data=[
        go.Bar(name="Sixes", x=bndry["batting_team"], y=bndry["sixes"]),
        go.Bar(name="Fours", x=bndry["batting_team"], y=bndry["fours"]),
    ])
    fig.update_layout(barmode="group", title="Boundaries (H2H)")
    st.plotly_chart(fig, use_container_width=True)

# -------------------------
# Main App
# -------------------------
def main():
    st.markdown('<h1 style="text-align:center; margin-bottom: 0.5rem;">🏏 IPL Data Analytics Dashboard</h1>', unsafe_allow_html=True)
    st.caption("Cleaned, normalized, and optimized for fast, insightful analysis")

    df = load_and_clean_data()
    if df is None or df.empty:
        st.error("Failed to load data. Ensure IPL.csv is present and readable.")
        return

    # Apply global filters
    df_f, sel_seasons, sel_teams, sel_venues = apply_filters(df)

    # Pre-compute shared aggregations on filtered data
    team_innings = compute_team_innings(df_f)
    wins = compute_win_table(df_f)

    st.sidebar.header("Category")
    page = st.sidebar.radio(
        "Select Analysis",
        ["Overview", "Team Performance", "Player Statistics", "Venue Analysis", "Season Trends", "Match Insights", "Head-to-Head"],
        index=0
    )

    if page == "Overview":
        overview_page(df_f, team_innings)
    elif page == "Team Performance":
        team_performance_page(df_f, team_innings, wins)
    elif page == "Player Statistics":
        player_stats_page(df_f)
    elif page == "Venue Analysis":
        venue_analysis_page(df_f, team_innings, wins)
    elif page == "Season Trends":
        season_trends_page(df_f)
    elif page == "Match Insights":
        match_insights_page(df_f)
    elif page == "Head-to-Head":
        head_to_head_page(df_f)

    # Footer filters summary
    st.markdown("---")
    st.caption(
        f"Filters — Seasons: {', '.join(sel_seasons) if sel_seasons else 'All'} | "
        f"Teams: {', '.join(sel_teams) if sel_teams else 'All'} | "
        f"Venues: {', '.join(sel_venues) if sel_venues else 'All'}"
    )

if __name__ == "__main__":
    main()
