# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import re
import zipfile

with zipfile.ZipFile('archive.zip', 'r') as zip_ref:
    zip_ref.extractall()

st.set_page_config(page_title="IPL Analytics", layout="wide")

# --------------------------
# Canonical team name mapping
# --------------------------
# Normalize to lowercase for matching; values are canonical display names.
TEAM_SYNONYMS = {
    "royal challengers bangalore": "Royal Challengers Bengaluru",
    "royal challengers bengaluru": "Royal Challengers Bengaluru",
    "rcb": "Royal Challengers Bengaluru",
    "delhi daredevils": "Delhi Capitals",
    "delhi capitals": "Delhi Capitals",
    "kings xi punjab": "Punjab Kings",
    "punjab kings": "Punjab Kings",
    "kolkata knight riders": "Kolkata Knight Riders",
    "mumbai indians": "Mumbai Indians",
    "chennai super kings": "Chennai Super Kings",
    "sunrisers hyderabad": "Sunrisers Hyderabad",
    "deccan chargers": "Deccan Chargers",
    "rajasthan royals": "Rajasthan Royals",
    "gujarat lions": "Gujarat Lions",
    "gujarat titans": "Gujarat Titans",
    "pune warriors": "Pune Warriors",
    "rising pune supergiants": "Rising Pune Supergiant",
    "rising pune supergiant": "Rising Pune Supergiant",
    "lucknow super giants": "Lucknow Super Giants",
    "lucknow supergiants": "Lucknow Super Giants"
}

def _strip_collapse_spaces(x):
    if pd.isna(x):
        return x
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    return s

def canonicalize_team(x):
    if pd.isna(x):
        return x
    s = _strip_collapse_spaces(x)
    key = s.lower()
    if key in TEAM_SYNONYMS:
        return TEAM_SYNONYMS[key]
    # Heuristic: handle Bangalore/Bengaluru variants
    s2 = s.replace("Bangalore", "Bengaluru")
    key2 = s2.lower()
    if key2 in TEAM_SYNONYMS:
        return TEAM_SYNONYMS[key2]
    return s  # unchanged if not mapped

@st.cache_data(show_spinner=True)
def load_raw():
    df = pd.read_csv("IPL.csv")
    return df

def clean_and_enrich(df):
    df = df.copy()

    # Drop obvious index column if present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Strip whitespace in object columns
    obj_cols = df.select_dtypes(include=["object"]).columns
    for c in obj_cols:
        df[c] = df[c].astype("string").str.strip()

    # Parse date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Canonicalize team name columns
    team_cols = [
        "batting_team", "bowling_team", "toss_winner",
        "match_won_by", "superover_winner", "team_reviewed"
    ]
    for c in team_cols:
        if c in df.columns:
            df[c] = df[c].apply(canonicalize_team)

    # Standardize person name spaces
    for c in ["batter", "bowler", "non_striker", "player_out", "player_of_match", "umpire"]:
        if c in df.columns:
            df[c] = df[c].apply(_strip_collapse_spaces)

    # Ensure numeric columns
    num_cols = [
        "innings","over","ball","ball_no","bat_pos","runs_batter","balls_faced","valid_ball",
        "runs_extras","runs_total","runs_bowler","non_striker_pos","runs_target",
        "balls_per_over","overs","team_runs","team_balls","team_wicket",
        "batter_runs","batter_balls","bowler_wicket","day","month","year"
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Derived flags
    df["is_boundary_4"] = (df["runs_batter"] == 4) & (~df.get("runs_not_boundary", False).astype(bool))
    df["is_boundary_6"] = (df["runs_batter"] == 6) & (~df.get("runs_not_boundary", False).astype(bool))
    df["is_dot_ball"] = (df["valid_ball"] == 1) & (df["runs_total"].fillna(0) == 0)

    # Ball index within innings for plotting
    # Assuming 'over' is 0-based, 'ball' is 1..balls_per_over
    df["balls_per_over"] = df["balls_per_over"].fillna(6).astype(int)
    df["ball_index"] = df["over"].fillna(0).astype(int) * df["balls_per_over"] + df["ball"].fillna(1).astype(int)

    # Innings totals: last legal delivery row per (match_id, innings)
    valid = df[df["valid_ball"] == 1].copy()
    valid = valid.sort_values(["match_id", "innings", "team_balls", "over", "ball"], na_position="last")
    last_innings_rows = valid.groupby(["match_id", "innings"], as_index=False).tail(1)

    innings_totals = last_innings_rows[[
        "match_id", "innings", "batting_team", "bowling_team",
        "team_runs", "team_wicket", "team_balls", "balls_per_over", "overs"
    ]].copy()

    # Match-level table
    match_cols = [
        "match_id","date","season","match_type","event_name","event_match_no","stage","match_number",
        "venue","city","gender","team_type","toss_winner","toss_decision","match_won_by","superover_winner",
        "win_outcome","result_type","method","overs"
    ]
    have_cols = [c for c in match_cols if c in df.columns]
    match_df = df.sort_values(["match_id", "date"]).groupby("match_id", as_index=False)[have_cols].first()

    # Winner resolution
    match_df["winner"] = match_df["superover_winner"].fillna(match_df["match_won_by"])
    match_df["winner"] = match_df["winner"].apply(canonicalize_team)

    # Identify batting-first and chasing teams from innings_totals
    first_innings = innings_totals[innings_totals["innings"] == 1][["match_id", "batting_team", "team_runs"]].rename(
        columns={"batting_team": "bat_first_team", "team_runs": "first_innings_runs"}
    )
    second_innings = innings_totals[innings_totals["innings"] == 2][["match_id", "batting_team", "team_runs"]].rename(
        columns={"batting_team": "chasing_team", "team_runs": "second_innings_runs"}
    )

    match_df = match_df.merge(first_innings, on="match_id", how="left")
    match_df = match_df.merge(second_innings, on="match_id", how="left")

    # Create convenience: list of two teams per match (from innings)
    teams_per_match = (innings_totals.groupby("match_id")["batting_team"]
                       .unique().reset_index(name="teams_list"))
    match_df = match_df.merge(teams_per_match, on="match_id", how="left")

    # Compute outcome tags
    match_df["won_batting_first"] = match_df["winner"] == match_df["bat_first_team"]
    match_df["won_chasing"] = match_df["winner"] == match_df["chasing_team"]

    # Minimal null safety
    for c in ["season","match_type","gender","team_type","stage","venue","city","toss_decision","result_type"]:
        if c in match_df.columns:
            match_df[c] = match_df[c].astype("string")

    return df, innings_totals, match_df

def apply_filters(df, match_df, seasons, mtypes, genders, team_types, stages, teams, venues, cities):
    df2 = df.copy()
    m2 = match_df.copy()

    if seasons:
        m2 = m2[m2["season"].isin(seasons)]
    if mtypes:
        m2 = m2[m2["match_type"].isin(mtypes)]
    if genders:
        m2 = m2[m2["gender"].isin(genders)]
    if team_types:
        m2 = m2[m2["team_type"].isin(team_types)]
    if stages:
        m2 = m2[m2["stage"].isin(stages)]
    if venues:
        m2 = m2[m2["venue"].isin(venues)]
    if cities:
        m2 = m2[m2["city"].isin(cities)]
    if teams:
        # Keep matches where any selected team participated
        m2 = m2[m2["teams_list"].apply(lambda t: any(tt in (t or []) for tt in teams))]

    # Filter ball-level df by the selected match_ids
    match_ids = set(m2["match_id"].unique())
    df2 = df2[df2["match_id"].isin(match_ids)]

    return df2, m2

def team_summary(match_df):
    rows = []
    for team in sorted(set(sum(match_df["teams_list"].dropna().tolist(), []))):
        matches = match_df[match_df["teams_list"].apply(lambda t: team in (t or []))]
        wins = (matches["winner"] == team).sum()
        nrs = matches["result_type"].str.lower().eq("no result").sum() if "result_type" in matches.columns else 0
        total = len(matches)
        losses = total - wins - nrs
        winp = (wins / (total - nrs) * 100) if (total - nrs) > 0 else np.nan
        toss_wins = (matches["toss_winner"] == team).sum()
        won_bf = matches["won_batting_first"].sum()
        won_chase = matches["won_chasing"].sum()
        rows.append({
            "Team": team,
            "Matches": total,
            "Wins": wins,
            "Losses": losses,
            "No Result": nrs,
            "Win %": round(winp, 2) if pd.notna(winp) else np.nan,
            "Toss Wins": toss_wins,
            "Wins Bat 1st": int(won_bf),
            "Wins Chasing": int(won_chase)
        })
    return pd.DataFrame(rows).sort_values(["Win %","Wins","Matches"], ascending=[False, False, False])

def batter_agg(df):
    # Aggregate basic batting metrics
    g = df.groupby("batter", dropna=True).agg(
        Runs=("runs_batter", "sum"),
        Balls=("balls_faced", "sum"),
        Fours=("is_boundary_4", "sum"),
        Sixes=("is_boundary_6", "sum")
    ).reset_index()
    # Dismissals
    outs = (df[(~df["wicket_kind"].isna()) & (df["player_out"] == df["batter"])]
            .groupby("batter").size().rename("Dismissals"))
    g = g.merge(outs, on="batter", how="left").fillna({"Dismissals": 0})
    # Rates
    g["SR"] = np.where(g["Balls"] > 0, g["Runs"] * 100.0 / g["Balls"], np.nan)
    g["Avg"] = np.where(g["Dismissals"] > 0, g["Runs"] * 1.0 / g["Dismissals"], np.nan)
    return g.sort_values(["Runs","SR"], ascending=[False, False])

def bowler_agg(df):
    # Only valid balls for bowling metrics
    d = df[df["valid_ball"] == 1].copy()
    g = d.groupby("bowler", dropna=True).agg(
        Balls=("valid_ball", "sum"),
        RunsConceded=("runs_bowler", "sum"),
        Wkts=("bowler_wicket", "sum")
    ).reset_index()
    # Overs and rates
    g["Overs"] = g["Balls"] / d["balls_per_over"].mode().iloc[0] if not d["balls_per_over"].mode().empty else g["Balls"]/6.0
    g["Econ"] = np.where(g["Overs"] > 0, g["RunsConceded"] / g["Overs"], np.nan)
    g["Avg"] = np.where(g["Wkts"] > 0, g["RunsConceded"] / g["Wkts"], np.nan)
    g["SR"] = np.where(g["Wkts"] > 0, g["Balls"] / g["Wkts"], np.nan)

    # Maiden overs: group by match, innings, over, bowler
    over_runs = (d.groupby(["match_id","innings","over","bowler"], dropna=True)
                   .agg(over_runs=("runs_total","sum"),
                        legal=("valid_ball","sum"))
                   .reset_index())
    maidens = (over_runs[(over_runs["legal"] > 0) & (over_runs["over_runs"] == 0)]
               .groupby("bowler").size().rename("Maidens"))
    g = g.merge(maidens, on="bowler", how="left").fillna({"Maidens": 0})
    return g.sort_values(["Wkts","Econ"], ascending=[False, True])

def venue_agg(match_df):
    # Average first-innings scores and win split
    g = match_df.groupby(["venue","city"], dropna=False).agg(
        Matches=("match_id", "nunique"),
        Avg_1st_Innings=("first_innings_runs", "mean"),
        Wins_Bat_First=("won_batting_first", "sum"),
        Wins_Chasing=("won_chasing", "sum")
    ).reset_index()
    g["Avg_1st_Innings"] = g["Avg_1st_Innings"].round(1)
    return g.sort_values(["Matches","Avg_1st_Innings"], ascending=[False, False])

def season_team_trend(match_df, team):
    m = match_df[match_df["teams_list"].apply(lambda t: team in (t or []))].copy()
    grp = m.groupby("season", dropna=False).agg(
        Matches=("match_id","nunique"),
        Wins=("winner", lambda s: (s == team).sum())
    ).reset_index()
    grp["Wins"] = grp["Wins"].astype(int)
    grp["Win %"] = np.where(grp["Matches"] > 0, grp["Wins"] * 100.0 / grp["Matches"], np.nan)
    return grp.sort_values("season")

def build_worm_chart(df_match):
    # cumulative team_runs per innings for a single match
    d = df_match.sort_values(["innings","ball_index"]).copy()
    # Use team_runs directly; it's cumulative within innings
    lines = d.groupby(["innings","batting_team","ball_index"], as_index=False)["team_runs"].max()
    chart = alt.Chart(lines).mark_line(point=False).encode(
        x=alt.X("ball_index:Q", title="Ball Index"),
        y=alt.Y("team_runs:Q", title="Cumulative Runs"),
        color=alt.Color("batting_team:N", title="Team"),
        tooltip=["innings","batting_team","ball_index","team_runs"]
    ).properties(height=350)
    return chart

def wicket_scatter(df_match):
    d = df_match[(df_match["valid_ball"] == 1) & (~df_match["wicket_kind"].isna())].copy()
    if d.empty:
        return None
    pts = d.groupby(["innings","batting_team","ball_index"], as_index=False).size()
    chart = alt.Chart(pts).mark_point(size=60, color="red").encode(
        x="ball_index:Q",
        y="innings:N",
        tooltip=["innings","batting_team","ball_index","size"]
    ).properties(height=120)
    return chart

def reviews_agg(df):
    # Robust parsing of review decisions
    d = df[~df["review_decision"].isna()].copy()
    if d.empty:
        return pd.DataFrame(columns=["Team","Reviews","Successes","Success %"])
    def is_success(x):
        s = str(x).lower()
        return any(k in s for k in ["overturn", "over-turned", "over ruled", "successful"]) and ("unsuccess" not in s)
    team_reviews = d.groupby("team_reviewed").agg(Reviews=("review_decision","count"),
                                                   Successes=("review_decision", lambda s: sum(is_success(x) for x in s))).reset_index()
    team_reviews.rename(columns={"team_reviewed":"Team"}, inplace=True)
    team_reviews["Success %"] = np.where(team_reviews["Reviews"]>0, (team_reviews["Successes"]*100.0/team_reviews["Reviews"]).round(1), np.nan)
    return team_reviews.sort_values(["Success %","Successes","Reviews"], ascending=[False, False, False])

def main():
    st.title("IPL Analytics Hub")

    df_raw = load_raw()
    df, innings_totals, match_df = clean_and_enrich(df_raw)

    # Sidebar filters
    st.sidebar.header("Global Filters")

    seasons = st.sidebar.multiselect("Season", sorted(match_df["season"].dropna().unique().tolist()))
    mtypes = st.sidebar.multiselect("Match Type", sorted(match_df["match_type"].dropna().unique().tolist()))
    genders = st.sidebar.multiselect("Gender", sorted(match_df["gender"].dropna().unique().tolist()))
    team_types = st.sidebar.multiselect("Team Type", sorted(match_df["team_type"].dropna().unique().tolist()))
    stages = st.sidebar.multiselect("Stage", sorted(match_df["stage"].dropna().unique().tolist()))
    venues = st.sidebar.multiselect("Venue", sorted(match_df["venue"].dropna().unique().tolist()))
    cities = st.sidebar.multiselect("City", sorted(match_df["city"].dropna().unique().tolist()))

    # Team list for filter helper
    all_teams = sorted(set(sum(match_df["teams_list"].dropna().tolist(), [])))
    teams = st.sidebar.multiselect("Filter Matches by Team", all_teams)

    df_f, match_f = apply_filters(df, match_df, seasons, mtypes, genders, team_types, stages, teams, venues, cities)

    st.caption(f"Matches in view: {match_f['match_id'].nunique()} | Deliveries in view: {len(df_f)}")

    category = st.sidebar.selectbox(
        "Analysis Category",
        ["Teams", "Batters", "Bowlers", "Matches", "Venues", "Reviews/Umpire"]
    )

    # Step 1: Data loading and cleaning info
    with st.expander("Step 1: Data Loading & Cleaning (view)"):
        st.write("Sample cleaned data")
        st.dataframe(df_f.head(50))
        st.write("Match-level table")
        st.dataframe(match_f.head(50))

    # Teams
    if category == "Teams":
        st.header("Team Analytics")

        ts = team_summary(match_f)
        st.subheader("Team Summary")
        st.dataframe(ts, use_container_width=True)

        team_sel = st.selectbox("Select Team for season-wise trend", [""] + all_teams)
        if team_sel:
            trend = season_team_trend(match_f, team_sel)
            if not trend.empty:
                c = alt.Chart(trend).transform_fold(
                    ["Matches","Wins","Win %"], as_=["Metric","Value"]
                ).mark_line(point=True).encode(
                    x=alt.X("season:N", title="Season"),
                    y=alt.Y("Value:Q", title="Value"),
                    color="Metric:N",
                    tooltip=["season","Metric","Value"]
                ).properties(height=350)
                st.altair_chart(c, use_container_width=True)
            else:
                st.info("No data for selected filters/team.")

    # Batters
    elif category == "Batters":
        st.header("Batter Analytics")
        team_filter = st.multiselect("Filter batters by batting team", all_teams)
        d = df_f.copy()
        if team_filter:
            d = d[d["batting_team"].isin(team_filter)]
        bat = batter_agg(d)
        st.subheader("Top Batters")
        st.dataframe(bat.head(200), use_container_width=True)

        batter_sel = st.selectbox("Inspect a batter", [""] + sorted(bat["batter"].dropna().astype(str).unique().tolist()))
        if batter_sel:
            bd = d[d["batter"] == batter_sel].copy()
            if not bd.empty:
                # Season-wise runs and SR
                by_season = bd.groupby("season", dropna=False).agg(
                    Runs=("runs_batter","sum"),
                    Balls=("balls_faced","sum")
                ).reset_index()
                by_season["SR"] = np.where(by_season["Balls"]>0, by_season["Runs"]*100.0/by_season["Balls"], np.nan)
                ch = alt.Chart(by_season).transform_fold(["Runs","SR"], as_=["Metric","Value"]).mark_bar().encode(
                    x=alt.X("season:N", title="Season"),
                    y=alt.Y("Value:Q", title="Value"),
                    color="Metric:N",
                    tooltip=["season","Metric","Value"]
                ).properties(height=300)
                st.altair_chart(ch, use_container_width=True)
                st.write("Recent deliveries (view)")
                st.dataframe(bd.sort_values(["match_id","innings","ball_index"]).tail(100))
            else:
                st.info("No deliveries for selected batter under current filters.")

    # Bowlers
    elif category == "Bowlers":
        st.header("Bowler Analytics")
        team_filter = st.multiselect("Filter bowlers by bowling team", all_teams)
        d = df_f.copy()
        if team_filter:
            d = d[d["bowling_team"].isin(team_filter)]
        bowl = bowler_agg(d)
        st.subheader("Top Bowlers")
        st.dataframe(bowl.head(200), use_container_width=True)

        bow_sel = st.selectbox("Inspect a bowler", [""] + sorted(bowl["bowler"].dropna().astype(str).unique().tolist()))
        if bow_sel:
            bd = d[d["bowler"] == bow_sel].copy()
            if not bd.empty:
                # Season-wise wickets and economy
                by_season = bd[bd["valid_ball"] == 1].groupby("season", dropna=False).agg(
                    Balls=("valid_ball","sum"),
                    Runs=("runs_bowler","sum"),
                    Wkts=("bowler_wicket","sum")
                ).reset_index()
                if not by_season.empty:
                    bpo = bd["balls_per_over"].mode().iloc[0] if not bd["balls_per_over"].mode().empty else 6
                    by_season["Overs"] = by_season["Balls"] / bpo
                    by_season["Econ"] = np.where(by_season["Overs"]>0, by_season["Runs"]/by_season["Overs"], np.nan)
                    chart = alt.Chart(by_season).transform_fold(["Wkts","Econ"], as_=["Metric","Value"]).mark_bar().encode(
                        x=alt.X("season:N", title="Season"),
                        y=alt.Y("Value:Q", title="Value"),
                        color="Metric:N",
                        tooltip=["season","Metric","Value"]
                    ).properties(height=300)
                    st.altair_chart(chart, use_container_width=True)
                st.write("Recent deliveries (view)")
                st.dataframe(bd.sort_values(["match_id","innings","ball_index"]).tail(100))
            else:
                st.info("No deliveries for selected bowler under current filters.")

    # Matches
    elif category == "Matches":
        st.header("Match Analytics")
        # Build a readable label for selection
        mf = match_f.copy()
        if mf.empty:
            st.info("No matches under current filters.")
        else:
            mf["label"] = mf.apply(lambda r: f"{r['date'].date() if pd.notna(r['date']) else ''} | {', '.join(r['teams_list']) if isinstance(r['teams_list'], (list,np.ndarray)) else ''} @ {r['venue']}", axis=1)
            sel = st.selectbox("Select Match", mf.sort_values("date")["label"].tolist())
            if sel:
                row = mf[mf["label"] == sel].iloc[0]
                mid = row["match_id"]
                md = df_f[df_f["match_id"] == mid].copy()
                st.subheader("Worm (Cumulative Runs)")
                worm = build_worm_chart(md)
                st.altair_chart(worm, use_container_width=True)
                st.subheader("Wicket Timeline")
                w = wicket_scatter(md)
                if w is not None:
                    st.altair_chart(w, use_container_width=True)
                else:
                    st.info("No wickets recorded for this match (under current data).")

                st.subheader("Innings Summary")
                it = (md[md["valid_ball"] == 1].sort_values(["innings","team_balls"])
                        .groupby(["match_id","innings","batting_team"], as_index=False)
                        .agg(Runs=("team_runs","last"),
                             Wkts=("team_wicket","last"),
                             Balls=("team_balls","last")))
                st.dataframe(it[["innings","batting_team","Runs","Wkts","Balls"]])

    # Venues
    elif category == "Venues":
        st.header("Venue Analytics")
        va = venue_agg(match_f)
        st.subheader("Venue Summary")
        st.dataframe(va, use_container_width=True)

        venue_sel = st.selectbox("Select Venue", [""] + sorted(match_f["venue"].dropna().unique().tolist()))
        if venue_sel:
            mv = match_f[match_f["venue"] == venue_sel]
            if mv.empty:
                st.info("No matches for the selected venue under current filters.")
            else:
                st.write(f"Matches at {venue_sel}: {mv['match_id'].nunique()}")
                # distribution: batting first wins by season
                distr = mv.groupby("season", dropna=False).agg(
                    Matches=("match_id","nunique"),
                    Wins_Bat_First=("won_batting_first","sum"),
                    Wins_Chasing=("won_chasing","sum")
                ).reset_index()
                chart = alt.Chart(distr.melt(id_vars="season", value_vars=["Wins_Bat_First","Wins_Chasing"],
                                             var_name="Outcome", value_name="Count")
                                  ).mark_bar().encode(
                    x=alt.X("season:N", title="Season"),
                    y=alt.Y("Count:Q"),
                    color=alt.Color("Outcome:N", title="Outcome"),
                    tooltip=["season","Outcome","Count"]
                ).properties(height=300)
                st.altair_chart(chart, use_container_width=True)

    # Reviews/Umpire
    elif category == "Reviews/Umpire":
        st.header("DRS Reviews & Umpiring")
        rev = reviews_agg(df_f)
        st.subheader("Team Review Success")
        st.dataframe(rev, use_container_width=True)

        st.subheader("Batter-specific Reviews")
        d = df_f[~df_f["review_batter"].isna()].copy()
        if not d.empty:
            br = d.groupby("review_batter").agg(Reviews=("review_decision","count")).reset_index().sort_values("Reviews", ascending=False)
            st.dataframe(br.head(200), use_container_width=True)
        else:
            st.info("No batter review data under current filters.")

    # Step 2: Notes
    with st.expander("Step 2: Notes & Assumptions"):
        st.markdown(
            "- Team names are canonicalized (e.g., 'Royal Challengers Bangalore' → 'Royal Challengers Bengaluru').\n"
            "- Innings totals are taken from the last legal delivery of each innings using cumulative team_runs.\n"
            "- Bowling economy and maiden overs are computed from legal deliveries only.\n"
            "- Review success is parsed heuristically from review_decision text."
        )

if __name__ == "__main__":
    main()
