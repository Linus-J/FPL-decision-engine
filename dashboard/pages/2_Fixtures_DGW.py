import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from config.settings import settings
from dashboard.data.fixtures import get_squad_dgw_exposure, get_upcoming_fixtures
from dashboard.data.squad import get_current_squad
from data.db import get_session

st.set_page_config(page_title="Fixtures & DGW — FPL Bot", page_icon="⚽", layout="wide")
st.title("Fixtures & Double Gameweeks")

db = get_session()
try:
    fixtures = get_upcoming_fixtures(db)
    squad_ids: list[int] = []
    if settings.fpl_team_id:
        squad = get_current_squad(db, settings.fpl_team_id)
        if not squad.empty:
            squad_ids = squad["player_id"].tolist()
    coverage = get_squad_dgw_exposure(db, squad_ids) if squad_ids else {}
finally:
    db.close()

st.subheader("Upcoming fixtures")
st.caption("FDR: 1 = easiest opponent, 5 = hardest (FPL's own per-team strength rating).")
if fixtures.empty:
    st.info("No upcoming fixtures in the DB yet.")
else:
    display = fixtures.assign(
        DGW=fixtures["is_dgw"].map({True: "DGW", False: ""}),
        Kickoff=fixtures["kickoff_time"].dt.strftime("%a %d %b, %H:%M"),
    )
    st.dataframe(
        display[["gameweek", "home", "home_fdr", "away", "away_fdr", "Kickoff", "DGW"]].rename(
            columns={
                "gameweek": "GW", "home": "Home", "home_fdr": "Home FDR",
                "away": "Away", "away_fdr": "Away FDR",
            }
        ),
        hide_index=True, use_container_width=True,
    )

st.subheader("Your squad's DGW exposure")
if not squad_ids:
    st.info("No squad loaded — set FPL_TEAM_ID or log a decision first.")
elif not coverage:
    st.info("No double gameweeks within the current lookahead window.")
else:
    for gw in sorted(coverage):
        row = coverage[gw]
        st.write(
            f"**GW{gw}** — {row['squad_players_involved']} squad player(s) with a double, "
            f"combined projected xPts: {row['combined_xpts']}"
        )
