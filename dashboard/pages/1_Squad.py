import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from config.settings import settings
from dashboard.data.squad import get_current_squad
from data.db import get_session

st.set_page_config(page_title="Squad — FPL Bot", page_icon="⚽", layout="wide")
st.title("Current Squad")

if not settings.fpl_team_id:
    st.error("FPL_TEAM_ID is not set in .env.")
    st.stop()

db = get_session()
try:
    squad = get_current_squad(db, settings.fpl_team_id)
finally:
    db.close()

if squad.empty:
    st.info("No squad data available yet — no live FPL picks and no logged decision found.")
    st.stop()

gw = int(squad["gameweek"].iloc[0])
st.caption(f"Showing squad for GW{gw}")

starting = squad[squad["is_starting"]].sort_values("xpts", ascending=False)
bench = squad[~squad["is_starting"]].sort_values("xpts", ascending=False)

total_xi_xpts = float(starting["xpts"].sum())
if "is_captain" in squad.columns:
    captain_row = squad[squad["is_captain"]]
    if not captain_row.empty:
        total_xi_xpts += float(captain_row["xpts"].iloc[0])

st.metric("Projected XI xPts (captain doubled)", round(total_xi_xpts, 1))

display_cols = {
    "web_name": "Player", "position": "Pos", "team_short": "Team",
    "now_cost": "£m", "xpts": "xPts",
}

st.subheader("Starting XI")
st.dataframe(
    starting[list(display_cols)].rename(columns=display_cols),
    hide_index=True, use_container_width=True,
)

st.subheader("Bench")
st.dataframe(
    bench[list(display_cols)].rename(columns=display_cols),
    hide_index=True, use_container_width=True,
)
