import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
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

no_projections = squad["xpts"].isna().all()

starting = squad[squad["is_starting"]].sort_values("xpts", ascending=False, na_position="last")
bench = squad[~squad["is_starting"]].sort_values("xpts", ascending=False, na_position="last")

if no_projections:
    fallback_total = squad.attrs.get("fallback_projected_total")
    if fallback_total is not None:
        st.metric("Projected XI xPts (at decision time)", round(float(fallback_total), 1))
    st.caption(
        "Per-player xPts aren't available yet — this is a true pre-season cold-start "
        "squad and the projection pipeline hasn't produced per-player projections for "
        "this gameweek (it needs current-season played history first). This will fill "
        "in once the season starts."
    )
else:
    total_xi_xpts = float(starting["xpts"].sum())
    if "is_captain" in squad.columns:
        captain_row = squad[squad["is_captain"]]
        if not captain_row.empty and pd.notna(captain_row["xpts"].iloc[0]):
            total_xi_xpts += float(captain_row["xpts"].iloc[0])
    st.metric("Projected XI xPts (captain doubled)", round(total_xi_xpts, 1))

display_cols = {
    "web_name": "Player", "position": "Pos", "team_short": "Team",
    "now_cost": "£m", "xpts": "xPts",
}


def _display(df: pd.DataFrame) -> pd.DataFrame:
    out = df[list(display_cols)].rename(columns=display_cols)
    out["xPts"] = out["xPts"].map(lambda v: "–" if pd.isna(v) else round(v, 1))
    return out


st.subheader("Starting XI")
st.dataframe(_display(starting), hide_index=True, use_container_width=True)

st.subheader("Bench")
st.dataframe(_display(bench), hide_index=True, use_container_width=True)
