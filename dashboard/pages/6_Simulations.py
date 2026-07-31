import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from dashboard.data.decisions import get_decision_history
from dashboard.data.simulations import get_leaderboard, get_real_squad_cumulative_actual
from data.db import get_session

st.set_page_config(page_title="Simulations — FPL Bot", page_icon="⚽", layout="wide")
st.title("Live Simulations")
st.caption(
    "~100 shadow managers varying risk posture, stepped forward alongside the "
    "real squad every scheduled cycle -- never submitted to the real FPL app. "
    "See plan/simulation-engine-v1.md."
)

db = get_session()
try:
    leaderboard = get_leaderboard(db, season="2026-27")
    real_actual = get_real_squad_cumulative_actual(db)
finally:
    db.close()

st.metric("Your real squad — cumulative actual points", round(real_actual, 1))

if leaderboard.empty:
    st.info("No simulations have run yet.")
    st.stop()

st.subheader("Leaderboard")
display_cols = {
    "rank": "Rank", "label": "Persona", "risk_mode": "Risk mode",
    "variance_weight": "Variance wt", "max_ownership_differential": "EO wt",
    "chip_aggressiveness": "Chip aggr.", "cumulative_actual": "Actual pts",
    "gws_scored": "GWs scored",
}
st.dataframe(
    leaderboard[list(display_cols)].rename(columns=display_cols),
    hide_index=True, use_container_width=True,
)

st.subheader("Drill down into a persona")
selected_label = st.selectbox("Persona", leaderboard["label"].tolist())
selected_id = int(leaderboard.loc[leaderboard["label"] == selected_label, "id"].iloc[0])

db = get_session()
try:
    history = get_decision_history(db, sim_manager_id=selected_id)
finally:
    db.close()

if history.empty:
    st.info("No decisions logged yet for this persona.")
else:
    for gw, group in history.groupby("gameweek", sort=False):
        with st.expander(f"Gameweek {gw}"):
            for _, row in group.iterrows():
                line = f"**{row['decision_type']}** — projected: {row['projected_gain']:.2f}"
                if pd.notna(row["actual_outcome"]):
                    line += f", actual: {row['actual_outcome']:.0f}"
                st.write(line)
                st.json(row["details"])
