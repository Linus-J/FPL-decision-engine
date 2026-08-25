import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from dashboard.data.decisions import get_decision_history
from dashboard.data.simulations import (
    get_axis_effects,
    get_calibration,
    get_leaderboard,
    get_real_squad_cumulative_actual,
)
from data.db import get_session

SEASON = "2026-27"

st.set_page_config(page_title="Simulations — FPL Bot", page_icon="⚽", layout="wide")
st.title("Live Simulations")
st.caption(
    "A one-factor-at-a-time experiment: a baseline persona at the real bot's "
    "exact configuration, then each decision parameter swept individually. "
    "Every persona faces the same fixtures and the same luck, so the PAIRED "
    "difference against the baseline is the signal — not the absolute score. "
    "See docs/decision-engine.md."
)

db = get_session()
try:
    leaderboard = get_leaderboard(db, season=SEASON)
    real_actual = get_real_squad_cumulative_actual(db)
finally:
    db.close()

st.metric("Your real squad — cumulative actual points", round(real_actual, 1))

if leaderboard.empty:
    st.info(
        "No scored gameweeks yet. Personas record decisions as soon as they run, "
        "but they are only scored once the gameweek finishes and "
        "scripts/backfill_decision_outcomes.py has run."
    )
    st.stop()

# --- Which parameter settings actually did better? ------------------------

st.subheader("Effect of each swept parameter")
st.caption(
    "Δ vs baseline is the paired difference summed over the season; "
    "'GWs better' counts how many gameweeks beat the baseline, which is the "
    "more robust of the two when the season is short."
)

effects = get_axis_effects(SEASON)
if effects.empty:
    st.info("No swept personas have been scored yet.")
else:
    # An axis whose settings all scored identically renders as a row of zero
    # bars, which reads as a broken panel rather than as the real finding it
    # usually is. Early in a season most axes CANNOT have had an effect yet:
    # five of the seven act on transfers and chips, and a cold-start GW1 has
    # neither. Say so instead of drawing an empty chart.
    for axis, group in effects.groupby("swept_axis"):
        settings = len(group)
        varies = group["mean_actual"].nunique() > 1
        label = f"{axis}  ({settings} settings)" if varies else f"{axis}  (no effect yet)"
        with st.expander(label, expanded=varies):
            if not varies:
                only = group["mean_actual"].iloc[0]
                gws = int(group["gws_scored"].max())
                st.info(
                    f"All {settings} settings scored the same "
                    f"({only:g} pts/GW over {gws} scored gameweek"
                    f"{'s' if gws != 1 else ''}), so this parameter has not "
                    "separated the cohort yet.\n\n"
                    "That is usually the season being short rather than the "
                    "parameter being inert: the transfer and chip axes cannot "
                    "show anything until transfers and chips actually happen, "
                    "and a cold-start opening gameweek has neither."
                )
            else:
                chart = group.set_index("value")[["delta_vs_baseline"]]
                st.bar_chart(chart)
            st.dataframe(
                group.rename(columns={
                    "value": "Value", "mean_actual": "Mean pts/GW",
                    "delta_vs_baseline": "Δ vs baseline",
                    "gws_better_than_baseline": "GWs better",
                    "hits_taken": "Hits", "gws_scored": "GWs",
                })[["Value", "Mean pts/GW", "Δ vs baseline", "GWs better", "Hits", "GWs"]],
                hide_index=True, use_container_width=True,
            )

# --- Is the engine calibrated? --------------------------------------------

st.subheader("Calibration — predicted vs actual")
st.caption(
    "Mean points are noisy; a mean SIGNED error is not, and with ~90 personas "
    "per gameweek it converges quickly. A persistent positive bias means the "
    "engine is over-predicting, which inflates every points-denominated "
    "threshold it uses (hit cost, chip thresholds, switching cost)."
)
cal = get_calibration(SEASON)
if cal.empty:
    st.info("Nothing scored yet.")
else:
    st.line_chart(cal.set_index("gameweek")[["mean_predicted", "mean_actual"]])
    st.dataframe(
        cal.rename(columns={
            "gameweek": "GW", "personas": "Personas",
            "mean_predicted": "Predicted", "mean_actual": "Actual", "bias": "Bias",
        }),
        hide_index=True, use_container_width=True,
    )
    st.metric("Mean bias across all scored gameweeks", round(cal["bias"].mean(), 2))

# --- Leaderboard ----------------------------------------------------------

st.subheader("Leaderboard")
display_cols = {
    "rank": "Rank", "label": "Persona", "swept_axis": "Swept axis",
    "total_actual": "Actual pts", "mean_actual": "Mean/GW",
    "delta_vs_baseline": "Δ vs baseline",
    "gws_better_than_baseline": "GWs better",
    "hits_taken": "Hits", "gws_scored": "GWs scored",
}
available = [c for c in display_cols if c in leaderboard.columns]
st.dataframe(
    leaderboard[available].rename(columns=display_cols),
    hide_index=True, use_container_width=True,
)

# --- Drill down -----------------------------------------------------------

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
