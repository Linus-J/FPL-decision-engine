import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import streamlit as st

from dashboard.data.decisions import get_decision_history
from data.db import get_session

st.set_page_config(page_title="Decision History — FPL Bot", page_icon="⚽", layout="wide")
st.title("Decision History")

db = get_session()
try:
    history = get_decision_history(db)
finally:
    db.close()

if history.empty:
    st.info("No decisions logged yet.")
    st.stop()

lineup_rows = history[history["decision_type"] == "lineup"].sort_values("gameweek")
if not lineup_rows.empty:
    st.subheader("Projected vs actual (starting XI, by gameweek)")
    chart_df = lineup_rows[["gameweek", "projected_gain", "actual_outcome"]].rename(
        columns={"projected_gain": "Projected", "actual_outcome": "Actual"}
    ).set_index("gameweek")
    st.line_chart(chart_df)
    if lineup_rows["actual_outcome"].isna().any():
        st.caption(
            "Actual outcomes appear once a gameweek finishes and "
            "`scripts/backfill_decision_outcomes.py` has run."
        )

st.subheader("Full decision log")
for gw, group in history.groupby("gameweek", sort=False):
    with st.expander(f"Gameweek {gw}"):
        for _, row in group.iterrows():
            line = f"**{row['decision_type']}** — projected: {row['projected_gain']:.2f}"
            if pd.notna(row["actual_outcome"]):
                line += f", actual: {row['actual_outcome']:.0f}"
            line += " (dry run)" if row["dry_run"] else " (live)"
            st.write(line)
            st.json(row["details"])
