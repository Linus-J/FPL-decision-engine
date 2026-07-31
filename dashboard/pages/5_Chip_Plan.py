import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from dashboard.data.decisions import get_latest_chip_plan, get_latest_transfer_plan
from data.db import get_session

st.set_page_config(page_title="Chip Plan — FPL Bot", page_icon="⚽", layout="wide")
st.title("Next Chip & Transfer Plan")

st.caption(
    "Display only — shows what the agent's decision engine already recommended "
    "on its last run. This page never re-runs the optimiser."
)

db = get_session()
try:
    chip_plan = get_latest_chip_plan(db)
    transfer_plan = get_latest_transfer_plan(db)
finally:
    db.close()

st.subheader("Chip")
if not chip_plan or not chip_plan["chip"]:
    st.write("No chip currently recommended.")
else:
    st.write(f"**GW{chip_plan['gameweek']}**: {chip_plan['chip']} — {chip_plan['reason']}")
    st.write(f"Expected gain: {chip_plan['expected_gain']:.2f} xPts")

st.subheader("Transfers")
if not transfer_plan or (not transfer_plan["transfers_in"] and not transfer_plan["transfers_out"]):
    st.write("No transfers currently planned.")
else:
    st.write(
        f"**GW{transfer_plan['gameweek']}** — hits taken: {transfer_plan['hits_taken']}, "
        f"net xPts gain: {transfer_plan['net_xpts_gain']:.2f}"
    )
    col1, col2 = st.columns(2)
    with col1:
        st.write("In")
        for p in transfer_plan["transfers_in"]:
            st.write(f"- {p['web_name']} (£{p['cost']}m)")
    with col2:
        st.write("Out")
        for p in transfer_plan["transfers_out"]:
            st.write(f"- {p['web_name']} (£{p['cost']}m)")
