import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from config.settings import settings
from projection.pipeline import _get_current_and_next_gw, _get_current_season

st.set_page_config(page_title="FPL Bot Dashboard", page_icon="⚽", layout="wide")

st.title("FPL 2026/27 Bot Dashboard")

current_gw, next_gw = _get_current_and_next_gw()
season = _get_current_season()

col1, col2, col3 = st.columns(3)
col1.metric("Season", season)
col2.metric("Current GW", current_gw)
col3.metric("Next GW", next_gw)

st.write(
    "Use the sidebar to navigate: **Squad**, **Fixtures & DGW**, **Injury News**, "
    "**Decision History**, **Chip Plan**. Everything here is read-only and reflects "
    "data as of the last agent run — this dashboard never triggers ingestion or submissions."
)

if not settings.fpl_team_id:
    st.warning(
        "FPL_TEAM_ID is not set in .env — the Squad and Fixtures pages "
        "need it to fetch your live picks."
    )
