import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from config.settings import settings
from dashboard.data.news import get_injury_news
from dashboard.data.squad import get_current_squad
from data.db import get_session

st.set_page_config(page_title="Injury News — FPL Bot", page_icon="⚽", layout="wide")
st.title("Injury & Availability News")

db = get_session()
try:
    squad_ids: list[int] = []
    if settings.fpl_team_id:
        squad = get_current_squad(db, settings.fpl_team_id)
        if not squad.empty:
            squad_ids = squad["player_id"].tolist()
    news = get_injury_news(db, squad_ids)
finally:
    db.close()

st.caption(
    "Transfer market rumours are not tracked yet — no reliable free source exists. "
    "This shows only FPL's own injury/availability status field."
)

display_cols = {
    "web_name": "Player", "position": "Pos", "team_short": "Team",
    "status": "Status", "news": "News", "chance_of_playing_next_round": "% chance next GW",
}

if news.empty:
    st.info("No injury/availability news right now.")
else:
    st.subheader("In your squad")
    squad_news = news[news["in_squad"]]
    if squad_news.empty:
        st.write("No news affecting your squad.")
    else:
        st.dataframe(
            squad_news[list(display_cols)].rename(columns=display_cols),
            hide_index=True, use_container_width=True,
        )

    st.subheader("League-wide")
    st.dataframe(
        news[~news["in_squad"]][list(display_cols)].rename(columns=display_cols),
        hide_index=True, use_container_width=True,
    )
