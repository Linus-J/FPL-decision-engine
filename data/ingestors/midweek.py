import logging
from datetime import timedelta

from sqlalchemy import text

from data.db import get_session

logger = logging.getLogger(__name__)

MIDWEEK_WINDOW_DAYS = 6


def compute_midweek_flags() -> dict[tuple[int, int], bool]:
    db = get_session()
    try:
        fixtures = db.execute(text("""
            SELECT id, gameweek, team_h_id, team_a_id, kickoff_time
            FROM fixtures
            WHERE kickoff_time IS NOT NULL AND gameweek IS NOT NULL
            ORDER BY kickoff_time
        """)).fetchall()
    finally:
        db.close()

    from datetime import datetime
    ko_by_team_date: dict[int, list[datetime]] = {}
    for fid, gw, th, ta, ko_str in fixtures:
        if isinstance(ko_str, str):
            ko = datetime.fromisoformat(ko_str)
        else:
            ko = ko_str
        ko_by_team_date.setdefault(th, []).append(ko)
        ko_by_team_date.setdefault(ta, []).append(ko)

    result: dict[tuple[int, int], bool] = {}
    db = get_session()
    try:
        gameweeks = db.execute(text(
            "SELECT id, deadline_time FROM gameweeks ORDER BY id"
        )).fetchall()
    finally:
        db.close()

    gw_deadlines: dict[int, object] = {}
    for gw_id, dl_str in gameweeks:
        if isinstance(dl_str, str):
            from datetime import datetime
            gw_deadlines[gw_id] = datetime.fromisoformat(dl_str)
        else:
            gw_deadlines[gw_id] = dl_str

    for gw_id, deadline in gw_deadlines.items():
        next_deadline = gw_deadlines.get(gw_id + 1)
        if next_deadline is None:
            continue
        for team_id, kickoffs in ko_by_team_date.items():
            pl_fixtures_this_gw = [
                ko for ko in kickoffs if deadline <= ko < next_deadline
            ]
            if len(pl_fixtures_this_gw) < 2:
                result[(team_id, gw_id)] = False
                continue
            sorted_ko = sorted(pl_fixtures_this_gw)
            gap = (sorted_ko[-1] - sorted_ko[0]).days
            result[(team_id, gw_id)] = gap >= 3

    return result


def get_team_midweek_flags(next_gw: int) -> dict[int, bool]:
    flags = compute_midweek_flags()
    return {team_id: has_midweek for (team_id, gw), has_midweek in flags.items() if gw == next_gw}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    flags = get_team_midweek_flags(next_gw=1)
    teams_with_midweek = [t for t, v in flags.items() if v]
    print(f"Teams with midweek fixtures GW1: {teams_with_midweek}")
