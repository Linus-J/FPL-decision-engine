import logging
import re

from sqlalchemy import text

from data.db import get_session

logger = logging.getLogger(__name__)

_SEVERITY_3 = re.compile(
    r"season[\s-]ending|rest of the season|long[\s-]term|surgery|torn|ruptur|fractur|broken",
    re.IGNORECASE,
)
_SEVERITY_2 = re.compile(
    r"\b(\d+)[\s-]*(week|month)s?\b|expected back|hamstring|knee|ankle|thigh|calf|hip|groin|shoulder|back injury|muscle injury",
    re.IGNORECASE,
)
_SEVERITY_1 = re.compile(
    r"knock|minor|tight|fatigue|match fitness|illness|cold|precaution|doubt|50%|75%|25%",
    re.IGNORECASE,
)
_SEVERITY_0_OVERRIDE = re.compile(
    r"has joined|on loan|permanently|transferred",
    re.IGNORECASE,
)


def _parse_severity(news: str) -> int:
    if not news:
        return 0
    if _SEVERITY_0_OVERRIDE.search(news):
        return 0
    if _SEVERITY_3.search(news):
        return 3
    if _SEVERITY_2.search(news):
        return 2
    if _SEVERITY_1.search(news):
        return 1
    return 0


def run_injury_parser() -> int:
    db = get_session()
    updated = 0
    try:
        rows = db.execute(text("SELECT id, news, status FROM players")).fetchall()
        for player_id, news, status in rows:
            if status in ("i", "u"):
                severity = max(_parse_severity(news or ""), 2)
            elif status == "d":
                severity = max(_parse_severity(news or ""), 1)
            else:
                severity = _parse_severity(news or "")

            db.execute(
                text("UPDATE players SET injury_severity = :sev WHERE id = :pid"),
                {"sev": severity, "pid": player_id},
            )
            updated += 1
        db.commit()
    finally:
        db.close()
    logger.info("Injury parser: updated %d players", updated)
    return updated


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_injury_parser()
