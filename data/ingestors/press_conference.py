import asyncio
import json
import logging
import re
from datetime import datetime, timedelta

import aiohttp
from sqlalchemy.dialects.sqlite import insert

from data.db import get_session
from data.models import Player, PlayerPressSignal

logger = logging.getLogger(__name__)

GUARDIAN_API = "https://content.guardianapis.com/search"
GUARDIAN_KEY = "test"

POSITIVE_SIGNALS = [
    "available", "fit and available", "fully fit", "trained fully", "back in training",
    "ready to play", "could feature", "in contention", "fit to play", "expected to play",
    "no new injuries", "everyone available",
]
NEGATIVE_SIGNALS = [
    "won't feature", "will not feature", "won't be involved", "will miss",
    "not available", "out for", "ruled out", "still injured", "doubt",
    "unlikely to play", "not ready", "recovering", "awaiting scan",
]


def _score_sentence(sentence: str) -> float:
    s = sentence.lower()
    pos = sum(1 for p in POSITIVE_SIGNALS if p in s)
    neg = sum(1 for n in NEGATIVE_SIGNALS if n in s)
    if pos + neg == 0:
        return 0.0
    return (pos - neg) / (pos + neg)


def _extract_player_signals(
    body: str,
    player_name_map: dict[str, int],
) -> list[tuple[int, float, str]]:
    """Real bug found 2026-07-28 (data-completeness audit): the old version
    matched a player's bare name via plain substring containment with no
    guard at all -- a short, common name like "Gabriel" or "James" would
    silently attribute a sentence's sentiment to whichever player happened
    to iterate first in the dict, even when it was actually about a
    different, unrelated player who shares that name. This checks the
    LONGEST candidate names first (so a full "first second" match wins over
    a shorter substring of the same sentence) and uses word-boundary regex
    matching instead of raw containment; `player_name_map` itself already
    excludes any name shared by more than one real player (see
    ``_build_player_name_map``), so an inherently ambiguous short name never
    reaches this far."""
    sentences = re.split(r"[.!?]", body)
    results: list[tuple[int, float, str]] = []
    names_longest_first = sorted(player_name_map, key=len, reverse=True)

    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 10:
            continue

        lowered = sentence.lower()
        matched_player_id: int | None = None
        for name in names_longest_first:
            if re.search(rf"\b{re.escape(name)}\b", lowered):
                matched_player_id = player_name_map[name]
                break

        if matched_player_id is None:
            continue

        score = _score_sentence(sentence)
        if score != 0.0:
            results.append((matched_player_id, score, sentence[:300]))

    return results


async def _fetch_articles(
    session: aiohttp.ClientSession, from_date: str
) -> list[dict]:
    params = {
        "q": "premier league team news fitness injury",
        "section": "football",
        "show-fields": "bodyText",
        "page-size": "20",
        "api-key": GUARDIAN_KEY,
        "from-date": from_date,
        "order-by": "newest",
    }
    async with session.get(GUARDIAN_API, params=params) as r:
        r.raise_for_status()
        data = json.loads(await r.text())
    return data.get("response", {}).get("results", [])


def _build_player_name_map() -> dict[str, int]:
    """name -> player_id for scanning free-text news sentences. A name
    shared by more than one real player (e.g. two different players both
    called "Gabriel") is dropped entirely rather than resolved to whichever
    one happened to be inserted last -- see ``_extract_player_signals``."""
    db = get_session()
    try:
        players = db.query(Player).filter(Player.status != "n").all()
        candidates: dict[str, set[int]] = {}
        for p in players:
            for key in (
                p.web_name.lower(),
                p.second_name.lower(),
                f"{p.first_name} {p.second_name}".lower(),
            ):
                candidates.setdefault(key, set()).add(p.id)
        return {name: ids.pop() for name, ids in candidates.items() if len(ids) == 1}
    finally:
        db.close()


async def ingest_press_signals(lookback_days: int = 3) -> int:
    from_date = (datetime.utcnow() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    today = datetime.utcnow().strftime("%Y-%m-%d")

    player_name_map = _build_player_name_map()
    inserted = 0

    headers = {"User-Agent": "Mozilla/5.0 (compatible; fpl-bot/2.0)"}
    async with aiohttp.ClientSession(headers=headers) as session:
        try:
            articles = await _fetch_articles(session, from_date)
        except Exception as exc:
            logger.warning("Guardian API fetch failed: %s", exc)
            return 0

    db = get_session()
    try:
        for article in articles:
            body = article.get("fields", {}).get("bodyText", "")
            url = article.get("webUrl", "")
            if not body:
                continue

            signals = _extract_player_signals(body, player_name_map)
            for player_id, sentiment, raw_quote in signals:
                stmt = (
                    insert(PlayerPressSignal)
                    .values(
                        player_id=player_id,
                        scraped_date=today,
                        sentiment=sentiment,
                        raw_quote=raw_quote,
                        source_url=url,
                    )
                    .on_conflict_do_update(
                        index_elements=["player_id", "scraped_date"],
                        set_={"sentiment": sentiment, "raw_quote": raw_quote, "source_url": url},
                    )
                )
                db.execute(stmt)
                inserted += 1

        db.commit()
    finally:
        db.close()

    logger.info(
        "Press signals: %d player signals ingested from %d articles", inserted, len(articles)
    )
    return inserted


def ingest_press_signals_sync(lookback_days: int = 3) -> int:
    return asyncio.run(ingest_press_signals(lookback_days))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    n = ingest_press_signals_sync()
    print(f"Ingested {n} press signals")
