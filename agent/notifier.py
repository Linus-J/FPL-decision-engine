import logging

import aiohttp

from config.settings import settings

logger = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"


async def _send(text: str) -> None:
    if not settings.telegram_bot_token or not settings.telegram_chat_id:
        logger.warning("Telegram not configured — skipping notification")
        return

    url = TELEGRAM_API.format(token=settings.telegram_bot_token, method="sendMessage")
    payload = {
        "chat_id": settings.telegram_chat_id,
        "text": text,
        "parse_mode": "HTML",
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                body = await resp.text()
                logger.error("Telegram send failed: HTTP %d — %s", resp.status, body)
            else:
                logger.info("Telegram notification sent")


def _chip_emoji(chip: str | None) -> str:
    return {
        "wildcard": "🃏",
        "freehit": "🎯",
        "bboost": "📈",
        "3xc": "3️⃣✖️",
    }.get(chip or "", "")


def _status_emoji(dry_run: bool) -> str:
    return "🔕 DRY RUN" if dry_run else "✅ LIVE"


def format_decision_message(decision: dict) -> str:
    gw = decision.get("gameweek", "?")
    dry_run = decision.get("dry_run", True)
    chip = decision.get("chip")
    transfers_in = decision.get("transfers_in", [])
    transfers_out = decision.get("transfers_out", [])
    hits = decision.get("hits_taken", 0)
    net_gain = decision.get("net_xpts_gain", 0.0)
    total_xpts = decision.get("total_xpts", 0.0)
    total_cost = decision.get("total_cost", 0.0)
    dgw_coverage = decision.get("dgw_coverage", 0)

    squad = decision.get("squad", [])
    captain = next((p for p in squad if p.get("is_captain")), None)
    vice = next((p for p in squad if p.get("is_vice_captain")), None)

    lines = [
        f"<b>FPL GW{gw} Decision {_status_emoji(dry_run)}</b>",
        "",
    ]

    if chip:
        lines.append(f"{_chip_emoji(chip)} <b>Chip:</b> {chip.upper()}")
        lines.append("")

    if transfers_in:
        lines.append(f"<b>Transfers ({len(transfers_in)} in{f', -{hits*4}pts hit' if hits else ''}):</b>")
        for t_in, t_out in zip(transfers_in, transfers_out):
            lines.append(f"  ➡️ {t_in['web_name']} (£{t_in['cost']}m) ← {t_out['web_name']}")
        lines.append(f"  📊 Net xPts gain: <b>+{net_gain:.1f}</b>")
        lines.append("")
    else:
        lines.append("↩️ No transfers")
        lines.append("")

    if captain:
        lines.append(f"⭐ <b>Captain:</b> {captain['web_name']} ({captain['position']} £{captain['now_cost']}m)")
    if vice:
        lines.append(f"🔸 <b>Vice:</b> {vice['web_name']} ({vice['position']} £{vice['now_cost']}m)")
    lines.append("")

    starting = [p for p in squad if p.get("is_starting") and not p.get("is_captain")]
    by_pos = {}
    for p in starting:
        by_pos.setdefault(p["position"], []).append(p["web_name"])

    lines.append("<b>Starting XI:</b>")
    for pos in ("GKP", "DEF", "MID", "FWD"):
        if pos in by_pos:
            lines.append(f"  {pos}: {', '.join(by_pos[pos])}")
    lines.append("")

    bench = sorted(
        [p for p in squad if not p.get("is_starting")],
        key=lambda p: p.get("bench_order", 99),
    )
    if bench:
        bench_names = ", ".join(p["web_name"] for p in bench)
        lines.append(f"<b>Bench:</b> {bench_names}")
        lines.append("")

    lines.append(f"💰 Squad cost: £{total_cost}m")
    lines.append(f"📐 Projected xPts: <b>{total_xpts:.1f}</b>")
    if dgw_coverage:
        lines.append(f"🔁 DGW coverage: {dgw_coverage} players")

    return "\n".join(lines)


async def notify(decision: dict) -> None:
    message = format_decision_message(decision)
    await _send(message)


def notify_sync(decision: dict) -> None:
    import asyncio
    asyncio.run(notify(decision))
