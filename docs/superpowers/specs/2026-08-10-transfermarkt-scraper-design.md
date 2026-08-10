# Transfermarkt scraper: auto-fill confirmed transfers + rumour candidates

**Date:** 2026-08-10
**Status:** Approved for implementation

## Motivation

The cold-start-lookahead-and-transfer-overrides feature (2026-08-10, same day) added `config/transfer_overrides.yaml` as a hand-edited mechanism for correcting a player's `team_id` ahead of FPL's own API, and for flagging rumoured departures. The user's own explicit ask when approving that feature: "ideally we would use some sort of data source to live fill this info... we should check carefully if this is possible since it is what I will be doing manually anyway."

A feasibility prototype (throwaway script, not committed) confirmed both of the user's proposed Transfermarkt pages are scrapable with plain `httpx` + `BeautifulSoup` — no headless browser needed (unlike FBref, which required a full Selenium/Chrome stack to bypass Cloudflare):

- **Transfers page** (`/premier-league/transfers/wettbewerb/GB1/plus/?saison_id=<year>&s_w=s&leihe=1&intern=0&intern=1`): real transfer data is server-rendered under `div.responsive-table > table` elements (no distinguishing class on the `<table>` itself), grouped into one table per destination club. Confirmed against the live page: Bruno Guimarães' Newcastle→Arsenal move (the user's own motivating example) appears as the first row of Arsenal's incoming-transfers table.
- **Rumours page** (`/premier-league/geruechte/wettbewerb/GB1`): a `table.items`-class table with columns `Player, Nation, Age, Club, Interested club, Most recent source from, Assessment`. The `Assessment` column (a credibility percentage, e.g. `71 %`, `53 %`, or `-` when unrated) is present in the base HTML — no AJAX sort request needed to read or sort by it ourselves.
- `robots.txt` disallows only the `wget` user-agent specifically; `User-agent: * / Allow: /` covers everything else, including a normal browser-like `httpx` client. No `Crawl-delay` applies to a generic client (only `Slurp`/Yahoo gets `Crawl-delay: 2`), but this design follows it anyway as a courtesy since request volume is already low (manual, on-demand runs only).

One design point flagged honestly: which specific DOM element carries each transfers-table's destination-club name (needed to know which PL club a `responsive-table` block belongs to) was not conclusively identified in the prototype — the prototype confirmed the ROWS are real and parseable, not the exact club-name association. This is implementation-detail work for the plan/build phase, not a feasibility risk (the data is visibly present on the page; it's a selector-finding task).

## Scope decisions (from user Q&A during brainstorming)

1. **Confirmed transfers: auto-apply.** Concrete, dated, factual transactions — once a player is confidently matched to our `code` and their reported new club differs from `players.team_id`, write directly into `config/transfer_overrides.yaml`'s `confirmed` list. No human gate (distinct from rumours, which stay a review queue).
2. **Rumours: human-reviewed candidates file, never auto-applied.** Matches the original design's stance ("a wrong automatic team assignment is worse than a missed one") — rumours are inherently uncertain even at a high Assessment score.
3. **Run cadence: manual, on-demand.** Matches this codebase's existing convention — `press_conference.py`/`injury_parser.py` are also standalone scripts, not wired into `scripts/run_weekly.py`'s automated pipeline. Wiring in later is a trivial follow-up (another `_run_or_warn` call) if the user wants it, explicitly out of scope for this pass.
4. **Rumour filtering: real Assessment percentage above a fixed floor only.** `-` (unrated) rows are dropped entirely — no credibility signal yet, too noisy to act on. A named constant (`_MIN_ASSESSMENT_PCT`, default `40`) gates which rows reach the candidates file at all, following this codebase's existing "named constant, not a magic number" convention (e.g. `MIN_PRIOR_APPEARANCES`, `_MIN_BUCKET_SAMPLES` in `cold_start.py`).
5. **Name matching: reuse `press_conference.py`'s exact conservative pattern.** Three name variants tried (`web_name`, `second_name`, `first_name + second_name`, all lowercased); any name shared by more than one current player is dropped from the matchable set entirely (never guessed); an unmatched or dropped name is skipped and logged at `warning`, never silently ignored.

## Module: `data/ingestors/transfermarkt.py`

Same shape as `press_conference.py`: a fetch layer, a parse layer, and a sync/write layer, kept separable so each can be tested independently of the network.

```python
HEADERS = {"User-Agent": "Mozilla/5.0 (...) Chrome/120.0 Safari/537.36", ...}
TRANSFERS_URL = "https://www.transfermarkt.com/premier-league/transfers/wettbewerb/GB1/plus/?saison_id={year}&s_w=s&leihe=1&intern=0&intern=1"
RUMOURS_URL = "https://www.transfermarkt.com/premier-league/geruechte/wettbewerb/GB1"

def _build_player_name_map() -> dict[str, int]:
    """name -> code, reusing press_conference.py's exact ambiguous-name-drop
    pattern (see that module's ``_build_player_name_map``) -- adapted to
    key on `code` (the stable cross-transfer identity Feature B's override
    mechanism needs) rather than `player_id`."""

def scrape_confirmed_transfers(html: str, name_map: dict[str, int], pl_team_ids: dict[str, int]) -> list[dict]:
    """Pure function: HTML in, candidate {code, team_id, reason, as_of}
    dicts out (Feature B's exact `confirmed` entry shape). No I/O -- the
    caller fetches the HTML and calls this, so tests feed it fixture HTML
    directly, no network mocking needed. `pl_team_ids`: club display-name ->
    our internal team_id, resolved from the `teams` table -- a transfer
    into a club we can't resolve (not currently in the PL) is skipped, not
    guessed."""

def scrape_rumours(html: str, name_map: dict[str, int]) -> list[dict]:
    """Pure function: HTML in, candidate {code, p_leave, reason, as_of}
    dicts out (Feature B's exact `rumoured` entry shape), filtered to
    current-club-is-PL and Assessment >= _MIN_ASSESSMENT_PCT, sorted
    Assessment descending."""

def sync_confirmed_overrides(candidates: list[dict]) -> None:
    """Reads config/transfer_overrides.yaml, merges `candidates` into its
    `confirmed` list under `source: transfermarkt` tagging, removes any
    existing `source: transfermarkt` entry whose team_id now already
    matches the live `players.team_id` (FPL caught up, override is
    redundant), writes the file back. NEVER touches an entry without
    `source: transfermarkt` (a hand-written entry is untouchable by this
    function, even if it looks stale) -- idempotent: rerunning with the
    same candidates produces a byte-identical file."""

def write_rumour_candidates(candidates: list[dict]) -> None:
    """Fully regenerates config/transfer_overrides_candidates.yaml (gitignored)
    from `candidates` -- no merge/idempotency logic needed, since nothing
    here is auto-applied; each run is a fresh snapshot for the user to
    review and hand-copy into transfer_overrides.yaml themselves."""
```

`config/transfer_overrides.yaml`'s `confirmed` entry shape gains one new optional field: `source` (absent = hand-written, `transfermarkt` = scraper-written) plus the existing `reason`/`as_of` fields, populated as e.g. `reason: "Transfermarkt: transferred to Arsenal"`, `as_of: <today's date, the scrape date>`. `data/overrides.py::load_team_overrides` is unaffected — it already reads only `code`/`team_id` and ignores unknown keys.

`config/transfer_overrides_candidates.yaml` is a **new, gitignored** file (add to `.gitignore`) — scraped/derived data, not something to version-control, with a header comment explaining it's a review queue: copy entries you trust into `transfer_overrides.yaml`'s `rumoured` list yourself.

## Error handling

Same "never crash the caller" posture as every other ingestor in this codebase: a network failure (timeout, non-200, connection error) is caught, logged at `warning`, and the affected scrape function returns an empty list — the other scrape function (transfers vs rumours) still runs independently. An unexpected HTML structure (a selector that used to match now matches nothing) degrades to an empty result with a `warning`, not an exception — Transfermarkt redesigns their site periodically, and a broken selector should never take down whatever else is running.

## Testing

- Fixture-HTML unit tests for `scrape_confirmed_transfers`/`scrape_rumours` (a trimmed real snippet captured during implementation, not a live network call) — covering: a real match producing a candidate, a name that doesn't match any current player (skipped), a name shared by two current players (skipped, logged), a club not resolvable to a `team_id` (skipped), a rumour below `_MIN_ASSESSMENT_PCT` (dropped), a rumour with `-` assessment (dropped), a rumour whose current club isn't PL (dropped).
- `sync_confirmed_overrides`: idempotency (two consecutive calls with identical candidates produce a byte-identical file), self-cleanup (a `source: transfermarkt` entry is removed once `players.team_id` matches; a hand-written entry with no `source` field survives even when its `team_id` looks stale relative to live data).
- `write_rumour_candidates`: sorts by Assessment descending, fully overwrites on rerun (no accumulation).
- `_build_player_name_map`: mirrors `press_conference.py`'s own test shape for the ambiguous-name-drop behavior.

## Out of scope

- Wiring into `scripts/run_weekly.py`'s automated pipeline (manual/on-demand only, per the user's explicit choice).
- Any dashboard/UI surfacing of `transfer_overrides_candidates.yaml`.
- Treating a rumour's "Interested club" (potential destination) as a new-signing candidate — only current-club departure risk is in scope, matching the original Feature B design's rumoured-tier intent (existing squad risk, not scouting).
- Resolving the exact destination-club-name DOM selector for the transfers page (confirmed present on the page, not yet pinned to a specific element) — implementation-plan-level work, not a design ambiguity.
