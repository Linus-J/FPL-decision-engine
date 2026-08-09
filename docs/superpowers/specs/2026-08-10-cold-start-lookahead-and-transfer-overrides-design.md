# Cold-start fixture lookahead + manual transfer/rumour overrides

**Date:** 2026-08-10
**Status:** Approved for implementation

## Motivation

Two confirmed gaps in the pre-season / GW1 squad-building path, found while investigating why repeated cold-start runs kept converging on similar squads:

1. **No fixture lookahead.** `projection/cold_start.py::project_cold_start` scores every player at `target_gw=1` only, from prior-season/peer historical data. `build_initial_squad` calls `optimise_squad(..., horizon=1, ...)`, hardcoded. A squad optimal for GW1 alone can load up on players whose GW2-6 fixtures are brutal (or miss out on ones like Haaland whose price is justified by a strong opening run) — the optimiser has no way to know.
2. **`team_id` is trusted unconditionally from the FPL API**, with no correction mechanism. A confirmed-but-not-yet-FPL-registered transfer (e.g. a summer signing before FPL updates the tag) is invisible to the max-3-per-club constraint and to fixture attribution. There's also no mechanism for flagging a *rumoured* move (relevant mid-season, e.g. the January window) even though `optimiser/departure_risk.py` already implements the exact graduated-discount logic needed — it's just never been fed real data (`p_leave_by_player` is always empty today).

## Feature A: cold-start fixture lookahead

### Data flow

`project_cold_start` gains a new `horizon: int = 1` parameter (default preserves today's exact single-row behaviour for existing callers, e.g. `tests/test_cold_start.py`). When `horizon > 1`, it emits one row per `(player, gw)` for `gw` in `[target_gw, target_gw + horizon)` instead of one row per player. `build_initial_squad` passes `horizon=cfg.cold_start_lookahead_gws` (new `OptimiserConfig` field, default `5`, matching the existing precedent of `optimiser/chips.py`'s `wildcard_eval_horizon_gws`) through to `project_cold_start`.

Each GW's `xpts`/`xpts_var` is the GW1 base value scaled by a fixture-difficulty multiplier for that specific gameweek's opponent:

```
xpts(gw)     = base_xpts * fixture_multiplier(opp_defence_strength(gw), was_home(gw))
xpts_var(gw) = base_xpts_var * fixture_multiplier(gw) ** 2   # keeps CV constant across GWs
```

`fixture_multiplier` is the existing, already-tested `projection/fixture_adjust.py::fixture_multiplier(opp_defence_strength, was_home)` — reused as-is, not reimplemented.

### New helper: `projection/cold_start.py::load_horizon_fixtures`

For each player's (corrected, post-override — see Feature B) `team_id`, look up their opponent and home/away status for each of the next `N` gameweeks from the `fixtures` table (same join pattern as `projection/pipeline.py::_build_live_fixture_context`, reused rather than duplicated where practical). For each fixture, resolve `opp_defence_strength`:

1. Current season's `TeamSeasonStrength.strength_defence_home/away` for the opponent, if non-zero.
2. **Fallback: last season's `TeamSeasonStrength` for the same club**, joined on the stable `code` field (not `team_id`, which is a per-season alphabetical index that shifts under promotion/relegation — same reasoning `load_prior_league_lookup` already uses for players). This exists specifically because as of 2026-08-10, `strength_defence_home/away` for 2026-27 is still `0` for every club (FPL hasn't published it yet this pre-season), while `strength_overall_home/away` *is* populated but on an incompatible scale — using it directly would actively mislead `fixture_multiplier`, not just fail to help, so it must not be used as a substitute.
3. If neither exists (new to the top flight and no prior-season row — shouldn't happen for the current 20 clubs, but degrade safely): `None`, which `fixture_multiplier` already treats as neutral (`1.0`).

Promoted-team fixtures resolve fine — the fallback lookup keys on the same `code`-based cross-season join `load_prior_league_lookup` uses, and if a club has no 2025-26 `TeamSeasonStrength` row at all (freshly promoted, e.g. from Championship), it falls through to the neutral-multiplier case above, same as case 3.

### `build_initial_squad`

Changes `optimise_squad(..., horizon=1, ...)` to `optimise_squad(..., horizon=cfg.cold_start_lookahead_gws, ...)`. No other change needed — `optimiser/squad.py::_multi_gw_xpts`/`_multi_gw_var` already sum across whatever horizon of projection rows they're given.

### Config addition (`config/strategy.py`, `OptimiserConfig`)

```python
# GWs to look ahead when building the GW1/pre-season initial squad
# (fixture-difficulty-weighted, not just single-GW xPts).
cold_start_lookahead_gws: int = 5
```

### Testing

- Unit test: `project_cold_start` with a synthetic fixture list produces distinct `xpts` per GW for the same player, tracking a hand-computed `fixture_multiplier`.
- Unit test: current-season-zero-strength → prior-season fallback resolves correctly for an established club; a club with no prior-season row degrades to neutral without crashing.
- Unit test: `build_initial_squad` end-to-end produces a `SquadSolution` whose `total_xpts` reflects the horizon sum, not a single GW.
- Existing `tests/test_cold_start.py` cases must keep passing with `cold_start_lookahead_gws` reduced to `1` (regression: confirms the change is backward-compatible when the horizon is 1).

## Feature B: manual team/rumour overrides

### Storage: `config/transfer_overrides.yaml`

Version-controlled, hand-edited, keyed by the player's stable FPL `code` (survives transfers; `id` does not always):

```yaml
confirmed:
  - code: 123456
    team_id: 1
    reason: "Signed from Newcastle, not yet reflected in FPL team_id"
    as_of: "2026-08-10"

rumoured:
  - code: 234567
    p_leave: 0.35
    reason: "Strongly linked to a January move per <source>"
    as_of: "2026-08-10"
```

### New module: `data/overrides.py`

- `load_team_overrides() -> dict[int, int]` — `code -> team_id`, parsed from the `confirmed` list. Empty dict if the file is missing or the list is empty (never crashes).
- `apply_team_overrides(players: pd.DataFrame) -> pd.DataFrame` — returns a copy with `team_id` replaced wherever `players.code` matches an override entry. Applied to the candidate pool BEFORE the max-3-per-club constraint and BEFORE Feature A's fixture lookahead, so both see the corrected club.
- `load_p_leave_overrides() -> dict[int, float]` — `player_id -> p_leave`, resolved from the `rumoured` list's `code`s against the current `players` table (a `code` with no matching current player is skipped, logged at `warning`, never crashes).

### Integration point

Both `projection/cold_start.py::load_current_players` and `agent/decision_engine.py::_load_players` — the two places that build the live candidate pool — call `apply_team_overrides` immediately after loading. This is the single shared seam, so cold-start and in-season transfer decisions (including the already-anticipated `DepartureRiskRules.january_window_start_gw/end_gw` mini-replan window) get identical treatment without duplicated logic.

`load_p_leave_overrides()`'s output feeds `optimiser/departure_risk.py::apply_departure_discount` at the same call sites `hard_excluded_ids`/`confirmed_p_leave` are already used from (`optimiser/transfers.py`) — no new discount math, just a real input where today `p_leave_by_player` is always empty.

### Rumour flagging

Whenever a built squad includes a player present in `rumoured`, log a `warning` naming the player and the `reason`/`as_of` from the YAML entry. This is deliberately just a log line for this pass (dashboard surfacing is a natural follow-up, not blocking).

### Testing

- Unit test: `apply_team_overrides` replaces `team_id` for a matched `code`, leaves everything else untouched, no-ops on an empty/missing file.
- Unit test: `load_p_leave_overrides` resolves `code -> player_id -> p_leave` correctly; skips (with a warning, not a crash) a `code` absent from the current player table.
- Integration test: a squad built with a `confirmed` override for a 4th player at an already-3-player club is rejected/rebalanced correctly by the existing max-3-per-club constraint (proves the override is visible to the ILP, not just cosmetic).
- Integration test: a `rumoured` entry above `rumour_floor_p_leave` measurably lowers that player's `effective_score` versus the same run without the override.

## Future work (explicitly out of scope for this pass)

Live-filling `transfer_overrides.yaml` instead of hand-editing it. Two free, no-new-budget paths were identified and are worth a follow-up spec once the manual mechanism has been used for a season:

1. **FPL's own `players.news` field already contains transfer language** ("has joined", "on loan", "permanently", "transferred" — `data/ingestors/injury_parser.py` already regex-matches this today, just to *zero out* injury severity, not to detect a club change). A small extension could flag these as candidate `confirmed` entries for human confirmation before being trusted (a wrong automatic team assignment is a worse failure mode than a missed one, so auto-apply without review is not recommended).
2. **The Guardian API scrape (`data/ingestors/press_conference.py`) already does free, per-player-attributed sentence sentiment scoring** for injury news. The same fetch-and-attribute pattern could run a second keyword set ("linked with", "in talks", "medical", "here we go", "agreed personal terms") to produce candidate `rumoured` entries.

Both would write into the same YAML shape defined above (or a mirrored DB table), so nothing in Feature B needs to change to support it later.

## Out of scope

- Automated news/rumour ingestion (see Future Work above).
- Dashboard surfacing of rumour flags (log-only for this pass).
- Backfilling `TeamSeasonStrength.strength_attack_home/away` (also currently 0 for 2026-27) — only `strength_defence_home/away` is read by Feature A, since `fixture_multiplier` only takes a defence-strength input today.
