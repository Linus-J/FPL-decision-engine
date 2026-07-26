"""bonus.py — P8 bonus component (Monte-Carlo, reduced-BPS).

Bonus is a fixture-relative rank (top-3 BPS → 3/2/1), so it can't be an analytic
per-player expectation — it's sampled per fixture inside the P10 assembly, over
each player's drawn events, using the 26/27 BPS simulator (`bps_sim`, T5a).

We only model a SUBSET of the ~33 BPS inputs (the ones our components + data
provide): appearance/goals/assists/clean-sheet/saves/CBIRT/key-passes/cards.
The rest (big chances, crosses, dribbles, pass-completion tiers, etc.) are 0 —
a documented **reduced-BPS** approximation. `reduced_full_agreement` measures
how much that reduction changes the awarded bonus vs the full-event recompute
(T5b), so the bias is known, not assumed. Pure + deterministic.
"""

from __future__ import annotations

from collections.abc import Mapping

from projection.bps_sim import award_bonus, compute_fixture_bonus, compute_player_bps

# BPS inputs our projection can populate (P1/P3–P7 + Understat key-passes + P9
# cards). Everything else the simulator reads defaults to 0 → reduced BPS.
MODELLED_BPS_FIELDS = frozenset({
    "position", "minutes", "goals", "assists", "clean_sheet", "saves",
    "clearances", "blocks", "interceptions", "tackles", "recoveries",
    "key_passes", "yellow_cards", "red_cards", "dribbles",
})

# Empirically calibrated 2026-07-26 against real 25/26 bonus-by-position
# rates (player_gw_stats.bonus, minutes>0). Originally 0.15, set when
# DEF/MID/FWD's competing defensive-action channel was crippled by FBref's
# free summary table lacking clearances/blocks/recoveries entirely (GK
# P(bonus>0) was 32.5% modelled vs 11.0% real, ~3x inflated). RE-CALIBRATED
# to 0.45 after WhoScored's real event stream (data/ingestors/whoscored.py)
# patched those fields in — DEF's real CBI data made them dramatically more
# competitive, which on its own pushed GK's modelled rate BELOW the real
# rate (7.5% vs 11.0% at the old 0.15) — 0.45 brings it back to ~11-12%.
# Never touches FPL save-points (only the bonus RANKING); not a change to
# the real BPSWeights rules. NOTE: DEF is now somewhat OVER-credited for
# bonus vs its own real rate (14.4% modelled vs 9.1% real) even after this
# fix — a separate, smaller residual not addressed here (see the P10 plan
# entry) — no compensating factor applied for that side yet.
GK_BONUS_SAVE_SCALE = 0.45
_GK_POSITIONS = frozenset({"GK", "GKP"})


def reduce_to_modelled(event: Mapping) -> dict:
    """Keep only the BPS inputs our components produce (zero the rest) — what
    the projection's sampled events actually populate."""
    return {k: v for k, v in event.items() if k in MODELLED_BPS_FIELDS}


def sample_fixture_bonus(events_by_player: Mapping[int, Mapping]) -> dict[int, int]:
    """Awarded bonus (3/2/1) for a fixture from per-player (sampled) events —
    the 26/27 BPS sim over reduced events, with the GK save-BPS calibration
    (``GK_BONUS_SAVE_SCALE``) applied. Called per scenario by P10."""
    adjusted = {
        pid: (
            {**ev, "saves": ev.get("saves", 0) * GK_BONUS_SAVE_SCALE}
            if ev.get("position") in _GK_POSITIONS else ev
        )
        for pid, ev in events_by_player.items()
    }
    return compute_fixture_bonus(adjusted)


def reduced_full_agreement(
    full_events_by_player: Mapping[int, Mapping],
) -> dict[str, float]:
    """Per fixture: compare bonus from FULL events vs the reduced (modelled-only)
    event set. Returns slot-exact rate + top-3-recipient Jaccard — the P8
    reduced-BPS bias vs the T5b full recompute."""
    full = compute_fixture_bonus(full_events_by_player)
    reduced = compute_fixture_bonus(
        {pid: reduce_to_modelled(e) for pid, e in full_events_by_player.items()}
    )
    slots = full.keys()
    exact = sum(1 for pid in slots if full[pid] == reduced[pid]) / len(slots) if slots else 1.0
    fset = {pid for pid, b in full.items() if b > 0}
    rset = {pid for pid, b in reduced.items() if b > 0}
    union = fset | rset
    jaccard = (len(fset & rset) / len(union)) if union else 1.0
    return {"slot_exact_rate": exact, "recipient_jaccard": jaccard}


def player_bps(event: Mapping, reduced: bool = True) -> int:
    """A player's 26/27 BPS from an event row — reduced (modelled fields only)
    by default, or full. Exposed for diagnostics/calibration."""
    return compute_player_bps(reduce_to_modelled(event) if reduced else event)


__all__ = [
    "MODELLED_BPS_FIELDS", "reduce_to_modelled", "sample_fixture_bonus",
    "reduced_full_agreement", "player_bps", "award_bonus",
]
