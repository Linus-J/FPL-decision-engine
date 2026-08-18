#!/usr/bin/env python
"""explain_squad.py — why the engine picked THIS squad (2026-08-18).

Answers the question a summary xPts number cannot: how confident is this
decision, and how close was it? Reports four things the optimiser knows but
never surfaces.

**Margin.** For each pick, the squad is re-solved with that player banned.
The drop in total xPts is what he is actually worth over the best legal
alternative — not his own projection, which is the number people wrongly
read as his value. A pick that survives by 0.2 points is a coin flip dressed
up as a decision; one that survives by 5 is a real edge. Same idea as a
shadow price, computed by re-solving because an ILP has no dual to read.

**Provenance.** What share of the decision rests on real bookmaker odds
versus a strength model versus a prior-season carry-over. Bookmakers only
price the near gameweeks, so a five-gameweek horizon is mostly model, and
the fixture-difficulty argument for holding a player through a hard run is
usually the least evidenced part of the whole projection.

**Correlated exposure.** The objective sums per-player variance and models
no covariance between teammates (see optimiser/scoring.py), so it cannot see
that a goalkeeper and centre-back from the same club share one clean sheet
and blank together. Clubs at the selection cap are corner solutions: the
optimiser took every player the rules allowed, which is exactly where an
unpriced correlation does the most damage.

**Bench.** What the bench cost and what it would contribute, so the standard
"bench fodder" trade is a visible choice rather than an accident.

    uv run python scripts/explain_squad.py --season 2026-27
    uv run python scripts/explain_squad.py --out /tmp/why.md
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

import pandas as pd  # noqa: E402
from sqlalchemy import text  # noqa: E402

from config.strategy import OPTIMISER, SQUAD  # noqa: E402
from data.db import get_session  # noqa: E402
from data.ingestors.odds_api import odds_coverage_by_gameweek  # noqa: E402
from optimiser.squad import optimise_squad  # noqa: E402
from projection import cold_start  # noqa: E402

POSITION_ORDER = {"GKP": 0, "DEF": 1, "MID": 2, "FWD": 3}


def _team_names() -> dict[int, str]:
    db = get_session()
    try:
        return {r[0]: r[1] for r in db.execute(text("SELECT id, name FROM teams")).fetchall()}
    finally:
        db.close()


def _squad_frame(solution, projections: pd.DataFrame, teams: dict[int, str]) -> pd.DataFrame:
    per_gw = projections.pivot_table(index="player_id", columns="gameweek", values="xpts")
    source = projections.drop_duplicates("player_id").set_index("player_id")["proj_source"]

    df = solution.squad.copy().join(per_gw, on="id")
    df["team"] = df["team_id"].map(teams)
    df["source"] = df["id"].map(source)
    starting = set(solution.starting_xi["id"])
    df["role"] = df["id"].map(lambda i: "XI" if i in starting else "bench")
    return df


def _margins(
    solution, projections: pd.DataFrame, players: pd.DataFrame, season: str, horizon: int
) -> dict[int, tuple[float, str]]:
    """Per pick: xPts lost if he were unavailable, and who takes his place.

    Re-solves the whole squad each time rather than swapping in the next-best
    player at the same position, because the budget freed by dropping a
    premium is usually spent somewhere else entirely — the honest replacement
    is a different squad, not a different player.
    """
    baseline = solution.total_xpts
    out: dict[int, tuple[float, str]] = {}
    owned = set(solution.squad["id"])
    for pid in solution.squad["id"]:
        try:
            alt = optimise_squad(
                projections=projections, players=players, budget=SQUAD.budget_total,
                horizon=horizon, season=season, force_exclude_ids=[int(pid)],
            )
        except Exception as exc:  # a banned player can make the squad infeasible
            out[int(pid)] = (float("nan"), f"no legal squad without him ({type(exc).__name__})")
            continue
        entered = set(alt.squad["id"]) - owned
        names = ", ".join(sorted(alt.squad.loc[alt.squad["id"].isin(entered), "web_name"]))
        out[int(pid)] = (baseline - alt.total_xpts, names or "(reshuffle only)")
    return out


def _provenance(
    projections: pd.DataFrame, squad_ids: set[int], season: str, horizon: int
) -> list[str]:
    lines: list[str] = []
    sq = projections[projections["player_id"].isin(squad_ids)]
    total = float(sq["xpts"].sum())

    coverage = odds_coverage_by_gameweek(season, horizon)
    priced_gws = {gw for gw, (have, _) in coverage.items() if have > 0}
    odds_backed = float(sq[sq["gameweek"].isin(priced_gws)]["xpts"].sum())

    lines.append("| gameweek | fixtures priced by bookmakers |")
    lines.append("| --- | --- |")
    for gw, (have, want) in sorted(coverage.items()):
        flag = "" if have == want else "  <-- strength model"
        lines.append(f"| GW{gw} | {have}/{want}{flag} |")
    lines.append("")
    if total > 0:
        lines.append(
            f"**{100 * odds_backed / total:.0f}% of this squad's projected points comes from "
            f"gameweeks with real odds; {100 * (total - odds_backed) / total:.0f}% comes from the "
            "strength model.** Bookmakers do not price beyond the next round or two, so any "
            "argument that rests on fixtures three or more gameweeks out is resting on the "
            "modelled half."
        )
    lines.append("")
    lines.append("Per-player projection source:")
    lines.append("")
    counts = projections.drop_duplicates("player_id")["proj_source"].value_counts()
    for src, n in counts.items():
        lines.append(f"- `{src}`: {n} players")
    return lines


def _exposure(df: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    by_club = df.groupby("team").size().sort_values(ascending=False)
    capped = [c for c, n in by_club.items() if n >= SQUAD.max_players_per_club]

    lines.append("| club | players |")
    lines.append("| --- | --- |")
    for club, n in by_club.items():
        flag = "  **at the cap**" if n >= SQUAD.max_players_per_club else ""
        lines.append(f"| {club} | {n}{flag} |")
    lines.append("")
    if capped:
        lines.append(
            f"{', '.join(capped)} sit at the {SQUAD.max_players_per_club}-player limit. That is a "
            "corner solution: the optimiser took every player the rules allowed and would have "
            "taken more. Nothing in the objective prices the fact that they rise and fall "
            "together — per-player variances are summed as if independent."
        )
        lines.append("")

    # The tightest correlation in FPL: one clean sheet, paid to two players.
    pairs: list[str] = []
    for club, group in df.groupby("team"):
        keepers = group[group["position"] == "GKP"]["web_name"].tolist()
        defenders = group[group["position"] == "DEF"]["web_name"].tolist()
        if keepers and defenders:
            pairs.append(f"- **{club}**: {', '.join(keepers)} + {', '.join(defenders)}")
    if pairs:
        lines.append(
            "Clean-sheet exposure doubled up — these share a single event, so they blank "
            "together and haul together:"
        )
        lines.append("")
        lines.extend(pairs)
    return lines


def _pool_section(
    pool: list, teams: dict[int, str], players: pd.DataFrame, optimal_ids: set[int]
) -> list[str]:
    """How much of the squad survives when the optimiser is made to choose again.

    The margin column answers "what does banning this one player cost?". This
    answers the broader question: of the ten best legal squads, how many
    contain him? A player in all ten is a conviction; one in three of ten is
    the model shrugging, and no single-solve statistic shows that.
    """
    lines: list[str] = []
    if len(pool) < 2:
        lines.append("Only one legal squad was found, so there is nothing to compare against.")
        return lines

    best = pool[0].total_xpts
    lines.append("Ranked by the OBJECTIVE — which is decayed, risk-adjusted and bench-weighted —")
    lines.append("while the xPts column is the true undiscounted total. The two can disagree, and")
    lines.append("where they do is exactly where those adjustments are doing the work: a squad")
    lines.append("with more raw expected points ranked below one with fewer means the objective")
    lines.append("preferred points sooner, on a safer distribution, or with a usable bench.")
    lines.append("")
    lines.append("| rank | true xPts | vs rank 1 | £m | players changed |")
    lines.append("| --- | --- | --- | --- | --- |")
    best_ids = set(pool[0].squad["id"])
    for rank, solution in enumerate(pool, start=1):
        ids = set(solution.squad["id"])
        lines.append(
            f"| {rank} | {solution.total_xpts:.2f} | {solution.total_xpts - best:+.2f} | "
            f"{solution.total_cost:.1f} | {len(ids - best_ids)} |"
        )
    lines.append("")

    within_one = sum(1 for s in pool if abs(s.total_xpts - best) <= 1.0)
    lines.append(
        f"**{within_one} of the {len(pool)} best squads sit within 1.0 xPts of the top one.** "
        "Read that alongside the appearance table below rather than on its own: a flat pool "
        "does not mean the whole squad is arbitrary, it usually means a settled core with a "
        "few interchangeable places at the bottom."
    )
    lines.append("")

    counts: dict[int, int] = {}
    for solution in pool:
        for pid in solution.squad["id"]:
            counts[int(pid)] = counts.get(int(pid), 0) + 1
    info = players.set_index("id")
    rows = sorted(counts.items(), key=lambda kv: (-kv[1], -float(info.loc[kv[0], "now_cost"])))

    lines.append(f"Appearances across the {len(pool)}-squad pool:")
    lines.append("")
    lines.append("| player | club | £m | in N squads | in the chosen squad |")
    lines.append("| --- | --- | --- | --- | --- |")
    for pid, count in rows:
        row = info.loc[pid]
        lines.append(
            f"| {row['web_name']} | {teams.get(int(row['team_id']), '?')} | "
            f"{row['now_cost']:.1f} | {count}/{len(pool)} | "
            f"{'yes' if pid in optimal_ids else 'no'} |"
        )
    lines.append("")
    unanimous = [pid for pid, c in counts.items() if c == len(pool)]
    contested = len(counts) - len(unanimous)
    lines.append(
        f"**{len(unanimous)} players appear in every squad in the pool; {contested} others are "
        f"contested.** The unanimous ones are the actual decision — the model wants them however "
        "it is made to choose again. The contested ones are where a hunch of your own costs "
        "nearly nothing to act on."
    )
    return lines


def build_report(season: str, pool_size: int = 0) -> str:
    teams = _team_names()
    solution, projections = cold_start.build_initial_squad(season)
    players = cold_start.apply_departure_gate(cold_start.load_current_players())
    players = players.merge(
        projections[["player_id", "start_probability"]].drop_duplicates("player_id"),
        left_on="id", right_on="player_id", how="left",
    ).drop(columns=["player_id"], errors="ignore")

    horizon = OPTIMISER.cold_start_lookahead_gws
    df = _squad_frame(solution, projections, teams)
    gw_cols = sorted(projections["gameweek"].unique())[:horizon]
    df["total"] = df[gw_cols].sum(axis=1)

    margins = _margins(solution, projections, players, season, horizon)
    df["margin"] = df["id"].map(lambda i: margins.get(int(i), (float("nan"), ""))[0])
    df["replacement"] = df["id"].map(lambda i: margins.get(int(i), (float("nan"), ""))[1])

    df = df.sort_values(
        ["role", "position", "total"],
        key=lambda s: s.map(POSITION_ORDER) if s.name == "position" else s,
        ascending=[True, True, False],
    )

    out: list[str] = []
    out.append(f"# Why this squad — {season} GW{min(gw_cols)}")
    out.append("")
    out.append(
        f"Total {solution.total_xpts:.1f} xPts over {len(gw_cols)} gameweeks, "
        f"£{solution.total_cost:.1f}m spent, captain "
        f"{df.loc[df['id'] == solution.captain_id, 'web_name'].iloc[0]}."
    )
    out.append("")

    out.append("## The picks")
    out.append("")
    header = ["player", "club", "pos", "£m", "role"] + [f"GW{g}" for g in gw_cols] + [
        "total", "margin", "if unavailable"
    ]
    out.append("| " + " | ".join(header) + " |")
    out.append("| " + " | ".join("---" for _ in header) + " |")
    for _, r in df.iterrows():
        cells = [
            str(r["web_name"]), str(r["team"]), str(r["position"]),
            f"{r['now_cost']:.1f}", str(r["role"]),
        ]
        cells += [f"{r[g]:.2f}" if pd.notna(r[g]) else "—" for g in gw_cols]
        cells += [
            f"{r['total']:.2f}",
            f"{r['margin']:.2f}" if pd.notna(r["margin"]) else "—",
            str(r["replacement"]),
        ]
        out.append("| " + " | ".join(cells) + " |")
    out.append("")
    out.append(
        "`margin` is what the squad loses if that player is banned and it re-solves — his value "
        "over the best alternative, not his own projection. Small margins are where to look "
        "first: they are picks the model is not really choosing between."
    )
    out.append("")

    thin = df[df["margin"] < 0.5].sort_values("margin")
    if not thin.empty:
        out.append("**Effectively arbitrary** (worth under 0.5 xPts more than their replacement): "
                   + ", ".join(f"{r['web_name']} ({r['margin']:.2f})" for _, r in thin.iterrows()))
        out.append("")

    out.append("## How much of this is measured, and how much is modelled")
    out.append("")
    out.extend(_provenance(projections, set(df["id"]), season, horizon))
    out.append("")

    out.append("## Correlated exposure")
    out.append("")
    out.extend(_exposure(df))
    out.append("")

    if pool_size > 1:
        from optimiser.squad import generate_squad_pool

        pool = generate_squad_pool(
            projections, players, n=pool_size, budget=SQUAD.budget_total,
            horizon=horizon, season=season,
        )
        out.append("## How much of this squad is actually a choice")
        out.append("")
        out.extend(_pool_section(pool, teams, players, set(df["id"])))
        out.append("")

    bench = df[df["role"] == "bench"]
    out.append("## The bench")
    out.append("")
    out.append(
        f"£{bench['now_cost'].sum():.1f}m of £{df['now_cost'].sum():.1f}m "
        f"({100 * bench['now_cost'].sum() / df['now_cost'].sum():.0f}% of the budget) for "
        f"{bench['total'].sum():.1f} xPts over the horizon — "
        f"{bench['total'].sum() / max(len(gw_cols), 1):.1f} a gameweek if they all played, which "
        "they are not meant to."
    )
    out.append("")
    slots = ", ".join(
        f"slot {k + 1} {w * OPTIMISER.bench_value_weight:.0%}"
        for k, w in enumerate(OPTIMISER.bench_slot_weights)
    )
    out.append(
        f"Bench weights in the objective: {slots}, reserve keeper "
        f"{OPTIMISER.bench_gk_weight * OPTIMISER.bench_value_weight:.0%}. These are the "
        "probabilities that each bench slot is actually reached by an automatic substitution, so "
        "the first substitute is bought as a real player and the last two as fodder — which is "
        "what they are."
    )
    out.append("")
    out.append(
        "The weights are static, so they do not tighten as a squad becomes more nailed-on, and "
        "they price no injury or rotation risk beyond each player's start probability. The cost "
        "of a thin bench still only appears when somebody in the XI does not play, and nothing "
        "here forecasts a specific absence."
    )
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", default="2026-27")
    parser.add_argument("--out", default=None, help="write Markdown here instead of stdout")
    parser.add_argument(
        "--pool", type=int, default=0, metavar="N",
        help="also report the N best distinct squads and how often each player appears "
             "across them (each one costs a further solve)",
    )
    args = parser.parse_args()

    report = build_report(args.season, pool_size=args.pool)
    if args.out:
        Path(args.out).write_text(report)
        print(f"wrote {args.out}")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
