#!/usr/bin/env python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns
from sqlalchemy import text

from data.db import get_session

OUT_DIR = Path("results/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

POSITION_COLOURS = {"GKP": "#f4c542", "DEF": "#4a90d9", "MID": "#5cb85c", "FWD": "#e05c5c"}
sns.set_theme(style="darkgrid", palette="muted", font_scale=1.1)


def _load_projections(gw: int | None = None) -> pd.DataFrame:
    db = get_session()
    try:
        gw_filter = "AND pp.gameweek = :gw" if gw else ""
        params = {"gw": gw} if gw else {}
        query = text(f"""
            SELECT p.web_name, p.position, p.now_cost, p.team_id,
                   t.name AS team_name,
                   pp.gameweek, pp.xpts, pp.start_probability, pp.cs_probability
            FROM player_projections pp
            JOIN players p ON p.id = pp.player_id
            JOIN teams t ON t.id = p.team_id
            {gw_filter}
        """)
        return pd.read_sql(query, db.bind, params=params)
    finally:
        db.close()


def _load_gw_stats_season(season: str = "2026-27") -> pd.DataFrame:
    db = get_session()
    try:
        query = text("""
            SELECT s.player_id, s.gameweek, s.total_points, s.minutes,
                   p.web_name, p.position, p.team_id
            FROM player_gw_stats s
            JOIN players p ON p.id = s.player_id
            WHERE s.season = :season
            ORDER BY s.player_id, s.gameweek
        """)
        return pd.read_sql(query, db.bind, params={"season": season})
    finally:
        db.close()


def plot_value_map(df: pd.DataFrame) -> None:
    next_gw = int(df["gameweek"].min())
    gw_df = df[df["gameweek"] == next_gw].copy()
    gw_df = gw_df[gw_df["xpts"] > 0]

    fig, ax = plt.subplots(figsize=(11, 7))

    for pos, grp in gw_df.groupby("position"):
        ax.scatter(
            grp["now_cost"], grp["xpts"],
            label=pos, color=POSITION_COLOURS.get(pos, "grey"),
            alpha=0.7, s=55, linewidths=0.4, edgecolors="white",
        )

    threshold = gw_df["xpts"].quantile(0.88)
    for _, row in gw_df[gw_df["xpts"] >= threshold].iterrows():
        ax.annotate(
            row["web_name"],
            (row["now_cost"], row["xpts"]),
            fontsize=7.5, xytext=(4, 3), textcoords="offset points",
        )

    ax.set_xlabel("Price (£m)")
    ax.set_ylabel("Projected xPts (GW{})".format(next_gw))
    ax.set_title("FPL Value Map — Price vs Projected Points")
    ax.legend(title="Position")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "value_map.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {OUT_DIR}/value_map.png")


def plot_top_picks(df: pd.DataFrame) -> None:
    next_gw = int(df["gameweek"].min())
    top = (
        df[df["gameweek"] == next_gw]
        .sort_values("xpts", ascending=False)
        .drop_duplicates("web_name")
        .head(20)
    )

    fig, ax = plt.subplots(figsize=(10, 7))
    colours = [POSITION_COLOURS.get(p, "grey") for p in top["position"]]
    bars = ax.barh(top["web_name"][::-1], top["xpts"][::-1], color=colours[::-1], edgecolor="white")

    for bar, val in zip(bars, top["xpts"][::-1]):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}", va="center", fontsize=8.5)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in POSITION_COLOURS.values()]
    ax.legend(handles, POSITION_COLOURS.keys(), title="Position", loc="lower right")
    ax.set_xlabel("Projected xPts")
    ax.set_title(f"Top 20 Players — GW{next_gw} Projections")
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
    fig.tight_layout()
    fig.savefig(OUT_DIR / "top_picks.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {OUT_DIR}/top_picks.png")


def plot_cs_by_team(df: pd.DataFrame) -> None:
    next_gw = int(df["gameweek"].min())
    def_df = df[(df["gameweek"] == next_gw) & (df["position"].isin(["GKP", "DEF"]))]

    team_cs = (
        def_df.groupby("team_name")["cs_probability"]
        .mean()
        .sort_values(ascending=True)
    )
    team_cs = team_cs[team_cs > 0]

    fig, ax = plt.subplots(figsize=(9, 8))
    norm = plt.Normalize(team_cs.min(), team_cs.max())
    colours = plt.cm.RdYlGn(norm(team_cs.values))
    ax.barh(team_cs.index, team_cs.values, color=colours, edgecolor="white")

    for i, val in enumerate(team_cs.values):
        ax.text(val + 0.003, i, f"{val:.0%}", va="center", fontsize=8.5)

    ax.set_xlabel("Mean CS Probability")
    ax.set_title(f"Clean Sheet Probability by Team — GW{next_gw}")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    fig.tight_layout()
    fig.savefig(OUT_DIR / "cs_by_team.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {OUT_DIR}/cs_by_team.png")


def plot_start_prob_violin(df: pd.DataFrame) -> None:
    next_gw = int(df["gameweek"].min())
    gw_df = df[df["gameweek"] == next_gw].copy()
    gw_df = gw_df[gw_df["start_probability"] > 0]

    fig, ax = plt.subplots(figsize=(9, 6))
    pos_order = ["GKP", "DEF", "MID", "FWD"]
    palette = {p: POSITION_COLOURS[p] for p in pos_order}

    sns.violinplot(
        data=gw_df, x="position", y="start_probability",
        order=pos_order, hue="position", palette=palette,
        legend=False, inner="quartile", ax=ax,
    )
    ax.set_xlabel("Position")
    ax.set_ylabel("P(Start / 60+ mins)")
    ax.set_title("Start Probability Distribution by Position")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    fig.tight_layout()
    fig.savefig(OUT_DIR / "start_prob_violin.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {OUT_DIR}/start_prob_violin.png")


def plot_points_heatmap(stats: pd.DataFrame) -> None:
    top_players = (
        stats.groupby("web_name")["total_points"]
        .sum()
        .sort_values(ascending=False)
        .head(25)
        .index.tolist()
    )

    pivot = (
        stats[stats["web_name"].isin(top_players)]
        .pivot_table(index="web_name", columns="gameweek", values="total_points", aggfunc="sum")
        .fillna(0)
    )
    pivot = pivot.loc[top_players]

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(
        pivot, ax=ax,
        cmap="YlOrRd", linewidths=0.4, linecolor="#222",
        annot=True, fmt=".0f", annot_kws={"size": 7},
        cbar_kws={"label": "Points"},
    )
    ax.set_title("2026-27 Points Heatmap — Top 25 Players")
    ax.set_xlabel("Gameweek")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "points_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {OUT_DIR}/points_heatmap.png")


if __name__ == "__main__":
    print("Loading data...")
    projections = _load_projections()
    stats = _load_gw_stats_season("2026-27")

    print("Generating plots...")
    plot_value_map(projections)
    plot_top_picks(projections)
    plot_cs_by_team(projections)
    plot_start_prob_violin(projections)

    if not stats.empty:
        plot_points_heatmap(stats)
    else:
        print("  Skipped heatmap: no 2026-27 GW stats yet")

    print(f"\nAll plots saved to {OUT_DIR}/")
