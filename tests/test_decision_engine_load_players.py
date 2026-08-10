"""_load_players (agent/decision_engine.py) is one of the two live
candidate-pool loaders (the other is projection/cold_start.py::
load_current_players) that must apply the manual team_id override (Feature
B, plan 2026-08-10) -- this is the single shared seam serving both
cold-start and in-season transfer decisions."""

from __future__ import annotations

import yaml
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from agent import decision_engine as de
from data import overrides as ov
from data.models import Base, Player


def test_load_players_applies_team_overrides(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'de.db'}")
    Base.metadata.create_all(bind=engine)
    Local = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    monkeypatch.setattr(de, "get_session", lambda: Local())

    s = Local()
    try:
        s.add(Player(fpl_id=1, code=111, first_name="A", second_name="A", web_name="Moved",
                     team_id=1, position="MID", now_cost=8.0, status="a"))
        s.commit()
    finally:
        s.close()

    overrides_path = tmp_path / "transfer_overrides.yaml"
    overrides_path.write_text(yaml.safe_dump({"confirmed": [{"code": 111, "team_id": 99}]}))
    monkeypatch.setattr(ov, "OVERRIDES_PATH", overrides_path)

    players = de._load_players()
    assert players.loc[players["web_name"] == "Moved", "team_id"].iloc[0] == 99
    assert "code" in players.columns
