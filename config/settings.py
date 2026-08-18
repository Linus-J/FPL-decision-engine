from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Anchored to the repository root, not the process's working directory
# (2026-08-18). A bare ".env" is resolved against the cwd, so the whole
# configuration silently disappears whenever the code is run from anywhere
# else -- and every field here has a default, so nothing raises: the odds
# ingest just logs "THE_ODDS_API_KEY not set" and skips, leaving fixtures on
# a flat league-average scoreline instead of real prices. Observed exactly
# that three days before the 2026-27 GW1 deadline, running from a worktree.
#
# This is the same failure as the DB_PATH incident of 2026-07-31, where a
# relative path meant a scheduled run spent five weeks reading a stale
# database. A path that depends on where you happened to start the process is
# not a configuration, it is a coin flip.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_ENV_FILE = _REPO_ROOT / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=_ENV_FILE, env_file_encoding="utf-8")

    fpl_email: str = Field(default="", alias="FPL_EMAIL")
    fpl_password: str = Field(default="", alias="FPL_PASSWORD")
    fpl_team_id: int = Field(default=0, alias="FPL_TEAM_ID")

    the_odds_api_key: str = Field(default="", alias="THE_ODDS_API_KEY")
    sportmonks_api_key: str = Field(default="", alias="SPORTMONKS_API_KEY")
    guardian_api_key: str = Field(default="test", alias="GUARDIAN_KEY")

    telegram_bot_token: str = Field(default="", alias="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: str = Field(default="", alias="TELEGRAM_CHAT_ID")

    dry_run: bool = Field(default=True, alias="DRY_RUN")
    db_path: str = Field(default="fpl_bot.db", alias="DB_PATH")


settings = Settings()
