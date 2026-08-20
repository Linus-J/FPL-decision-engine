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
    # extra="ignore" is load-bearing, not tidiness (2026-08-20). The comment
    # below on DRY_RUN/FPL_PASSWORD asserts "extra keys in .env are ignored,
    # so an old file keeps working" -- but pydantic-settings defaults to
    # extra="forbid", so that was never true. A .env still carrying the two
    # keys removed with the submission path raised at IMPORT time, and
    # `settings = Settings()` runs at module scope, so every entry point in
    # the project died before main() -- found one day before the 2026-27 GW1
    # deadline. Removing a field must not be able to brick the bot on
    # machines whose .env still has it.
    model_config = SettingsConfigDict(
        env_file=_ENV_FILE, env_file_encoding="utf-8", extra="ignore"
    )

    fpl_email: str = Field(default="", alias="FPL_EMAIL")
    fpl_team_id: int = Field(default=0, alias="FPL_TEAM_ID")

    the_odds_api_key: str = Field(default="", alias="THE_ODDS_API_KEY")
    sportmonks_api_key: str = Field(default="", alias="SPORTMONKS_API_KEY")
    guardian_api_key: str = Field(default="test", alias="GUARDIAN_KEY")

    telegram_bot_token: str = Field(default="", alias="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: str = Field(default="", alias="TELEGRAM_CHAT_ID")

    # DRY_RUN and FPL_PASSWORD were removed on 2026-08-18 along with the
    # submission path. There is no live mode to switch on and no login to
    # perform, so a stale value in .env can no longer mean anything. Extra
    # keys in .env are ignored, so an old file keeps working.
    db_path: str = Field(default="fpl_bot.db", alias="DB_PATH")


settings = Settings()
