from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    fpl_email: str = Field(default="", alias="FPL_EMAIL")
    fpl_password: str = Field(default="", alias="FPL_PASSWORD")
    fpl_team_id: int = Field(default=0, alias="FPL_TEAM_ID")

    the_odds_api_key: str = Field(default="", alias="THE_ODDS_API_KEY")

    telegram_bot_token: str = Field(default="", alias="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: str = Field(default="", alias="TELEGRAM_CHAT_ID")

    dry_run: bool = Field(default=True, alias="DRY_RUN")
    db_path: str = Field(default="fpl_bot.db", alias="DB_PATH")


settings = Settings()
