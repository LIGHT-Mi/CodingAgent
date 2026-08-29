from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


BACKEND_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    DATABASE_URL: str
    DEEPSEEK_API_KEY: SecretStr | None = None
    DEEPSEEK_MODEL: str = "deepseek-v4-flash"
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
    DEEPSEEK_TIMEOUT_SECONDS: float = Field(default=60.0, gt=0)
    MAX_AGENT_STEPS: int = Field(default=8, gt=0)
    ALLOWED_WORKSPACE_ROOT: Path = BACKEND_DIR.parent

    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
    )


settings = Settings()
