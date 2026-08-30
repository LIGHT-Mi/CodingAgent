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
    MAX_LLM_CONTEXT_CHARACTERS: int = Field(default=60_000, gt=0)
    MAX_CONTEXT_TOOL_RESULT_CHARACTERS: int = Field(default=12_000, gt=0)
    COMMAND_TIMEOUT_SECONDS: float = Field(default=30.0, gt=0)
    MAX_COMMAND_OUTPUT_BYTES_PER_STREAM: int = Field(default=65_536, ge=2)
    COMMAND_TERMINATION_GRACE_SECONDS: float = Field(default=2.0, gt=0)
    ALLOWED_WORKSPACE_ROOT: Path = BACKEND_DIR.parent

    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
    )


settings = Settings()
