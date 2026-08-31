from pathlib import Path
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


BACKEND_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    DATABASE_URL: str
    DEEPSEEK_API_KEY: SecretStr | None = None
    DEEPSEEK_MODEL: str = "deepseek-v4-flash"
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
    DEEPSEEK_TIMEOUT_SECONDS: float = Field(default=60.0, gt=0)
    MAX_AGENT_STEPS: int = Field(default=8, gt=0)
    MAX_LLM_RETRIES: int = Field(default=2, ge=0)
    LLM_RETRY_BASE_SECONDS: float = Field(default=1.0, ge=0)
    LLM_RETRY_MAX_SECONDS: float = Field(default=8.0, ge=0)
    AGENT_LOOP_REPEAT_THRESHOLD: int = Field(default=3, ge=2)
    MAX_LLM_CONTEXT_CHARACTERS: int = Field(default=60_000, gt=0)
    MAX_CONTEXT_TOOL_RESULT_CHARACTERS: int = Field(default=12_000, gt=0)
    COMMAND_TIMEOUT_SECONDS: float = Field(default=30.0, gt=0)
    MAX_COMMAND_OUTPUT_BYTES_PER_STREAM: int = Field(default=65_536, ge=2)
    COMMAND_TERMINATION_GRACE_SECONDS: float = Field(default=2.0, gt=0)
    ALLOWED_WORKSPACE_ROOT: Path = BACKEND_DIR.parent
    WEB_CORS_ALLOWED_ORIGINS: tuple[str, ...] = ()

    @field_validator("WEB_CORS_ALLOWED_ORIGINS")
    @classmethod
    def validate_cors_origins(
        cls,
        origins: tuple[str, ...],
    ) -> tuple[str, ...]:
        normalized_origins: list[str] = []
        for origin in origins:
            normalized = origin.strip()
            parsed = urlsplit(normalized)
            if normalized == "*":
                raise ValueError("WEB_CORS_ALLOWED_ORIGINS must not contain '*'")
            if (
                parsed.scheme not in {"http", "https"}
                or parsed.hostname is None
                or parsed.username is not None
                or parsed.password is not None
                or parsed.path not in {"", "/"}
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError(
                    "WEB_CORS_ALLOWED_ORIGINS must contain HTTP origins only"
                )
            normalized_origins.append(normalized.rstrip("/"))
        if len(normalized_origins) != len(set(normalized_origins)):
            raise ValueError("WEB_CORS_ALLOWED_ORIGINS must be unique")
        return tuple(normalized_origins)

    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
    )


settings = Settings()
