"""
Application configuration loaded from environment variables.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central config — reads from env vars / .env file."""

    # ── App ──
    APP_ENV: str = "development"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"

    # ── Database ──
    DATABASE_URL: str = "sqlite+aiosqlite:///./carbonlens.db"

    # ── CORS ──
    CORS_ORIGINS: list[str] = ["http://localhost:3000"]

    # ── ML Model ──
    MODEL_ARTIFACT_DIR: str = "./artifacts"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
