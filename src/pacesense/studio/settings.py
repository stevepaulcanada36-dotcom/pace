"""Runtime settings for PaceSense Studio, read once from the environment (12-factor style).

Every setting has a safe default, so `pacesense studio` works with no configuration. Invalid values fail fast
at start-up with a readable message instead of surfacing later as a 500.

    PACESENSE_DB                 SQLite file for saved plans and coach runners   (default: pacesense.sqlite3)
    PACESENSE_DATA               race CSV (e.g. from `pacesense import`)         (default: synthetic demo data)
    PACESENSE_SEED               seed for the synthetic data                     (default: config reproducibility.seed)
    PACESENSE_HOST / _PORT       bind address                                    (default: 127.0.0.1 / 8000)
    PACESENSE_LOG_LEVEL          DEBUG | INFO | WARNING | ERROR                  (default: INFO)
    PACESENSE_LOG_FORMAT         json | text                                     (default: text)
    PACESENSE_SESSION_TTL        idle seconds before a session is dropped        (default: 3600)
    PACESENSE_MAX_SESSIONS       sessions kept in memory at once                 (default: 200)
    PACESENSE_CORS_ORIGINS       comma-separated origins allowed to call the API (default: none)
    PACESENSE_AI                 1 to let the optional AI assistant answer questions (needs an OpenRouter key)
"""
from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator


class Settings(BaseModel):
    db_path: str = "pacesense.sqlite3"
    data_path: str | None = None
    seed: int | None = None
    host: str = "127.0.0.1"
    port: int = Field(8000, ge=1, le=65535)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["json", "text"] = "text"
    session_ttl_s: int = Field(3600, ge=60, le=7 * 24 * 3600)
    max_sessions: int = Field(200, ge=1, le=10_000)
    cors_origins: list[str] = Field(default_factory=list)
    ai_enabled: bool = False
    access_user: str | None = None
    access_password: str | None = None

    model_config = {"frozen": True}

    @field_validator("data_path")
    @classmethod
    def _data_exists(cls, v: str | None) -> str | None:
        if v and not os.path.isfile(v):
            raise ValueError(f"race data file not found: {v}")
        return v or None

    @field_validator("log_level", mode="before")
    @classmethod
    def _upper(cls, v):
        return str(v).upper()

    @model_validator(mode="after")
    def _access_pair(self) -> "Settings":
        if bool(self.access_user) != bool(self.access_password):
            raise ValueError("PACESENSE_ACCESS_USER and PACESENSE_ACCESS_PASSWORD must be set together")
        return self

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None, **overrides) -> Settings:
        e = os.environ if env is None else env
        raw = {
            "db_path": e.get("PACESENSE_DB"),
            "data_path": e.get("PACESENSE_DATA"),
            "seed": e.get("PACESENSE_SEED"),
            "host": e.get("PACESENSE_HOST"),
            "port": e.get("PACESENSE_PORT"),
            "log_level": e.get("PACESENSE_LOG_LEVEL"),
            "log_format": e.get("PACESENSE_LOG_FORMAT"),
            "session_ttl_s": e.get("PACESENSE_SESSION_TTL"),
            "max_sessions": e.get("PACESENSE_MAX_SESSIONS"),
            "cors_origins": [o.strip() for o in e.get("PACESENSE_CORS_ORIGINS", "").split(",") if o.strip()] or None,
            "ai_enabled": e.get("PACESENSE_AI", "").lower() in ("1", "true", "yes", "on") or None,
            "access_user": e.get("PACESENSE_ACCESS_USER"),
            "access_password": e.get("PACESENSE_ACCESS_PASSWORD"),
        }
        raw.update({k: v for k, v in overrides.items()})
        raw = {k: v for k, v in raw.items() if v is not None and v != ""}
        try:
            return cls(**raw)
        except ValidationError as exc:
            problems = "; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in exc.errors())
            raise SystemExit(f"PaceSense Studio: invalid settings ({problems})") from None


__all__ = ["Settings"]
