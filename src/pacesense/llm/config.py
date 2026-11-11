"""Central YAML configuration for PaceSense's AI provider and per-task models.

Config precedence:
  1. explicit constructor / browser-session key (if supplied)
  2. BOSON_API_KEY environment variable
  3. config/local.yaml secret (ignored by git)

Public defaults live in config/pacesense.yaml. Local overrides live in config/local.yaml.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    """Find the project root in a source checkout, with cwd as a practical fallback."""
    override = os.environ.get("PACESENSE_CONFIG_DIR")
    if override:
        return Path(override).expanduser().resolve()
    package_root = Path(__file__).resolve().parents[3]
    if (package_root / "config" / "pacesense.yaml").exists():
        return package_root
    cwd = Path.cwd()
    if (cwd / "config" / "pacesense.yaml").exists():
        return cwd
    return package_root


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Could not read PaceSense configuration {path}: {exc}") from exc
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"PaceSense configuration must be a YAML mapping: {path}")
    return value


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def config_dir() -> Path:
    """Return the folder containing pacesense.yaml and optional local.yaml."""
    override = os.environ.get("PACESENSE_CONFIG_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return project_root() / "config"


def load_config() -> dict[str, Any]:
    folder = config_dir()
    return _merge(_read_yaml(folder / "pacesense.yaml"),
                  _read_yaml(folder / "local.yaml"))


def boson_api_key(env: dict[str, str] | None = None) -> tuple[str | None, str | None]:
    """Return (key, source); the source label never contains the secret."""
    source = os.environ if env is None else env
    env_key = source.get("BOSON_API_KEY", "").strip()
    if env_key:
        return env_key, "environment"
    key = load_config().get("boson", {}).get("api_key")
    if isinstance(key, str) and key.strip() and key.strip() not in {"YOUR_BOSON_API_KEY", "YOUR_KEY_HERE"}:
        return key.strip(), "config/local.yaml"
    return None, None


def openrouter_api_key(env: dict[str, str] | None = None) -> tuple[str | None, str | None]:
    """Return (key, source) for OpenRouter; the source label never contains the secret."""
    source = os.environ if env is None else env
    env_key = source.get("OPENROUTER_API_KEY", "").strip()
    if env_key:
        return env_key, "environment"
    key = load_config().get("openrouter", {}).get("api_key")
    if isinstance(key, str) and key.strip() and key.strip() not in {"YOUR_OPENROUTER_API_KEY", "YOUR_KEY_HERE"}:
        return key.strip(), "config/local.yaml"
    return None, None
