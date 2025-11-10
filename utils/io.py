from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: str | os.PathLike) -> Dict[str, Any]:
    """Load a YAML file and return a dictionary."""
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _guess_project_root(config_path: Path) -> Path:
    """
    Walk upwards from the config file location to find a directory that
    looks like the project root (contains canonical folders such as scripts/).
    Fallback to the config directory if nothing matches.
    """
    current = config_path.resolve().parent
    while current != current.parent:
        if (current / "scripts").is_dir() and (current / "utils").is_dir():
            return current
        current = current.parent
    return config_path.resolve().parent


def resolve_environment_paths(cfg: Dict[str, Any], config_path: str | os.PathLike) -> Dict[str, Any]:
    """
    Normalize environment paths so they are absolute, anchored at the config file directory.
    This keeps notebooks or scripts launched from subdirectories from writing into unintended locations.
    """
    base = _guess_project_root(Path(config_path))
    env = cfg.get("environment", {})
    for key in ("cache_dir", "h5_path", "giverny_output"):
        value = env.get(key)
        if not value:
            continue
        path_obj = Path(value)
        if not path_obj.is_absolute():
            env[key] = str((base / path_obj).resolve())
    return cfg


def ensure_dir(path: str | os.PathLike) -> Path:
    """Create a directory if it does not exist and return the Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_env_token(env_var: str, dotenv: str | os.PathLike = ".env") -> str | None:
    """Retrieve an API token from the environment or a .env file."""
    token = os.getenv(env_var)
    if token:
        return token

    dotenv_path = Path(dotenv)
    if not dotenv_path.exists():
        return None

    for line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() == env_var:
            return value.strip()
    return None


def experiment_cache_dir(cfg: Dict[str, Any]) -> Path:
    """Resolve the cache directory for an experiment."""
    cache_root = cfg["environment"]["cache_dir"]
    exp = cfg.get("experiment", "default")
    return ensure_dir(Path(cache_root) / exp)
