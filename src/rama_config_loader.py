"""Lightweight configuration loader for the RaMA reproduction.

Resolution order (highest priority first):
    1. Environment variable (the name in brackets in the example config).
    2. ``configs/rama_config.yaml`` next to the package, if present.
    3. The hardcoded default passed by the caller.

This keeps private paths and API keys out of the source tree: reference
scripts call :func:`get` with a dotted key and an environment-variable name,
and a user supplies values through ``configs/rama_config.yaml`` or the
environment.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - yaml is an optional convenience
    yaml = None


def _find_config_file() -> Path | None:
    explicit = os.environ.get("RAMA_CONFIG")
    if explicit and Path(explicit).is_file():
        return Path(explicit)
    # This loader lives at <root>/src/, so the package
    # root (which holds configs/) is four levels up.
    root = Path(__file__).resolve().parents[1]
    candidate = root / "configs" / "rama_config.yaml"
    return candidate if candidate.is_file() else None


@lru_cache(maxsize=1)
def _load_file() -> dict[str, Any]:
    path = _find_config_file()
    if path is None or yaml is None:
        return {}
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return data if isinstance(data, dict) else {}


def _dig(data: dict[str, Any], dotted_key: str) -> Any:
    node: Any = data
    for part in dotted_key.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def get(dotted_key: str, env: str | None = None, default: Any = None) -> Any:
    """Resolve a config value by dotted key with optional env-var override.

    Args:
        dotted_key: e.g. ``"data.dataset_root"``.
        env: environment variable name that overrides the file value.
        default: returned when neither env nor file provides a value.
    """
    if env:
        env_val = os.environ.get(env)
        if env_val not in (None, ""):
            return env_val
    file_val = _dig(_load_file(), dotted_key)
    if file_val not in (None, ""):
        return file_val
    return default
