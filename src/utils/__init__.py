"""Shared utilities for the TamilTech-QA project."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict

import yaml

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover - python-dotenv is optional at runtime
    pass

_ENV_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")


def _expand_env(node: Any) -> Any:
    """Recursively expand ``${VAR}`` references in a YAML structure.

    Args:
        node: A loaded YAML node (dict, list, str, or scalar).

    Returns:
        The same structure with environment variables substituted in-place.
        Missing variables expand to an empty string and a warning is printed.

    Example:
        >>> os.environ["FOO"] = "bar"
        >>> _expand_env({"key": "${FOO}"})
        {'key': 'bar'}
    """
    if isinstance(node, dict):
        return {k: _expand_env(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_expand_env(v) for v in node]
    if isinstance(node, str):

        def sub(match: re.Match[str]) -> str:
            var = match.group(1)
            return os.environ.get(var, "")

        return _ENV_PATTERN.sub(sub, node)
    return node


def load_config(path: str | os.PathLike[str]) -> Dict[str, Any]:
    """Load a YAML config file and expand ``${ENV_VAR}`` references.

    Args:
        path: Path to a YAML file.

    Returns:
        Parsed config as a nested dictionary.

    Example:
        >>> cfg = load_config("config/data_config.yaml")
        >>> isinstance(cfg, dict)
        True
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return _expand_env(raw or {})


def project_root() -> Path:
    """Return the repository root directory.

    Returns:
        Absolute :class:`pathlib.Path` pointing to the repo root.
    """
    return Path(__file__).resolve().parents[2]


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    """Create a directory (including parents) if it does not exist.

    Args:
        path: Directory path.

    Returns:
        The path as a :class:`pathlib.Path`.

    Example:
        >>> ensure_dir("data/raw").exists()
        True
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
