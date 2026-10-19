"""
GraphGuard: Explainable GNN for Intrusion Detection
Utility: Configuration loader
"""
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Searches for config.yaml in:
    1. Explicit path provided
    2. Current working directory
    3. Project root (parent directories up to 3 levels)

    Args:
        config_path: Explicit path to config.yaml. If None, auto-discovered.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If no config.yaml can be found.
    """
    if config_path:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        with open(path, "r") as f:
            return yaml.safe_load(f)

    # Auto-discover
    search_dirs = [Path.cwd()]
    p = Path.cwd()
    for _ in range(4):
        p = p.parent
        search_dirs.append(p)

    for d in search_dirs:
        candidate = d / "config.yaml"
        if candidate.exists():
            with open(candidate, "r") as f:
                cfg = yaml.safe_load(f)
            # Resolve project root
            if "paths" in cfg and "project_root" in cfg["paths"]:
                cfg["_project_root"] = Path(cfg["paths"]["project_root"])
            else:
                cfg["_project_root"] = d
            return cfg

    raise FileNotFoundError(
        "Could not locate config.yaml. Run scripts from the project root or pass --config explicitly."
    )


def resolve_path(cfg: Dict[str, Any], *path_keys: str) -> Path:
    """
    Resolve a path from config relative to project root.

    Args:
        cfg: Loaded configuration dict.
        *path_keys: Keys to traverse in cfg["paths"], e.g. ("data", "processed").

    Returns:
        Absolute Path object.
    """
    root = cfg.get("_project_root", Path.cwd())
    node = cfg["paths"]
    for key in path_keys:
        node = node[key]
    return Path(root) / node


def get_project_root(cfg: Dict[str, Any]) -> Path:
    """Return the project root path from config."""
    return Path(cfg.get("_project_root", cfg["paths"]["project_root"]))
