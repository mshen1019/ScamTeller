"""Minimal ``.env`` reader.

Parsed by hand rather than with python-dotenv: the only runtime dependency is httpx, and reading ten lines of KEY=VALUE
isn't worth another package.

Two deliberate behaviours:

* **Existing environment variables are not overwritten**. ``export`` always beats the file; otherwise "but I did export it"
  becomes an untraceable problem while debugging.
* **Only variable names are returned, never values**. Callers can print which keys were loaded without writing the keys themselves into logs.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Maximum number of levels to walk up looking for ``.env``. Enough for a ~/workspace/<project> layout.
MAX_PARENTS = 4


def parse_env_file(path: str | Path) -> dict[str, str]:
    """Parse ``KEY=value``. Supports an ``export`` prefix, ``#`` comments, single and double quotes."""
    out: dict[str, str] = {}
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        out[key] = value
    return out


def find_env_file(start: str | Path = ".") -> Path | None:
    """Find the first ``.env`` walking up from ``start``."""
    here = Path(start).resolve()
    for parent in [here, *here.parents][:MAX_PARENTS + 1]:
        candidate = parent / ".env"
        if candidate.is_file():
            return candidate
    return None


def load_env_file(path: str | Path | None = None, *, start: str | Path = ".") -> tuple[Path | None, list[str]]:
    """Load ``.env`` and return ``(file used, names of variables newly set)``.

    If ``path`` is given the file must exist; otherwise search upwards and do nothing if none is found.
    """
    if path is not None:
        target = Path(path).expanduser()
        if not target.is_file():
            raise FileNotFoundError(f"env file not found: {target}")
    else:
        target = find_env_file(start)
        if target is None:
            return None, []

    loaded: list[str] = []
    for key, value in parse_env_file(target).items():
        if os.environ.get(key, "").strip():
            continue                      # export wins
        if not value:
            continue
        os.environ[key] = value
        loaded.append(key)
    return target, loaded
