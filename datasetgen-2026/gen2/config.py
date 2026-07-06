"""Configuration loading for gen2."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class Cfg:
    """Attribute-access wrapper over the YAML config tree."""

    def __init__(self, d: dict):
        self._d = d

    def __getattr__(self, k: str) -> Any:
        try:
            v = self._d[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return Cfg(v) if isinstance(v, dict) else v

    def __getitem__(self, k):
        v = self._d[k]
        return Cfg(v) if isinstance(v, dict) else v

    def get(self, k, default=None):
        v = self._d.get(k, default)
        return Cfg(v) if isinstance(v, dict) else v

    def raw(self) -> dict:
        return self._d


def load_config(path: str | Path) -> Cfg:
    with open(path) as f:
        return Cfg(yaml.safe_load(f))
