import os
from pathlib import Path
from types import SimpleNamespace

import yaml


def _ns(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_ns(v) for v in obj]
    if isinstance(obj, str):
        return os.path.expandvars(obj)  # lets configs reference $DATA_ROOT etc.
    return obj


def load_config(path):
    with open(Path(path)) as f:
        return _ns(yaml.safe_load(f))
