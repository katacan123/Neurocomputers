# utils/io_utils.py

from pathlib import Path
import yaml
from typing import Any, Dict, Union


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a YAML config file and return it as a dict.
    """
    path = Path(path)
    with path.open("r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Create directory if it doesn't exist and return Path.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
