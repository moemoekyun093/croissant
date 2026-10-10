"""Small I/O helpers."""
import json
import os
from pathlib import Path
from typing import Any, Optional


def atomic_json(path: Path, value: Any, indent: Optional[int] = None) -> None:
    """Write JSON to ``path`` atomically (write ``.partial``, then rename).

    A crash mid-write can never leave a truncated file at ``path``, which
    keeps resumable checkpoints/manifests valid.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("w", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, indent=indent)
    os.replace(partial, path)


def load_json(path: Path) -> Any:
    with Path(path).open(encoding="utf-8") as source:
        return json.load(source)
