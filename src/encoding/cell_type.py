"""
Detects whether a cell's raw text should be treated as numeric or text,
so the cell encoder can route it to the right embedder.

Kept deliberately simple for a first pass: strip common numeric
punctuation (commas, currency symbols, percent signs) and try a float
parse. Extend here later if you need date detection, unit handling, etc.
"""

import re
from enum import Enum


class CellType(Enum):
    NUMERIC = "numeric"
    TEXT = "text"


_NUMERIC_STRIP_RE = re.compile(r"[,\$%\s]")


def try_parse_numeric(cell: str) -> float | None:
    """Returns the parsed float if `cell` looks numeric, else None."""

    if cell is None:
        return None

    stripped = _NUMERIC_STRIP_RE.sub("", cell)

    if stripped in ("", "-", "."):
        return None

    try:
        return float(stripped)
    except ValueError:
        return None


def detect_cell_type(cell: str) -> CellType:
    return (
        CellType.NUMERIC
        if try_parse_numeric(cell) is not None
        else CellType.TEXT
    )