"""Test-only pipeline helpers.

Lives inside ``experiments/`` so the security allow-list in
``explorer/lib/experiments.resolve_function`` (R13 Rec #3 — only
modules under ``experiments.*`` may be invoked from a recipe)
recognises these as legitimate. They are otherwise unused at runtime.
"""

from __future__ import annotations

import numpy as np


def _add_scalar(arr: np.ndarray, x: int = 0) -> np.ndarray:
    """Add a scalar to every element. Used by ``test_experiments`` to
    exercise parameter coercion + pipeline dispatch without depending
    on numpy / scipy heavy paths."""
    return arr.astype(np.float32, copy=False) + float(x)
