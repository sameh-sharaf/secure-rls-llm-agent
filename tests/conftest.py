"""Shared fixtures.

`cohort_floor` toggles the k-anonymity minimum. It patches the flag in every
module that reads it, not just where it is defined -- `sql_guard` and `gateway`
both do `from ...spec import ENFORCE_MIN_COHORT`, which binds the value at
import time, so patching `spec.ENFORCE_MIN_COHORT` alone changes nothing they
ever read.

That is the same mistake that made the ablation harness silently measure
nothing (see tests/test_ablation_harness.py). Once was a bug; twice would be a
habit.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

#: Every module that reads the flag, and therefore every module that must be
#: patched for a toggle to take effect.
_READERS = (
    "secure_rls.security.spec",
    "secure_rls.security.sql_guard",
    "secure_rls.security.gateway",
)


@contextmanager
def _set_floor(enabled: bool) -> Iterator[None]:
    import importlib

    modules = [importlib.import_module(name) for name in _READERS]
    saved = [getattr(m, "ENFORCE_MIN_COHORT") for m in modules]
    try:
        for module in modules:
            module.ENFORCE_MIN_COHORT = enabled
        yield
    finally:
        for module, previous in zip(modules, saved, strict=True):
            module.ENFORCE_MIN_COHORT = previous


@pytest.fixture
def cohort_floor():
    """Run a block with the k-anonymity floor forced on or off."""
    return _set_floor
