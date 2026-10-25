"""Tiny helpers so the tests run both under pytest and under run_tests.py."""
from __future__ import annotations

import contextlib
from typing import Iterator


@contextlib.contextmanager
def raises(exc: type[BaseException]) -> Iterator[None]:
    """Assert that the block raises ``exc`` (pytest.raises replacement)."""
    try:
        yield
    except exc:
        return
    raise AssertionError(f"{exc.__name__} not raised")


def close(a: float, b: float, tol: float = 1e-9) -> bool:
    """Absolute/relative closeness."""
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))
