from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter
from typing import Iterator


@contextmanager
def profile_stage(label: str) -> Iterator[None]:
    started = perf_counter()
    try:
        yield
    finally:
        print(f"[PROFILE] {label}: {perf_counter() - started:.4f} seconds")