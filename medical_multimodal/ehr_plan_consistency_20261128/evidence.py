"""Validate paginated evidence before evaluating a time-window mean."""
from dataclasses import dataclass
from math import isfinite
from statistics import mean


class CoverageError(ValueError):
    """The executor cannot establish that the evidence is complete."""


@dataclass(frozen=True)
class Page:
    rows: tuple
    cursor: str | None


@dataclass(frozen=True)
class CoveredMean:
    value: float | None
    pages_read: int
    added_in_window: int
    confirmation_only_reads: int


def read_mean(first, fetch, *, hours=24, unique_ages=False, max_pages=16):
    """Consume newest-first pages until the window is covered or fail before computing."""
    if not isfinite(hours) or hours <= 0 or max_pages < 0:
        raise ValueError("invalid window or page budget")
    current = first
    last_age = None
    visited = set()
    values = []
    pages_read = added = confirmation = 0
    while True:
        count = len(values)
        for age, value in current.rows:
            if not isfinite(age) or age < 0 or not isfinite(value):
                raise CoverageError("invalid observation")
            if last_age is not None and (age < last_age or (unique_ages and age == last_age)):
                raise CoverageError("receipt violates its timestamp ordering contract")
            last_age = age
            if age <= hours:
                values.append(value)
        if pages_read:
            new_values = len(values) - count
            added += new_values
            confirmation += new_values == 0
        covered = last_age is not None and (
            last_age > hours or (unique_ages and last_age == hours)
        )
        if current.cursor is None or covered:
            return CoveredMean(mean(values) if values else None, pages_read, added, confirmation)
        if current.cursor in visited:
            raise CoverageError("pagination cursor repeated")
        if pages_read == max_pages:
            raise CoverageError("page budget exhausted before the window was covered")
        visited.add(current.cursor)
        current = fetch(current.cursor)
        pages_read += 1
