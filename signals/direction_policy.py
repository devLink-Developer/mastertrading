from __future__ import annotations

from typing import Optional

from django.conf import settings


VALID_DIRECTION_MODES = {"both", "long_only", "short_only", "disabled"}


def normalize_direction_mode(raw_mode: str | None) -> str:
    mode = (raw_mode or "").strip().lower()
    return mode if mode in VALID_DIRECTION_MODES else "both"


def normalize_direction(direction: str | None) -> Optional[str]:
    value = (direction or "").strip().lower()
    if value in {"long", "buy"}:
        return "long"
    if value in {"short", "sell"}:
        return "short"
    return None


def get_direction_mode(
    symbol: str | None = None,
    overrides: dict | None = None,
) -> str:
    per_symbol = overrides if overrides is not None else getattr(settings, "PER_INSTRUMENT_DIRECTION", {})
    if symbol:
        mode = per_symbol.get(str(symbol).upper())
        if mode:
            return normalize_direction_mode(mode)
    return normalize_direction_mode(getattr(settings, "SIGNAL_DIRECTION_MODE", "both"))


def is_direction_allowed(
    direction: str | None,
    symbol: str | None = None,
    mode: str | None = None,
    overrides: dict | None = None,
) -> bool:
    canonical_direction = normalize_direction(direction)
    if canonical_direction is None:
        return False

    active_mode = normalize_direction_mode(
        mode if mode is not None else get_direction_mode(symbol=symbol, overrides=overrides)
    )
    if active_mode == "disabled":
        return False
    if active_mode == "both":
        return True
    if active_mode == "long_only":
        return canonical_direction == "long"
    if active_mode == "short_only":
        return canonical_direction == "short"
    return True
