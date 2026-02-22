from __future__ import annotations

from datetime import datetime, timezone

# UTC sessions (explicitly checked in order)
SESSIONS: dict[str, tuple[int, int]] = {
    "overlap": (12, 14),
    "london": (6, 14),
    "ny": (14, 20),
    "dead": (20, 23),
    "asia": (23, 6),
}

DEFAULT_SCORE_MIN: dict[str, float] = {
    "overlap": 0.73,
    "london": 0.75,
    "ny": 0.75,
    "asia": 0.80,
    "dead": 1.00,
}

DEFAULT_RISK_MULT: dict[str, float] = {
    "overlap": 1.0,
    "london": 1.0,
    "ny": 1.0,
    "asia": 1.0,
    "dead": 0.0,
}


def get_current_session(utc_hour: int | None = None) -> str:
    if utc_hour is None:
        utc_hour = datetime.now(timezone.utc).hour
    for name, (start, end) in SESSIONS.items():
        if start < end:
            if start <= utc_hour < end:
                return name
        else:
            if utc_hour >= start or utc_hour < end:
                return name
    return "dead"


def is_dead_session(session: str) -> bool:
    return session == "dead"


def _override_float(overrides: dict | None, session: str, default: float) -> float:
    if not overrides:
        return default
    try:
        if session in overrides:
            return float(overrides[session])
    except Exception:
        return default
    return default


def get_session_score_min(session: str, overrides: dict | None = None) -> float:
    default = DEFAULT_SCORE_MIN.get(session, 0.75)
    return _override_float(overrides, session, default)


def get_session_risk_mult(session: str, overrides: dict | None = None) -> float:
    default = DEFAULT_RISK_MULT.get(session, 1.0)
    return _override_float(overrides, session, default)
