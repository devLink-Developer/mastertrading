from __future__ import annotations

from datetime import datetime, timezone

SESSION_WINDOWS: tuple[tuple[str, tuple[int, int], tuple[int, int]], ...] = (
    ("ny_open", (13, 30), (14, 0)),
    ("overlap", (12, 0), (13, 30)),
    ("london", (6, 0), (12, 0)),
    ("ny", (14, 0), (20, 0)),
    ("dead", (20, 0), (23, 0)),
    ("asia", (23, 0), (6, 0)),
)

SESSION_NAMES = tuple(name for name, _start, _end in SESSION_WINDOWS)
WEEKDAY_NAMES = (
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
)

DEFAULT_SCORE_MIN: dict[str, float] = {
    "ny_open": 0.68,
    "overlap": 0.73,
    "london": 0.75,
    "ny": 0.75,
    "asia": 0.80,
    "dead": 1.00,
}

DEFAULT_RISK_MULT: dict[str, float] = {
    "ny_open": 0.65,
    "overlap": 1.0,
    "london": 1.0,
    "ny": 1.0,
    "asia": 1.0,
    "dead": 0.0,
}

DEFAULT_WEEKDAY_SCORE_OFFSET: dict[str, float] = {
    "monday": 0.01,
    "friday": 0.01,
    "saturday": 0.03,
    "sunday": 0.03,
}

DEFAULT_WEEKDAY_RISK_MULT: dict[str, float] = {
    "monday": 0.95,
    "friday": 0.95,
    "saturday": 0.85,
    "sunday": 0.85,
}


def _coerce_utc_parts(utc_time: datetime | int | None) -> tuple[int, int]:
    if utc_time is None:
        current = datetime.now(timezone.utc)
        return current.hour, current.minute
    if isinstance(utc_time, datetime):
        if utc_time.tzinfo is None:
            current = utc_time.replace(tzinfo=timezone.utc)
        else:
            current = utc_time.astimezone(timezone.utc)
        return current.hour, current.minute
    try:
        return int(utc_time), 0
    except Exception:
        current = datetime.now(timezone.utc)
        return current.hour, current.minute


def _time_to_minutes(hour: int, minute: int) -> int:
    return (int(hour) * 60) + int(minute)


def get_current_session(utc_time: datetime | int | None = None) -> str:
    hour, minute = _coerce_utc_parts(utc_time)
    current_minute = _time_to_minutes(hour, minute)
    for name, start, end in SESSION_WINDOWS:
        start_minute = _time_to_minutes(*start)
        end_minute = _time_to_minutes(*end)
        if start_minute < end_minute:
            if start_minute <= current_minute < end_minute:
                return name
        else:
            if current_minute >= start_minute or current_minute < end_minute:
                return name
    return "dead"


def is_dead_session(session: str) -> bool:
    return str(session or "").strip().lower() == "dead"


def get_weekday_name(utc_time: datetime | int | None = None) -> str:
    if utc_time is None:
        current = datetime.now(timezone.utc)
        return WEEKDAY_NAMES[current.weekday()]
    if isinstance(utc_time, datetime):
        current = utc_time if utc_time.tzinfo else utc_time.replace(tzinfo=timezone.utc)
        current = current.astimezone(timezone.utc)
        return WEEKDAY_NAMES[current.weekday()]
    try:
        idx = int(utc_time)
    except Exception:
        current = datetime.now(timezone.utc)
        return WEEKDAY_NAMES[current.weekday()]
    if 0 <= idx < len(WEEKDAY_NAMES):
        return WEEKDAY_NAMES[idx]
    current = datetime.now(timezone.utc)
    return WEEKDAY_NAMES[current.weekday()]


def _override_float(overrides: dict | None, key: str, default: float) -> float:
    if not overrides:
        return default
    try:
        if key in overrides:
            return float(overrides[key])
        if "*" in overrides:
            return float(overrides["*"])
    except Exception:
        return default
    return default


def get_session_score_min(session: str, overrides: dict | None = None) -> float:
    default = DEFAULT_SCORE_MIN.get(str(session or "").strip().lower(), 0.75)
    return _override_float(overrides, str(session or "").strip().lower(), default)


def get_session_risk_mult(session: str, overrides: dict | None = None) -> float:
    default = DEFAULT_RISK_MULT.get(str(session or "").strip().lower(), 1.0)
    return _override_float(overrides, str(session or "").strip().lower(), default)


def get_weekday_score_offset(weekday_name: str, overrides: dict | None = None) -> float:
    key = str(weekday_name or "").strip().lower()
    default = DEFAULT_WEEKDAY_SCORE_OFFSET.get(key, 0.0)
    return _override_float(overrides, key, default)


def get_weekday_risk_mult(weekday_name: str, overrides: dict | None = None) -> float:
    key = str(weekday_name or "").strip().lower()
    default = DEFAULT_WEEKDAY_RISK_MULT.get(key, 1.0)
    return _override_float(overrides, key, default)
