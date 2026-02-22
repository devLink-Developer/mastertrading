from __future__ import annotations

from typing import Any

from django.conf import settings

from signals.models import StrategyConfig

REPORT_CONTROL_NAME = "control_performance_report"
REPORT_CONTROL_VERSION = "runtime_ctrl_v1"


def _to_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    txt = str(value or "").strip().lower()
    if txt in {"1", "true", "yes", "on"}:
        return True
    if txt in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _to_int(value: Any, default: int, lo: int, hi: int) -> int:
    try:
        num = int(value)
    except Exception:
        num = int(default)
    return max(lo, min(hi, num))


def default_report_config() -> dict[str, Any]:
    mode = str(getattr(settings, "PERFORMANCE_REPORT_BEAT_MODE", "interval") or "interval").strip().lower()
    if mode not in {"interval", "daily"}:
        mode = "interval"
    return {
        "enabled": bool(getattr(settings, "PERFORMANCE_REPORT_ENABLED", True)),
        "beat_enabled": bool(getattr(settings, "PERFORMANCE_REPORT_BEAT_ENABLED", True)),
        "mode": mode,
        "beat_minutes": _to_int(getattr(settings, "PERFORMANCE_REPORT_BEAT_MINUTES", 15), 15, 1, 1440),
        "beat_hour": _to_int(getattr(settings, "PERFORMANCE_REPORT_BEAT_HOUR", 0), 0, 0, 23),
        "beat_minute": _to_int(getattr(settings, "PERFORMANCE_REPORT_BEAT_MINUTE", 0), 0, 0, 59),
        "window_minutes": _to_int(getattr(settings, "PERFORMANCE_REPORT_WINDOW_MINUTES", 15), 15, 1, 1440),
    }


def _normalize_with_defaults(payload: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    out = dict(defaults)
    if not isinstance(payload, dict):
        return out

    if "enabled" in payload:
        out["enabled"] = _to_bool(payload.get("enabled"), out["enabled"])
    if "beat_enabled" in payload:
        out["beat_enabled"] = _to_bool(payload.get("beat_enabled"), out["beat_enabled"])

    mode = str(payload.get("mode", out["mode"]) or out["mode"]).strip().lower()
    if mode in {"interval", "daily"}:
        out["mode"] = mode

    out["beat_minutes"] = _to_int(payload.get("beat_minutes", out["beat_minutes"]), out["beat_minutes"], 1, 1440)
    out["beat_hour"] = _to_int(payload.get("beat_hour", out["beat_hour"]), out["beat_hour"], 0, 23)
    out["beat_minute"] = _to_int(payload.get("beat_minute", out["beat_minute"]), out["beat_minute"], 0, 59)
    out["window_minutes"] = _to_int(payload.get("window_minutes", out["window_minutes"]), out["window_minutes"], 1, 1440)
    return out


def _params_from_config(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "beat_enabled": bool(cfg.get("beat_enabled", True)),
        "mode": str(cfg.get("mode", "interval")),
        "beat_minutes": int(cfg.get("beat_minutes", 15)),
        "beat_hour": int(cfg.get("beat_hour", 0)),
        "beat_minute": int(cfg.get("beat_minute", 0)),
        "window_minutes": int(cfg.get("window_minutes", 15)),
    }


def resolve_report_config() -> dict[str, Any]:
    defaults = default_report_config()
    try:
        row, _ = StrategyConfig.objects.get_or_create(
            name=REPORT_CONTROL_NAME,
            version=REPORT_CONTROL_VERSION,
            defaults={
                "enabled": bool(defaults["enabled"]),
                "params_json": _params_from_config(defaults),
            },
        )
    except Exception:
        return defaults

    payload = row.params_json if isinstance(row.params_json, dict) else {}
    merged = _normalize_with_defaults(payload, defaults)
    merged["enabled"] = bool(row.enabled)
    return merged


def update_report_config(**changes: Any) -> dict[str, Any]:
    defaults = resolve_report_config()
    try:
        row, _ = StrategyConfig.objects.get_or_create(
            name=REPORT_CONTROL_NAME,
            version=REPORT_CONTROL_VERSION,
            defaults={
                "enabled": bool(defaults["enabled"]),
                "params_json": _params_from_config(defaults),
            },
        )
    except Exception:
        return defaults

    payload = row.params_json if isinstance(row.params_json, dict) else {}
    payload.update({k: v for k, v in changes.items() if k in {
        "beat_enabled",
        "mode",
        "beat_minutes",
        "beat_hour",
        "beat_minute",
        "window_minutes",
    }})

    if "enabled" in changes:
        row.enabled = _to_bool(changes.get("enabled"), bool(row.enabled))

    normalized = _normalize_with_defaults(payload, defaults)
    row.params_json = _params_from_config(normalized)
    try:
        row.save(update_fields=["enabled", "params_json"])
    except Exception:
        return resolve_report_config()
    return resolve_report_config()
