from __future__ import annotations

from typing import Mapping

from django.conf import settings

from .models import StrategyConfig


FEATURE_FLAGS_VERSION = "feature_flags_v1"
FEATURE_KEYS = {
    "multi": "feature_multi_strategy",
    "trend": "feature_mod_trend",
    "meanrev": "feature_mod_meanrev",
    "carry": "feature_mod_carry",
    "grid": "feature_mod_grid",
    "microvol": "feature_mod_microvol",
    "allocator": "feature_allocator",
}


def feature_flag_defaults() -> dict[str, bool]:
    return {
        FEATURE_KEYS["multi"]: bool(getattr(settings, "MULTI_STRATEGY_ENABLED", False)),
        FEATURE_KEYS["trend"]: bool(getattr(settings, "MODULE_TREND_ENABLED", True)),
        FEATURE_KEYS["meanrev"]: bool(getattr(settings, "MODULE_MEANREV_ENABLED", True)),
        FEATURE_KEYS["carry"]: bool(getattr(settings, "MODULE_CARRY_ENABLED", True)),
        FEATURE_KEYS["grid"]: bool(getattr(settings, "MODULE_GRID_ENABLED", False)),
        FEATURE_KEYS["microvol"]: bool(getattr(settings, "MODULE_MICROVOL_ENABLED", False)),
        FEATURE_KEYS["allocator"]: bool(getattr(settings, "ALLOCATOR_ENABLED", True)),
    }


def resolve_feature_flags(defaults: Mapping[str, bool]) -> dict[str, bool]:
    """
    Resolve runtime feature flags.

    Source is controlled by settings.FEATURE_FLAGS_SOURCE:
    - "env": use environment defaults as single source of truth.
    - "db": allow StrategyConfig rows to override runtime flags.
    """
    resolved = {str(name): bool(val) for name, val in defaults.items()}
    if not resolved:
        return {}

    if str(getattr(settings, "FEATURE_FLAGS_SOURCE", "env")).strip().lower() != "db":
        return resolved

    names = list(resolved.keys())
    try:
        rows = list(
            StrategyConfig.objects.filter(version=FEATURE_FLAGS_VERSION, name__in=names)
            .values("name", "enabled")
        )
        found = {str(row["name"]): bool(row["enabled"]) for row in rows}
        missing = [name for name in names if name not in found]
        if missing:
            StrategyConfig.objects.bulk_create(
                [
                    StrategyConfig(
                        name=name,
                        version=FEATURE_FLAGS_VERSION,
                        enabled=resolved[name],
                        params_json={"feature_flag": True},
                    )
                    for name in missing
                ],
                ignore_conflicts=True,
            )
        for name in names:
            if name in found:
                resolved[name] = found[name]
    except Exception:
        # Safe fallback to environment defaults when DB is unavailable.
        return resolved
    return resolved


def resolve_runtime_flags(
    overrides: Mapping[str, bool] | None = None,
) -> dict[str, bool]:
    defaults = feature_flag_defaults()
    if overrides:
        defaults.update({str(k): bool(v) for k, v in overrides.items()})
    return resolve_feature_flags(defaults)
