from __future__ import annotations

import json
import logging
from io import StringIO
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import redis
from django.conf import settings
from django.utils import timezone as dj_tz

from .models import MacroLiquiditySnapshot

logger = logging.getLogger(__name__)

HOWELL_CACHE_KEY = "signals:howell_liquidity:latest"
CORE_COMPONENT_WEIGHTS = {
    "fed_net_liquidity_z": 0.55,
    "financial_conditions_z": 0.25,
    "dollar_z": 0.20,
}
OPTIONAL_COMPONENT_WEIGHTS = {
    "stablecoin_growth_z": 0.0,
    "btc_etf_flow_z": 0.0,
}


def _redis_client():
    try:
        return redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)
    except Exception:
        return None


def _fred_series_url(series_id: str) -> str:
    return f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"


def _clip_score(value: float | None) -> float:
    clip = max(0.5, float(getattr(settings, "HOWELL_LIQUIDITY_COMPONENT_CLIP", 3.0) or 3.0))
    if value is None or not np.isfinite(value):
        return 0.0
    return round(float(max(-clip, min(clip, value))), 4)


def _clip_confidence(value: float) -> float:
    return round(float(max(0.0, min(0.99, value))), 4)


def _read_csv_from_url(url: str) -> pd.DataFrame:
    request = Request(url, headers={"User-Agent": "mastertrading-howell-overlay/1.0"})
    with urlopen(request, timeout=20) as response:
        payload = response.read().decode("utf-8")
    return pd.read_csv(StringIO(payload))


def _fetch_fred_series(series_id: str) -> pd.Series:
    df = _read_csv_from_url(_fred_series_url(series_id))
    if "DATE" not in df.columns or series_id not in df.columns:
        raise ValueError(f"fred series {series_id} missing expected columns")

    values = pd.to_numeric(df[series_id], errors="coerce")
    dates = pd.to_datetime(df["DATE"], utc=False, errors="coerce")
    series = pd.Series(values.values, index=dates).dropna().sort_index()
    series = series[~series.index.isna()]

    lookback_days = max(
        365,
        int(getattr(settings, "HOWELL_LIQUIDITY_LOOKBACK_DAYS", 1460) or 1460),
    )
    cutoff = pd.Timestamp(dj_tz.now().date()) - pd.Timedelta(days=lookback_days)
    series = series[series.index >= cutoff]
    if series.empty:
        raise ValueError(f"fred series {series_id} returned no rows after cutoff")
    return series.astype(float)


def _latest_zscore(series: pd.Series, *, minimum_obs: int = 20) -> float | None:
    clean = pd.Series(series).replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if len(clean) < minimum_obs:
        return None
    window = min(
        len(clean),
        max(
            minimum_obs,
            int(getattr(settings, "HOWELL_LIQUIDITY_ZSCORE_WINDOW_OBS", 104) or 104),
        ),
    )
    tail = clean.tail(window)
    std = float(tail.std(ddof=0))
    if std <= 1e-12:
        return 0.0
    return float((tail.iloc[-1] - tail.mean()) / std)


def _component_from_series(
    key: str,
    oriented_series: pd.Series,
    *,
    weight: float,
    source_ids: list[str],
) -> dict | None:
    clean = pd.Series(oriented_series).replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    change_window = max(
        2,
        int(getattr(settings, "HOWELL_LIQUIDITY_CHANGE_WINDOW_OBS", 13) or 13),
    )
    if len(clean) < max(24, change_window + 12):
        return None

    level_z = _latest_zscore(clean)
    change_z = _latest_zscore(clean.diff(change_window))
    if level_z is None or change_z is None:
        return None

    latest_idx = pd.Timestamp(clean.index[-1])
    return {
        "key": key,
        "weight": float(weight),
        "score": _clip_score(0.6 * level_z + 0.4 * change_z),
        "momentum": _clip_score(change_z),
        "level_z": _clip_score(level_z),
        "change_z": _clip_score(change_z),
        "asof": latest_idx.date().isoformat(),
        "latest_value": round(float(clean.iloc[-1]), 6),
        "source_ids": list(source_ids),
    }


def _build_core_components() -> dict[str, dict]:
    walcl = _fetch_fred_series("WALCL")
    wtregen = _fetch_fred_series("WTREGEN")
    rrpontsyd = _fetch_fred_series("RRPONTSYD")
    fci_source_id = "ANFCI"
    try:
        financial_conditions = _fetch_fred_series(fci_source_id)
    except Exception:
        fci_source_id = "NFCI"
        financial_conditions = _fetch_fred_series(fci_source_id)
    dollar = _fetch_fred_series("DTWEXBGS")

    net_liquidity_df = pd.concat(
        {
            "walcl": walcl,
            "wtregen": wtregen,
            "rrpontsyd": rrpontsyd,
        },
        axis=1,
    ).sort_index().ffill().dropna()
    net_liquidity = (
        net_liquidity_df["walcl"]
        - net_liquidity_df["wtregen"]
        - net_liquidity_df["rrpontsyd"]
    )

    components: dict[str, dict] = {}
    items = [
        (
            "fed_net_liquidity_z",
            net_liquidity,
            CORE_COMPONENT_WEIGHTS["fed_net_liquidity_z"],
            ["WALCL", "WTREGEN", "RRPONTSYD"],
        ),
        (
            "financial_conditions_z",
            -financial_conditions,
            CORE_COMPONENT_WEIGHTS["financial_conditions_z"],
            [fci_source_id],
        ),
        (
            "dollar_z",
            -dollar,
            CORE_COMPONENT_WEIGHTS["dollar_z"],
            ["DTWEXBGS"],
        ),
    ]
    for key, oriented, weight, source_ids in items:
        component = _component_from_series(
            key,
            oriented,
            weight=weight,
            source_ids=source_ids,
        )
        if component:
            components[key] = component
    return components


def _classify_regime(
    composite_score: float,
    composite_momentum: float,
    components: dict[str, dict],
) -> str:
    stress_score_max = float(
        getattr(settings, "HOWELL_LIQUIDITY_STRESS_SCORE_MAX", -0.75) or -0.75
    )
    stress_fci_max = float(
        getattr(settings, "HOWELL_LIQUIDITY_STRESS_FCI_Z_MAX", -0.90) or -0.90
    )
    expanding_score_min = float(
        getattr(settings, "HOWELL_LIQUIDITY_EXPANDING_SCORE_MIN", 0.35) or 0.35
    )
    expanding_momentum_min = float(
        getattr(settings, "HOWELL_LIQUIDITY_EXPANDING_MOMENTUM_MIN", 0.10) or 0.10
    )
    late_expanding_score_min = float(
        getattr(settings, "HOWELL_LIQUIDITY_LATE_EXPANDING_SCORE_MIN", 0.05) or 0.05
    )
    rollover_momentum_max = float(
        getattr(settings, "HOWELL_LIQUIDITY_ROLLOVER_MOMENTUM_MAX", -0.20) or -0.20
    )
    fci_score = float(
        (components.get("financial_conditions_z") or {}).get("score", 0.0) or 0.0
    )

    if composite_score <= stress_score_max or fci_score <= stress_fci_max:
        return "stress"
    if (
        composite_score >= expanding_score_min
        and composite_momentum >= expanding_momentum_min
    ):
        return "expanding"
    if composite_momentum <= rollover_momentum_max:
        return "rollover"
    if composite_score >= late_expanding_score_min:
        return "late_expanding"
    return "rollover"


def _compose_snapshot(components: dict[str, dict]) -> dict:
    active_components = [component for component in components.values() if component]
    if not active_components:
        return {
            "asof": dj_tz.now(),
            "regime": "unavailable",
            "confidence": 0.0,
            "composite_score": 0.0,
            "composite_momentum": 0.0,
            "fed_net_liquidity_z": 0.0,
            "financial_conditions_z": 0.0,
            "dollar_z": 0.0,
            "stablecoin_growth_z": 0.0,
            "btc_etf_flow_z": 0.0,
            "details_json": {"reason": "no_components"},
        }

    total_weight = sum(float(component["weight"]) for component in active_components) or 1.0
    composite_score = sum(
        float(component["weight"]) * float(component["score"])
        for component in active_components
    ) / total_weight
    composite_momentum = sum(
        float(component["weight"]) * float(component["momentum"])
        for component in active_components
    ) / total_weight
    regime = _classify_regime(composite_score, composite_momentum, components)

    score_sign = 0.0 if abs(composite_score) < 1e-9 else float(np.sign(composite_score))
    agreement = 0.0
    if active_components:
        agreement = sum(
            1.0
            for component in active_components
            if abs(float(component["score"])) < 0.15
            or float(np.sign(float(component["score"]))) == score_sign
        ) / float(len(active_components))

    today = dj_tz.now().date()
    source_age_days = {}
    for component in active_components:
        asof = pd.Timestamp(component["asof"]).date()
        source_age_days[component["key"]] = max(0, (today - asof).days)
    max_age_days = max(source_age_days.values()) if source_age_days else 0
    confidence = _clip_confidence(
        0.30
        + 0.12 * len(active_components)
        + 0.20 * agreement
        - 0.02 * max(0, max_age_days - 7)
    )

    component_values = {
        key: round(float((components.get(key) or {}).get("score", 0.0) or 0.0), 4)
        for key in (
            "fed_net_liquidity_z",
            "financial_conditions_z",
            "dollar_z",
            "stablecoin_growth_z",
            "btc_etf_flow_z",
        )
    }
    component_meta = {
        component["key"]: {
            "score": float(component["score"]),
            "momentum": float(component["momentum"]),
            "level_z": float(component["level_z"]),
            "change_z": float(component["change_z"]),
            "latest_value": float(component["latest_value"]),
            "asof": component["asof"],
            "weight": float(component["weight"]),
            "source_ids": list(component["source_ids"]),
        }
        for component in active_components
    }

    return {
        "asof": dj_tz.now(),
        "regime": regime,
        "confidence": confidence,
        "composite_score": round(float(composite_score), 4),
        "composite_momentum": round(float(composite_momentum), 4),
        **component_values,
        "details_json": {
            "shadow_only": True,
            "active_components": list(component_meta.keys()),
            "component_count": len(component_meta),
            "component_weights": {
                **CORE_COMPONENT_WEIGHTS,
                **OPTIONAL_COMPONENT_WEIGHTS,
            },
            "components": component_meta,
            "source_age_days": source_age_days,
            "source_max_age_days": max_age_days,
            "agreement": round(float(agreement), 4),
        },
    }


def _snapshot_row_to_dict(row: MacroLiquiditySnapshot) -> dict:
    return {
        "asof": row.asof.isoformat(),
        "regime": row.regime,
        "confidence": round(float(row.confidence or 0.0), 4),
        "composite_score": round(float(row.composite_score or 0.0), 4),
        "composite_momentum": round(float(row.composite_momentum or 0.0), 4),
        "fed_net_liquidity_z": round(float(row.fed_net_liquidity_z or 0.0), 4),
        "financial_conditions_z": round(float(row.financial_conditions_z or 0.0), 4),
        "dollar_z": round(float(row.dollar_z or 0.0), 4),
        "stablecoin_growth_z": round(float(row.stablecoin_growth_z or 0.0), 4),
        "btc_etf_flow_z": round(float(row.btc_etf_flow_z or 0.0), 4),
        "details_json": dict(row.details_json or {}),
    }


def _cache_snapshot(payload: dict) -> None:
    client = _redis_client()
    if client is None:
        return
    try:
        ttl_seconds = max(
            3600,
            int(getattr(settings, "HOWELL_LIQUIDITY_CACHE_TTL_HOURS", 24) or 24) * 3600,
        )
        client.setex(
            HOWELL_CACHE_KEY,
            ttl_seconds,
            json.dumps(payload, ensure_ascii=True, separators=(",", ":")),
        )
    except Exception:
        pass


def _latest_db_snapshot() -> dict | None:
    row = MacroLiquiditySnapshot.objects.order_by("-asof", "-id").first()
    if not row:
        return None
    payload = _snapshot_row_to_dict(row)
    _cache_snapshot(payload)
    return payload


def refresh_liquidity_snapshot() -> dict | None:
    if not bool(getattr(settings, "HOWELL_LIQUIDITY_ENABLED", False)):
        return None

    components = _build_core_components()
    payload = _compose_snapshot(components)
    snapshot = MacroLiquiditySnapshot.objects.create(**payload)
    result = _snapshot_row_to_dict(snapshot)
    _cache_snapshot(result)
    logger.info(
        "howell liquidity snapshot regime=%s score=%.3f momentum=%.3f confidence=%.2f components=%s",
        result["regime"],
        result["composite_score"],
        result["composite_momentum"],
        result["confidence"],
        ",".join(result["details_json"].get("active_components", [])),
    )
    return result


def get_cached_liquidity_snapshot() -> dict | None:
    if not bool(getattr(settings, "HOWELL_LIQUIDITY_ENABLED", False)):
        return None

    client = _redis_client()
    if client is not None:
        try:
            raw = client.get(HOWELL_CACHE_KEY)
            if raw:
                payload = json.loads(raw)
                if isinstance(payload, dict):
                    return payload
        except Exception:
            pass
    return _latest_db_snapshot()


def _shadow_bucket(symbol: str) -> str:
    sym = str(symbol or "").strip().upper()
    tier_map = getattr(settings, "INSTRUMENT_TIER_MAP", {}) or {}
    tier = str(tier_map.get(sym, "")).strip().lower()
    if sym in {"BTCUSDT", "ETHUSDT"} or tier == "base":
        return "base"
    return "alt"


def _preview_risk_mult(regime: str, bucket: str) -> float:
    bucket_key = "BASE" if bucket == "base" else "ALT"
    regime_key = str(regime or "unavailable").strip().lower()
    mapping = {
        ("BASE", "expanding"): 1.00,
        ("BASE", "late_expanding"): 0.90,
        ("BASE", "rollover"): 0.75,
        ("BASE", "stress"): 0.55,
        ("ALT", "expanding"): 1.00,
        ("ALT", "late_expanding"): 0.80,
        ("ALT", "rollover"): 0.60,
        ("ALT", "stress"): 0.35,
        ("BASE", "unavailable"): 1.00,
        ("ALT", "unavailable"): 1.00,
    }
    env_name = f"HOWELL_LIQUIDITY_PREVIEW_{bucket_key}_MULT_{regime_key.upper()}"
    default = mapping.get((bucket_key, regime_key), 1.0)
    return round(float(getattr(settings, env_name, default) or default), 4)


def howell_shadow_diagnostic(symbol: str, snapshot: dict | None = None) -> dict | None:
    if not bool(getattr(settings, "HOWELL_LIQUIDITY_ENABLED", False)):
        return None

    payload = snapshot or get_cached_liquidity_snapshot()
    bucket = _shadow_bucket(symbol)
    if not payload:
        return {
            "enabled": True,
            "shadow_only": True,
            "regime": "unavailable",
            "confidence": 0.0,
            "bucket": bucket,
            "preview_risk_mult": _preview_risk_mult("unavailable", bucket),
            "reason": "no_snapshot",
        }

    details = payload.get("details_json", {}) if isinstance(payload, dict) else {}
    return {
        "enabled": True,
        "shadow_only": True,
        "regime": str(payload.get("regime", "unavailable")),
        "confidence": round(float(payload.get("confidence", 0.0) or 0.0), 4),
        "composite_score": round(float(payload.get("composite_score", 0.0) or 0.0), 4),
        "composite_momentum": round(float(payload.get("composite_momentum", 0.0) or 0.0), 4),
        "bucket": bucket,
        "preview_risk_mult": _preview_risk_mult(payload.get("regime", "unavailable"), bucket),
        "snapshot_asof": payload.get("asof"),
        "source_max_age_days": details.get("source_max_age_days"),
        "component_count": details.get("component_count", 0),
        "active_components": details.get("active_components", []),
        "reason": "diagnostic_only",
    }
