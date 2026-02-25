from __future__ import annotations

import logging
from datetime import timedelta

from django.conf import settings
from django.db.models import Q

from core.models import Instrument
from signals.allocator import (
    MODULE_ORDER,
    default_risk_budget_map,
    default_weight_map,
    dynamic_weight_map,
    resolve_symbol_allocation,
)
from signals.feature_flags import FEATURE_KEYS, resolve_runtime_flags
from signals.models import Signal
from signals.sessions import get_current_session, get_session_risk_mult
from signals.modules import carry as carry_module
from signals.modules import meanrev as meanrev_module
from signals.modules import trend as trend_module
from signals.modules.common import (
    acquire_task_lock,
    emit_signal,
    get_multi_universe_instruments,
    latest_candles,
    latest_funding_rates,
    now_utc,
    strategy_module,
)

logger = logging.getLogger(__name__)


def _module_detector(module_name: str):
    if module_name == "trend":
        return trend_module.detect
    if module_name == "meanrev":
        return meanrev_module.detect
    if module_name == "carry":
        return carry_module.detect
    raise ValueError(f"unknown module {module_name}")


def run_module_engine(module_name: str) -> str:
    flags = resolve_runtime_flags()
    if not flags.get(FEATURE_KEYS["multi"], False):
        return f"{module_name}:disabled_multi"
    if not flags.get(FEATURE_KEYS.get(module_name, ""), False):
        return f"{module_name}:disabled_flag"

    if not acquire_task_lock(f"module:{module_name}", ttl_seconds=55):
        logger.info("%s module skipped due lock", module_name)
        return f"{module_name}:locked"

    emitted = 0
    now = now_utc()
    lookback = int(getattr(settings, "MODULE_LOOKBACK_BARS", 240))
    warmup = int(getattr(settings, "MODULE_SYMBOL_WARMUP_BARS", 300))
    detector = _module_detector(module_name)

    instruments = get_multi_universe_instruments()
    for inst in instruments:
        df_ltf = latest_candles(inst, "5m", lookback=lookback)
        df_htf = latest_candles(inst, "1h", lookback=lookback)
        if len(df_ltf) < warmup or len(df_htf) < max(80, warmup // 2):
            continue
        funding = latest_funding_rates(inst, lookback=80) if module_name == "carry" else []
        session = get_current_session(now.hour)
        result = detector(df_ltf, df_htf, funding, session)
        if not result:
            continue
        direction = str(result.get("direction", "")).strip().lower()
        if direction not in {"long", "short"}:
            continue
        strategy = f"mod_{module_name}_{direction}"
        ok = emit_signal(
            inst,
            strategy=strategy,
            module=module_name,
            direction=direction,
            raw_score=float(result.get("raw_score", 0.0)),
            confidence=float(result.get("confidence", 0.0)),
            reasons=dict(result.get("reasons", {})),
            symbol_state="open",
            dedup_seconds=int(getattr(settings, "MODULE_SIGNAL_TTL_SECONDS", 120)),
            ts=now,
        )
        if ok:
            emitted += 1
    logger.info("%s module emitted=%d", module_name, emitted)
    return f"{module_name}:emitted={emitted}"


def _active_modules(flags: dict[str, bool]) -> list[str]:
    modules = []
    if flags.get(FEATURE_KEYS["trend"], False):
        modules.append("trend")
    if flags.get(FEATURE_KEYS["meanrev"], False):
        modules.append("meanrev")
    if flags.get(FEATURE_KEYS["carry"], False):
        modules.append("carry")
    if bool(getattr(settings, "ALLOCATOR_INCLUDE_SMC", False)):
        modules.append("smc")
    if bool(getattr(settings, "LIVE_GRADUAL_ENABLED", True)):
        cap = max(1, int(getattr(settings, "LIVE_GRADUAL_MAX_MODULES", len(modules))))
        raw_priority = getattr(settings, "LIVE_GRADUAL_MODULE_PRIORITY", {}) or {}
        priority_map: dict[str, float] = {}
        if isinstance(raw_priority, dict):
            for key, value in raw_priority.items():
                try:
                    priority_map[str(key).strip().lower()] = float(value)
                except Exception:
                    continue
        if not priority_map:
            weights = default_weight_map()
            priority_map = {m: float(weights.get(m, 0.0)) for m in MODULE_ORDER}
        modules = sorted(
            modules,
            key=lambda name: (
                -float(priority_map.get(name, 0.0)),
                MODULE_ORDER.index(name) if name in MODULE_ORDER else 999,
            ),
        )
        modules = modules[:cap]
    return modules


def _allowed_symbols_for_module(module_name: str, instruments: list[Instrument]) -> set[str]:
    symbols = [inst.symbol for inst in instruments]
    if not bool(getattr(settings, "LIVE_GRADUAL_ENABLED", True)):
        return set(symbols)
    cap = max(1, int(getattr(settings, "LIVE_GRADUAL_MAX_SYMBOLS_PER_MODULE", len(symbols))))
    return set(sorted(symbols)[:cap])


def _smc_reason_payload(payload: dict | None) -> dict:
    if not isinstance(payload, dict):
        return {}
    reason = payload.get("reason")
    if isinstance(reason, dict):
        return reason
    reasons = payload.get("reasons")
    if isinstance(reasons, dict):
        return reasons
    return {}


def _smc_confluence_meta(payload: dict | None, confidence: float, fallback_session: str) -> dict:
    reason = _smc_reason_payload(payload)
    conditions = reason.get("conditions") if isinstance(reason.get("conditions"), dict) else {}
    liquidity_ok = bool(conditions.get("liquidity_sweep"))
    structure_ok = bool(conditions.get("structure_break"))
    if not liquidity_ok:
        liquidity_ok = str(reason.get("liquidity_sweep", "")).strip().lower() not in {"", "none", "null"}
    if not structure_ok:
        structure_ok = str(reason.get("structure_break", "")).strip().lower() not in {"", "none", "null"}

    signal_session = str(reason.get("session") or fallback_session).strip().lower()
    allowed_sessions = set(getattr(settings, "ALLOCATOR_SMC_ALLOWED_SESSIONS", set()) or set())
    session_ok = (not allowed_sessions) or (signal_session in allowed_sessions)
    min_conf = float(getattr(settings, "ALLOCATOR_SMC_CONFLUENCE_MIN_SCORE", 0.80))
    score_ok = float(confidence) >= min_conf
    smc_confluence = bool(liquidity_ok and structure_ok and session_ok and score_ok)

    return {
        "smc_confluence": smc_confluence,
        "smc_liquidity_ok": bool(liquidity_ok),
        "smc_structure_ok": bool(structure_ok),
        "smc_session": signal_session,
        "smc_session_ok": bool(session_ok),
        "smc_score_ok": bool(score_ok),
    }


def run_allocator_cycle() -> str:
    flags = resolve_runtime_flags()
    if not flags.get(FEATURE_KEYS["multi"], False):
        return "allocator:disabled_multi"
    if not flags.get(FEATURE_KEYS["allocator"], False):
        return "allocator:disabled_flag"

    if not acquire_task_lock("allocator", ttl_seconds=55):
        logger.info("allocator skipped due lock")
        return "allocator:locked"

    instruments = get_multi_universe_instruments()
    if not instruments:
        return "allocator:no_instruments"

    active_modules = _active_modules(flags)
    if not active_modules:
        return "allocator:no_active_modules"

    now = now_utc()
    window_seconds = max(70, int(getattr(settings, "ALLOCATOR_WINDOW_SECONDS", 130)))
    window_start = now - timedelta(seconds=window_seconds)
    min_modules = max(1, int(getattr(settings, "ALLOCATOR_MIN_MODULES_ACTIVE", 2)))
    # Dynamic weights: Bayesian rolling adjustment when enabled
    if bool(getattr(settings, "ALLOCATOR_DYNAMIC_WEIGHTS_ENABLED", False)):
        weights = dynamic_weight_map(base_weights=default_weight_map())
    else:
        weights = default_weight_map()
    risk_budgets = default_risk_budget_map()
    threshold = float(getattr(settings, "ALLOCATOR_NET_THRESHOLD", 0.20))
    base_risk = float(getattr(settings, "RISK_PER_TRADE_PCT", 0.01))
    session = get_current_session(now.hour)
    session_risk_mult = float(
        get_session_risk_mult(session, getattr(settings, "SESSION_RISK_MULTIPLIER", {}))
    )

    symbol_allow_map = {
        module: _allowed_symbols_for_module(module, instruments) for module in active_modules
    }
    inst_by_id = {inst.id: inst for inst in instruments}
    signal_qs = (
        Signal.objects.filter(
            ts__gte=window_start,
            instrument_id__in=list(inst_by_id.keys()),
        )
        .filter(
            Q(strategy__startswith="mod_trend_")
            | Q(strategy__startswith="mod_meanrev_")
            | Q(strategy__startswith="mod_carry_")
            | Q(strategy__startswith="smc_")
        )
        .order_by("-ts")
    )

    latest: dict[tuple[int, str], Signal] = {}
    for sig in signal_qs:
        module_name, direction = strategy_module(sig.strategy)
        if module_name not in active_modules:
            continue
        if direction not in {"long", "short"}:
            continue
        inst = inst_by_id.get(sig.instrument_id)
        if not inst:
            continue
        if inst.symbol not in symbol_allow_map.get(module_name, set()):
            continue
        key = (sig.instrument_id, module_name)
        if key not in latest:
            latest[key] = sig

    by_symbol: dict[int, list[dict]] = {inst.id: [] for inst in instruments}
    for (inst_id, module_name), sig in latest.items():
        payload = sig.payload_json or {}
        confidence = float(payload.get("confidence", sig.score or 0.0))
        raw_score = float(payload.get("raw_score", confidence))
        direction = str(payload.get("direction", "")).strip().lower()
        if direction not in {"long", "short"}:
            _, direction = strategy_module(sig.strategy)
        smc_meta = {}
        if module_name == "smc":
            smc_meta = _smc_confluence_meta(payload, confidence, session)
        by_symbol[inst_id].append(
            {
                "module": module_name,
                "direction": direction,
                "confidence": confidence,
                "raw_score": raw_score,
                "signal_id": sig.id,
                "ts": sig.ts.isoformat(),
                "reasons": payload.get("reasons", {}) if isinstance(payload, dict) else {},
                **smc_meta,
            }
        )

    emitted = 0
    hmm_enabled = bool(getattr(settings, "HMM_REGIME_ENABLED", False))
    for inst in instruments:
        module_rows = by_symbol.get(inst.id, [])

        # Regime-adjusted risk multiplier
        inst_regime_mult = 1.0
        if hmm_enabled:
            from .regime import regime_risk_mult as _rrm
            inst_regime_mult = _rrm(inst.symbol)

        if len(module_rows) < min_modules:
            alloc = {
                "direction": "flat",
                "raw_score": 0.0,
                "net_score": 0.0,
                "confidence": 0.0,
                "risk_budget_pct": 0.0,
                "symbol_state": "blocked",
                "reasons": {
                    "session": session,
                    "active_module_count": len(module_rows),
                    "required_modules": min_modules,
                    "module_rows": module_rows,
                },
            }
        else:
            alloc = resolve_symbol_allocation(
                module_rows,
                threshold=threshold,
                base_risk_pct=base_risk,
                session_risk_mult=session_risk_mult * inst_regime_mult,
                weights=weights,
                risk_budgets=risk_budgets,
                min_active_modules=min_modules,
            )
            alloc["reasons"]["session"] = session
            alloc["reasons"]["module_rows"] = module_rows
            alloc["reasons"]["regime_risk_mult"] = round(inst_regime_mult, 4)

        strategy = f"alloc_{alloc['direction']}"
        ok = emit_signal(
            inst,
            strategy=strategy,
            module="allocator",
            direction=str(alloc["direction"]),
            raw_score=float(alloc["raw_score"]),
            confidence=float(alloc["confidence"]),
            reasons=dict(alloc["reasons"]),
            net_score=float(alloc["net_score"]),
            risk_budget_pct=float(alloc["risk_budget_pct"]),
            symbol_state=str(alloc["symbol_state"]),
            dedup_seconds=45,
            ts=now,
        )
        if ok:
            emitted += 1
    logger.info(
        "allocator emitted=%d session=%s active_modules=%s",
        emitted,
        session,
        active_modules,
    )
    return f"allocator:emitted={emitted}"
