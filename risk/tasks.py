from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import Any

import redis
from celery import shared_task
from django.conf import settings
from django.core.management import call_command
from django.db.models import Sum
from django.utils import timezone as dj_tz

from core.exchange_runtime import get_runtime_exchange_context
from execution.models import BalanceSnapshot, OperationReport, Order, Position
from signals.models import Signal

from .notifications import send_telegram
from .report_controls import resolve_report_config


def _to_float(val) -> float:
    try:
        return float(val)
    except Exception:
        return 0.0


def _module_count(signals_qs, prefix: str) -> int:
    return int(signals_qs.filter(strategy__startswith=prefix).count())


def _count_outcomes(ops_qs, outcome: str) -> int:
    return int(ops_qs.filter(outcome=outcome).count())


def _redis_client():
    try:
        return redis.from_url(settings.CELERY_BROKER_URL)
    except Exception:
        return None


def _decode_redis(raw: Any) -> str:
    if raw is None:
        return ""
    if isinstance(raw, bytes):
        try:
            return raw.decode("utf-8")
        except Exception:
            return str(raw)
    return str(raw)


def _is_report_due(config: dict[str, Any], now) -> tuple[bool, str, str]:
    if not bool(config.get("beat_enabled", True)):
        return False, "beat_disabled", ""

    mode = str(config.get("mode", "interval")).strip().lower()
    if mode == "daily":
        hour = max(0, min(23, int(config.get("beat_hour", 0) or 0)))
        minute = max(0, min(59, int(config.get("beat_minute", 0) or 0)))
        if now.hour != hour or now.minute != minute:
            return False, f"waiting_daily_{hour:02d}:{minute:02d}", ""
        return True, "daily", now.strftime("%Y%m%d")

    interval_min = max(1, min(1440, int(config.get("beat_minutes", 15) or 15)))
    minute_slot = int(now.timestamp() // 60)
    if minute_slot % interval_min != 0:
        return False, f"waiting_interval_{interval_min}m", ""
    return True, "interval", str(minute_slot)


def _already_sent_in_slot(mode: str, slot_id: str) -> bool:
    if not slot_id:
        return False
    client = _redis_client()
    if client is None:
        return False
    key = f"risk:performance_report:last_slot:{mode}"
    try:
        return _decode_redis(client.get(key)) == slot_id
    except Exception:
        return False


def _mark_sent_slot(mode: str, slot_id: str) -> None:
    if not slot_id:
        return
    client = _redis_client()
    if client is None:
        return
    key = f"risk:performance_report:last_slot:{mode}"
    try:
        client.set(key, slot_id, ex=172800)  # 48h TTL
    except Exception:
        return


def _build_performance_report(window_minutes: int) -> str:
    now = dj_tz.now()
    since = now - timedelta(minutes=max(1, int(window_minutes)))

    runtime = get_runtime_exchange_context()
    service = str(runtime.get("service") or "unknown").upper()
    env = "DEMO" if bool(runtime.get("sandbox")) else "LIVE"
    asset = str(runtime.get("primary_asset") or "USDT")

    signals_qs = Signal.objects.filter(ts__gte=since)
    alloc_long = int(signals_qs.filter(strategy="alloc_long").count())
    alloc_short = int(signals_qs.filter(strategy="alloc_short").count())
    alloc_flat = int(signals_qs.filter(strategy="alloc_flat").count())

    mod_trend = _module_count(signals_qs, "mod_trend_")
    mod_meanrev = _module_count(signals_qs, "mod_meanrev_")
    mod_carry = _module_count(signals_qs, "mod_carry_")
    mod_grid = _module_count(signals_qs, "mod_grid_")
    smc = _module_count(signals_qs, "smc_")

    orders_filled = int(
        Order.objects.filter(opened_at__gte=since, status=Order.OrderStatus.FILLED).count()
    )
    orders_rejected = int(
        Order.objects.filter(opened_at__gte=since, status=Order.OrderStatus.REJECTED).count()
    )

    ops_qs = OperationReport.objects.filter(closed_at__gte=since)
    ops_count = int(ops_qs.count())
    wins = _count_outcomes(ops_qs, OperationReport.Outcome.WIN)
    losses = _count_outcomes(ops_qs, OperationReport.Outcome.LOSS)
    be = _count_outcomes(ops_qs, OperationReport.Outcome.BE)
    pnl_abs = _to_float(ops_qs.aggregate(total=Sum("pnl_abs")).get("total"))
    win_rate = (wins / (wins + losses) * 100.0) if (wins + losses) > 0 else 0.0

    open_positions_qs = Position.objects.filter(is_open=True)
    open_positions = int(open_positions_qs.count())
    unrealized = _to_float(open_positions_qs.aggregate(total=Sum("unrealized_pnl")).get("total"))

    snap = BalanceSnapshot.objects.order_by("-created_at").first()
    equity = _to_float(getattr(snap, "equity_usdt", 0.0))
    free = _to_float(getattr(snap, "free_usdt", 0.0))
    lev = _to_float(getattr(snap, "eff_leverage", 0.0))

    pnl_icon = "\U0001F7E2" if pnl_abs >= 0 else "\U0001F534"
    unrealized_icon = "\U0001F7E2" if unrealized >= 0 else "\U0001F534"

    lines = [
        f"\U0001F4CA <b>Reporte {window_minutes}m</b>",
        f"<b>Env:</b> {service} {env}",
        f"<b>Ventana:</b> {window_minutes}m",
        "",
        "<b>Senales</b>",
        (
            f"mod trend={mod_trend} | meanrev={mod_meanrev} | carry={mod_carry} | grid={mod_grid} | smc={smc}\n"
            f"alloc long={alloc_long} short={alloc_short} flat={alloc_flat}"
        ),
        "",
        "<b>Ejecucion</b>",
        f"orders filled={orders_filled} rejected={orders_rejected}",
        f"ops cerradas={ops_count} ({wins}W/{losses}L/{be}BE) WR={win_rate:.1f}%",
        f"{pnl_icon} pnl cerrada={pnl_abs:+.4f} {asset}",
        f"{unrealized_icon} pnl abierta={unrealized:+.4f} {asset} | open_pos={open_positions}",
        "",
        f"<b>Cuenta:</b> equity={equity:.2f} {asset} | free={free:.2f} {asset} | lev={lev:.2f}x",
        f"<b>Hora:</b> {now.strftime('%Y-%m-%d %H:%M:%S')} UTC",
    ]
    return "\n".join(lines)


@shared_task
def send_performance_report() -> str:
    config = resolve_report_config()
    if not bool(getattr(settings, "PERFORMANCE_REPORT_ENABLED", True)):
        return "performance_report:disabled"
    if not bool(config.get("enabled", True)):
        return "performance_report:runtime_disabled"
    if not bool(getattr(settings, "TELEGRAM_ENABLED", False)):
        return "performance_report:telegram_disabled"

    now = dj_tz.now()
    due, mode_or_reason, slot_id = _is_report_due(config, now)
    if not due:
        return f"performance_report:skip={mode_or_reason}"
    mode = mode_or_reason
    if _already_sent_in_slot(mode, slot_id):
        return f"performance_report:skip=already_sent:{mode}:{slot_id}"

    window_minutes = max(1, min(1440, int(config.get("window_minutes", 15) or 15)))
    message = _build_performance_report(window_minutes=window_minutes)
    sent = send_telegram(message, parse_mode="HTML")
    if sent:
        _mark_sent_slot(mode, slot_id)
    return f"performance_report:sent={1 if sent else 0}:window={window_minutes}m"


@shared_task
def run_nightly_monte_carlo() -> str:
    """Run nightly Monte Carlo stress report and optionally notify Telegram."""
    if not bool(getattr(settings, "MONTE_CARLO_NIGHTLY_ENABLED", False)):
        return "monte_carlo_nightly:disabled"

    report_dir = Path(settings.BASE_DIR) / "reports" / "monte_carlo"
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = dj_tz.now().strftime("%Y%m%d_%H%M%S")
    out_path = report_dir / f"nightly_{stamp}.json"

    args = {
        "days": int(getattr(settings, "MONTE_CARLO_NIGHTLY_DAYS", 90) or 90),
        "sims": int(getattr(settings, "MONTE_CARLO_NIGHTLY_SIMS", 10000) or 10000),
        "ruin_threshold": float(getattr(settings, "MONTE_CARLO_NIGHTLY_RUIN_THRESHOLD", 20.0) or 20.0),
        "json": str(out_path),
        "regime_aware": bool(getattr(settings, "MONTE_CARLO_NIGHTLY_REGIME_AWARE", True)),
        "stress_profile": str(getattr(settings, "MONTE_CARLO_NIGHTLY_STRESS_PROFILE", "balanced") or "balanced"),
    }
    call_command("monte_carlo", **args)

    if bool(getattr(settings, "TELEGRAM_ENABLED", False)) and bool(
        getattr(settings, "MONTE_CARLO_NIGHTLY_NOTIFY", True)
    ):
        try:
            payload = json.loads(out_path.read_text(encoding="utf-8"))
            mc = payload.get("monte_carlo", {}) if isinstance(payload, dict) else {}
            stress = payload.get("stress", {}) if isinstance(payload, dict) else {}
            ruin = _to_float(mc.get("risk_of_ruin_pct"))
            mean_dd = _to_float(mc.get("mean_max_dd_pct"))
            p95_dd = _to_float((mc.get("max_dd_percentiles") or {}).get("p95"))
            mean_ret = _to_float(mc.get("mean_final_return_pct"))
            profile = str(stress.get("profile") or "none")
            message = (
                "\U0001F30C <b>Monte Carlo Nightly</b>\n"
                f"profile={profile} sims={args['sims']} days={args['days']}\n"
                f"risk_of_ruin={ruin:.2f}% | mean_max_dd={mean_dd:.2f}% | p95_dd={p95_dd:.2f}%\n"
                f"mean_return={mean_ret:+.2f}%\n"
                f"file={out_path.name}"
            )
            send_telegram(message, parse_mode="HTML")
        except Exception:
            pass

    return f"monte_carlo_nightly:ok:{out_path.name}"
