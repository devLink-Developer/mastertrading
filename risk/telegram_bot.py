"""
Interactive Telegram bot for MasterTrading.
Listens for commands and shows an interactive menu with inline buttons.

Run as a standalone process: python manage.py telegram_bot
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

from decimal import Decimal

from django.conf import settings
from django.utils import timezone as dj_tz

from adapters import get_default_adapter
from core.exchange_runtime import extract_balance_values, get_runtime_exchange_context
from risk.report_controls import resolve_report_config, update_report_config
from signals.feature_flags import (
    FEATURE_FLAGS_VERSION,
    FEATURE_KEYS,
    feature_flag_defaults,
    resolve_runtime_flags,
)
from signals.models import StrategyConfig
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    MessageHandler,
    filters,
)

logger = logging.getLogger(__name__)

E_BOT = "\U0001F916"
E_POS = "\U0001F4CA"
E_BAL = "\U0001F4B0"
E_SIG = "\U0001F4C8"
E_OPS = "\U0001F4CB"
E_STATUS = "\u2699\ufe0f"
E_KS = "\U0001F6D1"
E_STRAT = "\U0001F9E9"
E_REPORT = "\U0001F4CC"
E_BACK = "\u2b05\ufe0f"
E_GREEN = "\U0001F7E2"
E_RED = "\U0001F534"
E_YELLOW = "\U0001F7E1"
E_UP = "\u2b06\ufe0f"
E_DOWN = "\u2b07\ufe0f"
E_CHECK = "\u2705"
E_CROSS = "\u274c"
E_NEUTRAL = "\u27b0"

REPORT_INTERVAL_PRESETS = [5, 10, 15, 30, 60, 120]
REPORT_WINDOW_PRESETS = [15, 30, 60, 120, 240, 480]
REPORT_DAILY_PRESETS = [(0, 0), (8, 0), (12, 0), (20, 0)]

MAIN_MENU_KEYBOARD = InlineKeyboardMarkup(
    [
        [
            InlineKeyboardButton(f"{E_POS} Posiciones", callback_data="positions"),
            InlineKeyboardButton(f"{E_BAL} Balance", callback_data="balance"),
        ],
        [
            InlineKeyboardButton(f"{E_SIG} Senales Hoy", callback_data="signals"),
            InlineKeyboardButton(f"{E_OPS} Operaciones Hoy", callback_data="operations"),
        ],
        [
            InlineKeyboardButton(f"{E_STATUS} Estado Bot", callback_data="status"),
            InlineKeyboardButton(f"{E_KS} Kill Switch", callback_data="killswitch"),
        ],
        [
            InlineKeyboardButton(f"{E_STRAT} Estrategias", callback_data="features"),
            InlineKeyboardButton(f"{E_REPORT} Reportes", callback_data="reports"),
        ],
    ]
)

FEATURE_TOGGLES = [
    ("multi", FEATURE_KEYS["multi"], "Multi"),
    ("trend", FEATURE_KEYS["trend"], "Trend"),
    ("meanrev", FEATURE_KEYS["meanrev"], "MeanRev"),
    ("carry", FEATURE_KEYS["carry"], "Carry"),
    ("allocator", FEATURE_KEYS["allocator"], "Allocator"),
]
FEATURE_TOGGLE_MAP = {alias: (name, label) for alias, name, label in FEATURE_TOGGLES}


def _features_keyboard(flags: dict[str, bool]) -> InlineKeyboardMarkup:
    buttons = []
    for alias, name, label in FEATURE_TOGGLES:
        enabled = bool(flags.get(name, False))
        icon = E_GREEN if enabled else E_RED
        buttons.append(
            InlineKeyboardButton(
                f"{icon} {label}",
                callback_data=f"feature_toggle:{alias}",
            )
        )
    rows = [buttons[i : i + 2] for i in range(0, len(buttons), 2)]
    rows.append([InlineKeyboardButton("\U0001F504 Refrescar", callback_data="feature_refresh")])
    rows.append([InlineKeyboardButton(f"{E_BACK} Volver al Menu", callback_data="back")])
    return InlineKeyboardMarkup(rows)


def _toggle_feature(alias: str) -> tuple[bool, str]:
    feature = FEATURE_TOGGLE_MAP.get(alias)
    if not feature:
        return False, "Feature no reconocida."
    feature_name, label = feature
    defaults = feature_flag_defaults()
    row, _ = StrategyConfig.objects.get_or_create(
        name=feature_name,
        version=FEATURE_FLAGS_VERSION,
        defaults={
            "enabled": bool(defaults.get(feature_name, False)),
            "params_json": {"feature_flag": True},
        },
    )
    row.enabled = not bool(row.enabled)
    params = row.params_json or {}
    params["feature_flag"] = True
    row.params_json = params
    row.save(update_fields=["enabled", "params_json"])
    return True, f"{label} {'ON' if row.enabled else 'OFF'}"


async def _render_features_message(extra: str = "") -> tuple[str, InlineKeyboardMarkup]:
    flags = await asyncio.to_thread(resolve_runtime_flags)
    lines = [
        f"{E_STRAT} <b>Estrategias (Feature Flags)</b>",
        "",
        "Cambios en vivo (sin reinicio):",
    ]
    for _alias, name, label in FEATURE_TOGGLES:
        enabled = bool(flags.get(name, False))
        icon = E_GREEN if enabled else E_RED
        lines.append(f"{icon} <b>{label}</b>: {'ON' if enabled else 'OFF'}")
    if extra:
        lines.extend(["", f"{E_CHECK} {extra}"])
    return "\n".join(lines), _features_keyboard(flags)


def _report_choice_label(enabled: bool, label: str) -> str:
    marker = E_CHECK if enabled else E_NEUTRAL
    return f"{marker} {label}"


def _reports_keyboard(cfg: dict[str, Any]) -> InlineKeyboardMarkup:
    mode = str(cfg.get("mode", "interval")).strip().lower()
    beat_minutes = int(cfg.get("beat_minutes", 15) or 15)
    beat_hour = int(cfg.get("beat_hour", 0) or 0)
    beat_minute = int(cfg.get("beat_minute", 0) or 0)
    window_minutes = int(cfg.get("window_minutes", 15) or 15)
    enabled = bool(cfg.get("enabled", True))
    beat_enabled = bool(cfg.get("beat_enabled", True))

    rows: list[list[InlineKeyboardButton]] = [
        [
            InlineKeyboardButton(
                f"{E_GREEN if enabled else E_RED} Reporte {'ON' if enabled else 'OFF'}",
                callback_data="report_toggle_enabled",
            ),
            InlineKeyboardButton(
                f"{E_GREEN if beat_enabled else E_RED} Scheduler {'ON' if beat_enabled else 'OFF'}",
                callback_data="report_toggle_beat",
            ),
        ],
        [
            InlineKeyboardButton(
                _report_choice_label(mode == "interval", "Modo Intervalo"),
                callback_data="report_mode:interval",
            ),
            InlineKeyboardButton(
                _report_choice_label(mode == "daily", "Modo Diario"),
                callback_data="report_mode:daily",
            ),
        ],
    ]

    interval_buttons = [
        InlineKeyboardButton(
            _report_choice_label(mode == "interval" and beat_minutes == minutes, f"{minutes}m"),
            callback_data=f"report_interval:{minutes}",
        )
        for minutes in REPORT_INTERVAL_PRESETS
    ]
    rows.append(interval_buttons[:3])
    rows.append(interval_buttons[3:])

    daily_buttons = [
        InlineKeyboardButton(
            _report_choice_label(
                mode == "daily" and beat_hour == hour and beat_minute == minute,
                f"{hour:02d}:{minute:02d}",
            ),
            callback_data=f"report_time:{hour:02d}{minute:02d}",
        )
        for hour, minute in REPORT_DAILY_PRESETS
    ]
    rows.append(daily_buttons[:2])
    rows.append(daily_buttons[2:])

    window_buttons = [
        InlineKeyboardButton(
            _report_choice_label(window_minutes == minutes, f"W {minutes}m"),
            callback_data=f"report_window:{minutes}",
        )
        for minutes in REPORT_WINDOW_PRESETS
    ]
    rows.append(window_buttons[:3])
    rows.append(window_buttons[3:])

    rows.append([InlineKeyboardButton("\U0001F504 Refrescar", callback_data="reports_refresh")])
    rows.append([InlineKeyboardButton(f"{E_BACK} Volver al Menu", callback_data="back")])
    return InlineKeyboardMarkup(rows)


async def _render_reports_message(
    extra: str = "",
    cfg: dict[str, Any] | None = None,
) -> tuple[str, InlineKeyboardMarkup]:
    if cfg is None:
        cfg = await asyncio.to_thread(resolve_report_config)

    enabled = bool(cfg.get("enabled", True))
    beat_enabled = bool(cfg.get("beat_enabled", True))
    mode = str(cfg.get("mode", "interval")).strip().lower()
    if mode not in {"interval", "daily"}:
        mode = "interval"
    beat_minutes = int(cfg.get("beat_minutes", 15) or 15)
    beat_hour = int(cfg.get("beat_hour", 0) or 0)
    beat_minute = int(cfg.get("beat_minute", 0) or 0)
    window_minutes = int(cfg.get("window_minutes", 15) or 15)

    lines = [
        f"{E_REPORT} <b>Reportes de Performance</b>",
        "",
        f"{E_GREEN if enabled else E_RED} <b>Reporte:</b> {'ON' if enabled else 'OFF'}",
        f"{E_GREEN if beat_enabled else E_RED} <b>Scheduler:</b> {'ON' if beat_enabled else 'OFF'}",
        f"\u23f1\ufe0f <b>Modo:</b> {'Intervalo' if mode == 'interval' else 'Diario'}",
    ]
    if mode == "interval":
        lines.append(f"\u23f3 <b>Cada:</b> {beat_minutes} minutos")
    else:
        lines.append(f"\U0001F551 <b>Hora:</b> {beat_hour:02d}:{beat_minute:02d} UTC")
    lines.extend(
        [
            f"\U0001F50D <b>Ventana:</b> {window_minutes} minutos",
            "",
            "Cambios en vivo (sin reinicio).",
        ]
    )

    if not bool(getattr(settings, "PERFORMANCE_REPORT_ENABLED", True)):
        lines.extend(
            [
                "",
                f"{E_YELLOW} Bloqueado por env: PERFORMANCE_REPORT_ENABLED=false",
            ]
        )
    if extra:
        lines.extend(["", f"{E_CHECK} {extra}"])
    return "\n".join(lines), _reports_keyboard(cfg)


def _authorized(update: Update) -> bool:
    """Only allow the configured chat_id to interact."""
    chat_id = str(update.effective_chat.id)
    allowed = str(settings.TELEGRAM_CHAT_ID)
    return chat_id == allowed


async def cmd_start(update: Update, context):
    if not _authorized(update):
        return
    await update.message.reply_text(
        f"{E_BOT} <b>MasterTrading Bot</b>\n\nElige una opcion:",
        parse_mode="HTML",
        reply_markup=MAIN_MENU_KEYBOARD,
    )


async def cmd_hola(update: Update, context):
    """Respond to common greeting/menu messages with the menu."""
    if not _authorized(update):
        return
    text = (update.message.text or "").strip().lower()
    if text in ("hola", "hi", "hello", "menu"):
        await update.message.reply_text(
            f"{E_BOT} <b>MasterTrading Bot</b>\n\nElige una opcion:",
            parse_mode="HTML",
            reply_markup=MAIN_MENU_KEYBOARD,
        )


async def cmd_status(update: Update, context):
    if not _authorized(update):
        return
    msg = await _get_status_text()
    await update.message.reply_text(msg, parse_mode="HTML", reply_markup=MAIN_MENU_KEYBOARD)


async def button_handler(update: Update, context):
    """Handle inline button presses."""
    query = update.callback_query
    if not _authorized(update):
        await query.answer("No autorizado")
        return

    await query.answer()
    data = query.data

    if data == "positions":
        msg = await _get_positions_text()
    elif data == "balance":
        msg = await _get_balance_text()
    elif data == "signals":
        msg = await _get_signals_text()
    elif data == "operations":
        msg = await _get_operations_text()
    elif data == "status":
        msg = await _get_status_text()
    elif data == "killswitch":
        msg = await _get_killswitch_text()
    elif data == "features":
        msg, kb = await _render_features_message()
        await query.edit_message_text(msg, parse_mode="HTML", reply_markup=kb)
        return
    elif data == "feature_refresh":
        msg, kb = await _render_features_message()
        await query.edit_message_text(msg, parse_mode="HTML", reply_markup=kb)
        return
    elif data.startswith("feature_toggle:"):
        alias = data.split(":", 1)[1].strip().lower()
        ok, result = await asyncio.to_thread(_toggle_feature, alias)
        await query.answer(result)
        msg, kb = await _render_features_message(result if ok else f"{E_YELLOW} {result}")
        await query.edit_message_text(msg, parse_mode="HTML", reply_markup=kb)
        return
    elif data == "reports":
        msg, kb = await _render_reports_message()
        await query.edit_message_text(msg, parse_mode="HTML", reply_markup=kb)
        return
    elif data == "reports_refresh":
        msg, kb = await _render_reports_message()
        await query.edit_message_text(msg, parse_mode="HTML", reply_markup=kb)
        return
    elif data == "report_toggle_enabled":
        current = await asyncio.to_thread(resolve_report_config)
        next_value = not bool(current.get("enabled", True))
        cfg = await asyncio.to_thread(update_report_config, enabled=next_value)
        msg, kb = await _render_reports_message(
            f"Reporte {'ON' if cfg.get('enabled') else 'OFF'}",
            cfg,
        )
        await query.edit_message_text(msg, parse_mode="HTML", reply_markup=kb)
        return
    elif data == "report_toggle_beat":
        current = await asyncio.to_thread(resolve_report_config)
        next_value = not bool(current.get("beat_enabled", True))
        cfg = await asyncio.to_thread(update_report_config, beat_enabled=next_value)
        msg, kb = await _render_reports_message(
            f"Scheduler {'ON' if cfg.get('beat_enabled') else 'OFF'}",
            cfg,
        )
        await query.edit_message_text(msg, parse_mode="HTML", reply_markup=kb)
        return
    elif data.startswith("report_mode:"):
        mode = data.split(":", 1)[1].strip().lower()
        if mode not in {"interval", "daily"}:
            msg, kb = await _render_reports_message(f"{E_YELLOW} Modo invalido")
            await query.edit_message_text(msg, parse_mode="HTML", reply_markup=kb)
            return
        cfg = await asyncio.to_thread(update_report_config, mode=mode)
        msg, kb = await _render_reports_message(f"Modo {mode.upper()} aplicado", cfg)
        await query.edit_message_text(msg, parse_mode="HTML", reply_markup=kb)
        return
    elif data.startswith("report_interval:"):
        raw = data.split(":", 1)[1].strip()
        try:
            minutes = int(raw)
        except Exception:
            minutes = 15
        cfg = await asyncio.to_thread(update_report_config, mode="interval", beat_minutes=minutes)
        msg, kb = await _render_reports_message(f"Intervalo cada {cfg.get('beat_minutes')}m", cfg)
        await query.edit_message_text(msg, parse_mode="HTML", reply_markup=kb)
        return
    elif data.startswith("report_time:"):
        raw = data.split(":", 1)[1].strip()
        if len(raw) != 4 or not raw.isdigit():
            msg, kb = await _render_reports_message(f"{E_YELLOW} Hora invalida")
            await query.edit_message_text(msg, parse_mode="HTML", reply_markup=kb)
            return
        hour = int(raw[:2])
        minute = int(raw[2:])
        cfg = await asyncio.to_thread(
            update_report_config,
            mode="daily",
            beat_hour=hour,
            beat_minute=minute,
        )
        msg, kb = await _render_reports_message(
            f"Hora diaria {int(cfg.get('beat_hour', 0)):02d}:{int(cfg.get('beat_minute', 0)):02d} UTC",
            cfg,
        )
        await query.edit_message_text(msg, parse_mode="HTML", reply_markup=kb)
        return
    elif data.startswith("report_window:"):
        raw = data.split(":", 1)[1].strip()
        try:
            minutes = int(raw)
        except Exception:
            minutes = 15
        cfg = await asyncio.to_thread(update_report_config, window_minutes=minutes)
        msg, kb = await _render_reports_message(f"Ventana {cfg.get('window_minutes')}m", cfg)
        await query.edit_message_text(msg, parse_mode="HTML", reply_markup=kb)
        return
    elif data == "back":
        await query.edit_message_text(
            f"{E_BOT} <b>MasterTrading Bot</b>\n\nElige una opcion:",
            parse_mode="HTML",
            reply_markup=MAIN_MENU_KEYBOARD,
        )
        return
    else:
        msg = f"{E_YELLOW} Opcion no reconocida."

    back_kb = InlineKeyboardMarkup(
        [[InlineKeyboardButton(f"{E_BACK} Volver al Menu", callback_data="back")]]
    )
    await query.edit_message_text(msg, parse_mode="HTML", reply_markup=back_kb)


async def _get_positions_text() -> str:
    from execution.models import Position

    positions = await asyncio.to_thread(
        lambda: list(Position.objects.filter(is_open=True).select_related("instrument"))
    )
    if not positions:
        return f"{E_POS} <b>Posiciones Abiertas</b>\n\n{E_CHECK} No hay posiciones abiertas."

    lines = [f"{E_POS} <b>Posiciones Abiertas</b>\n"]
    total_pnl = Decimal("0")
    for p in positions:
        side_str = "LONG" if p.side == "long" else "SHORT"
        state = "GANANDO" if p.unrealized_pnl >= 0 else "PERDIENDO"
        state_icon = E_GREEN if p.unrealized_pnl >= 0 else E_RED
        side_icon = E_UP if side_str == "LONG" else E_DOWN
        lines.append(
            f"{state_icon} <b>{p.instrument.symbol}</b> {side_icon} {side_str} [{state}]\n"
            f"Qty: {p.qty} | Entry: {p.avg_price:.4f}\n"
            f"Last: {p.last_price:.4f} | PnL: {float(p.pnl_pct):+.2%} ({float(p.unrealized_pnl):+.4f} USDT)"
        )
        total_pnl += p.unrealized_pnl
    total_icon = E_GREEN if total_pnl >= 0 else E_RED
    lines.append(f"\n{total_icon} <b>PnL Total:</b> {float(total_pnl):+.4f} USDT")
    return "\n".join(lines)


async def _get_balance_text() -> str:
    from execution.models import BalanceSnapshot

    ctx = await asyncio.to_thread(get_runtime_exchange_context)
    snap = await asyncio.to_thread(lambda: BalanceSnapshot.objects.order_by("-created_at").first())

    live_free = None
    live_equity = None
    live_asset = str(ctx.get("primary_asset") or "USDT")
    live_error = ""
    try:
        balance = await asyncio.to_thread(lambda: get_default_adapter().fetch_balance())
        free_val, equity_val, asset = extract_balance_values(
            balance,
            list(ctx.get("balance_assets") or ["USDT"]),
        )
        live_free = free_val
        live_equity = equity_val
        live_asset = asset
    except Exception as exc:
        live_error = str(exc)

    service = str(ctx.get("service") or "unknown").upper()
    env = "DEMO" if bool(ctx.get("sandbox")) else "LIVE"
    env_icon = "\U0001F9EA" if env == "DEMO" else "\U0001F7E2"

    lines = [
        f"{E_BAL} <b>Balance</b>",
        "",
        f"\U0001F3E6 <b>Servicio:</b> {service}",
        f"{env_icon} <b>Entorno:</b> {env}",
    ]

    if live_equity is not None and live_free is not None:
        lines.extend(
            [
                f"<b>Activo Cuenta:</b> {live_asset}",
                f"\U0001F4BC <b>Equity (live):</b> {live_equity:.2f} {live_asset}",
                f"\U0001F9FE <b>Libre (live):</b> {live_free:.2f} {live_asset}",
            ]
        )
    elif live_error:
        lines.append(f"<b>Balance live:</b> no disponible ({live_error[:140]})")

    if snap:
        lines.extend(
            [
                f"<b>Notional (snapshot):</b> ${float(snap.notional_usdt):.2f}",
                f"\U0001F4A1 <b>Leverage Ef.:</b> {float(snap.eff_leverage):.2f}x",
                f"\U0001F551 <b>Snapshot:</b> {snap.created_at.strftime('%H:%M:%S')}",
            ]
        )
    else:
        lines.append("<b>Snapshot:</b> no disponible")

    return "\n".join(lines)


async def _get_signals_text() -> str:
    from signals.models import Signal

    today = dj_tz.now().date()
    signals = await asyncio.to_thread(
        lambda: list(
            Signal.objects.filter(ts__date=today)
            .select_related("instrument")
            .order_by("-ts")[:10]
        )
    )
    if not signals:
        return f"{E_SIG} <b>Senales Hoy</b>\n\n{E_CHECK} No se emitieron senales hoy."

    lines = [f"{E_SIG} <b>Senales Hoy</b>\n"]
    for s in signals:
        direction = "LONG" if "long" in s.strategy else "SHORT"
        dir_icon = E_UP if direction == "LONG" else E_DOWN
        payload = s.payload_json or {}
        reason = payload.get("reason", {})
        htf = reason.get("htf_trend", "?")
        lines.append(
            f"{dir_icon} <b>{s.instrument.symbol}</b> {direction} score={s.score:.2f} HTF={htf}\n"
            f"{s.ts.strftime('%H:%M')}"
        )
    return "\n".join(lines)


async def _get_operations_text() -> str:
    from execution.models import OperationReport

    today = dj_tz.now().date()
    ops = await asyncio.to_thread(
        lambda: list(
            OperationReport.objects.filter(closed_at__date=today)
            .select_related("instrument")
            .order_by("-closed_at")[:10]
        )
    )
    if not ops:
        return f"{E_OPS} <b>Operaciones Hoy</b>\n\n{E_CHECK} No hay operaciones cerradas hoy."

    lines = [f"{E_OPS} <b>Operaciones Hoy</b>\n"]
    total_pnl = 0.0
    wins = losses = 0
    for o in ops:
        side_str = "LONG" if o.side == "buy" else "SHORT"
        side_icon = E_UP if side_str == "LONG" else E_DOWN
        pnl = float(o.pnl_abs)
        total_pnl += pnl
        if o.outcome == "win":
            wins += 1
            mark = f"{E_CHECK} WIN"
        elif o.outcome == "loss":
            losses += 1
            mark = f"{E_CROSS} LOSS"
        else:
            mark = f"{E_NEUTRAL} BE"
        lines.append(
            f"[{mark}] <b>{o.instrument.symbol}</b> {side_icon} {side_str} -> {o.reason}\n"
            f"PnL: {float(o.pnl_pct):+.2%} ({pnl:+.4f} USDT)\n"
            f"Entry: {float(o.entry_price):.4f} -> Exit: {float(o.exit_price):.4f}"
        )
    total_ops = wins + losses
    wr = (wins / total_ops * 100) if total_ops > 0 else 0
    total_icon = E_GREEN if total_pnl >= 0 else E_RED
    lines.append(
        f"\n\U0001F4CC <b>Resumen:</b> {wins}W / {losses}L ({wr:.0f}% WR)\n"
        f"{total_icon} <b>PnL Total:</b> {total_pnl:+.4f} USDT"
    )
    return "\n".join(lines)


async def _get_status_text() -> str:
    from execution.models import BalanceSnapshot, Position
    from signals.models import Signal
    import redis as _redis

    ctx = await asyncio.to_thread(get_runtime_exchange_context)
    risk_ns = str(ctx.get("risk_namespace") or "global")
    primary_asset = str(ctx.get("primary_asset") or "USDT")

    snap = await asyncio.to_thread(lambda: BalanceSnapshot.objects.order_by("-created_at").first())
    open_count = await asyncio.to_thread(lambda: Position.objects.filter(is_open=True).count())
    today = dj_tz.now().date()
    signals_today = await asyncio.to_thread(lambda: Signal.objects.filter(ts__date=today).count())
    current_equity = float(snap.equity_usdt) if snap else None
    equity_asset = primary_asset
    try:
        balance = await asyncio.to_thread(lambda: get_default_adapter().fetch_balance())
        _free_live, equity_live, asset_live = extract_balance_values(
            balance,
            list(ctx.get("balance_assets") or ["USDT"]),
        )
        if equity_live > 0:
            current_equity = equity_live
            equity_asset = asset_live
    except Exception:
        pass

    dd_info = "N/A"
    try:
        client = _redis.from_url(settings.CELERY_BROKER_URL)
        start_raw = client.get(f"risk:equity_start:{risk_ns}:{today.isoformat()}")
        if start_raw and current_equity is not None:
            start_val = float(start_raw)
            dd = (current_equity - start_val) / start_val if start_val else 0
            dd_info = f"{dd:+.2%}"
    except Exception:
        pass

    cb_status = "N/A"
    try:
        from risk.models import CircuitBreakerConfig

        cb = await asyncio.to_thread(CircuitBreakerConfig.get)
        if cb.is_tripped:
            cb_status = f"TRIPPED: {cb.trip_reason}"
        else:
            cb_status = "Normal"
    except Exception:
        pass

    equity_str = f"{current_equity:.2f} {equity_asset}" if current_equity is not None else "N/A"
    mode = settings.MODE
    trading = "ON" if settings.TRADING_ENABLED else "OFF"
    trading_icon = E_GREEN if settings.TRADING_ENABLED else E_RED
    service = str(ctx.get("service") or "unknown").upper()
    env = "DEMO" if bool(ctx.get("sandbox")) else "LIVE"
    env_icon = "\U0001F9EA" if env == "DEMO" else "\U0001F7E2"

    return (
        f"{E_STATUS} <b>Estado del Bot</b>\n\n"
        f"\U0001F3E6 <b>Servicio:</b> {service}\n"
        f"{env_icon} <b>Entorno:</b> {env}\n"
        f"\U0001F4BB <b>Modo:</b> {mode}\n"
        f"{trading_icon} <b>Trading:</b> {trading}\n"
        f"{E_BAL} <b>Equity:</b> {equity_str}\n"
        f"\U0001F4C9 <b>DD Hoy:</b> {dd_info} (limite: -{settings.DAILY_DD_LIMIT:.0%})\n"
        f"{E_POS} <b>Posiciones:</b> {open_count}\n"
        f"{E_SIG} <b>Senales Hoy:</b> {signals_today}\n"
        f"{E_KS} <b>Circuit Breaker:</b> {cb_status}\n"
        f"\U0001F4AA <b>Leverage Max:</b> {settings.MAX_EFF_LEVERAGE}x\n"
        f"\U0001F9EE <b>Risk/Trade:</b> {settings.RISK_PER_TRADE_PCT:.1%}\n"
        f"\U0001F6E1 <b>SL:</b> ATRx{settings.ATR_MULT_SL} (min {getattr(settings, 'MIN_SL_PCT', 0.008):.1%})\n"
        f"\U0001F3AF <b>TP:</b> ATRx{settings.ATR_MULT_TP}"
    )


async def _get_killswitch_text() -> str:
    from risk.models import CircuitBreakerConfig

    try:
        cb = await asyncio.to_thread(CircuitBreakerConfig.get)
    except Exception:
        return f"{E_KS} <b>Kill Switch</b>\n\n{E_CROSS} No se pudo leer el circuit breaker."

    if cb.is_tripped:
        return (
            f"{E_KS} <b>Kill Switch - ACTIVO</b>\n\n"
            f"{E_RED} <b>Razon:</b> {cb.trip_reason}\n"
            f"\U0001F551 <b>Activado:</b> {cb.tripped_at.strftime('%Y-%m-%d %H:%M') if cb.tripped_at else '?'}\n"
            f"\u23f3 <b>Cooldown:</b> {cb.cooldown_minutes_after_trigger} min\n\n"
            "El bot NO abrira trades hasta que expire el cooldown o se resetee manualmente."
        )
    return (
        f"{E_KS} <b>Kill Switch</b>\n\n"
        f"{E_GREEN} <b>Normal</b> - No hay kill switch activo.\n\n"
        "<b>Limites configurados:</b>\n"
        f"DD Diario: {cb.max_daily_dd_pct}%\n"
        f"DD Total: {cb.max_total_dd_pct}%\n"
        f"Perdidas Consecutivas: {cb.max_consecutive_losses}\n"
        f"Peak Equity: ${float(cb.peak_equity):.2f}"
    )


def run_bot():
    token = settings.TELEGRAM_BOT_TOKEN
    if not token:
        logger.error("TELEGRAM_BOT_TOKEN not set, cannot start interactive bot")
        return

    app = ApplicationBuilder().token(token).build()

    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("menu", cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, cmd_hola))
    app.add_handler(CallbackQueryHandler(button_handler))

    logger.info("Telegram interactive bot started (polling)")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    run_bot()
