"""
Telegram notification service for critical trading events.
"""
from __future__ import annotations

import logging

import httpx
from django.conf import settings

from core.exchange_runtime import get_runtime_exchange_context

logger = logging.getLogger(__name__)

E_ROCKET = "\U0001F680"
E_TROPHY = "\U0001F3C6"
E_STOP = "\U0001F6D1"
E_WARN = "\u26a0\ufe0f"
E_ERROR = "\U0001F6A8"
E_UP = "\u2b06\ufe0f"
E_DOWN = "\u2b07\ufe0f"
E_GREEN = "\U0001F7E2"
E_RED = "\U0001F534"


_MODULE_TITLES: dict[str, tuple[str, str]] = {
    "allocator": ("Allocator", "Gestor Dinamico de Estrategias"),
    "trend": ("Trend", "Seguidor de Tendencia"),
    "meanrev": ("Mean Reversion", "Reversion a la Media"),
    "carry": ("Carry", "Captura de Funding"),
    "smc": ("SMC", "Smart Money Concepts"),
}


def _runtime_context() -> tuple[str, str]:
    try:
        ctx = get_runtime_exchange_context()
        label = str(ctx.get("label") or "").strip()
        service = str(ctx.get("service") or "unknown").upper()
        env = "DEMO" if bool(ctx.get("sandbox")) else "LIVE"
        asset = str(ctx.get("primary_asset") or "USDT")
        return label or f"{service} {env}", asset
    except Exception:
        return "", "USDT"


def _strategy_module(strategy_name: str) -> str:
    name = str(strategy_name or "").strip().lower()
    if name.startswith("alloc_"):
        return "allocator"
    if name.startswith("smc_"):
        return "smc"
    if name.startswith("mod_"):
        parts = name.split("_")
        if len(parts) >= 3:
            return parts[1]
    return ""


def _module_title(module: str) -> tuple[str, str]:
    key = str(module or "").strip().lower()
    return _MODULE_TITLES.get(key, (key.upper() if key else "N/A", "Estrategia no clasificada"))


def _normalize_active_modules(active_modules: list[str] | None) -> list[str]:
    if not active_modules:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for module in active_modules:
        key = str(module or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _strategy_block(strategy_name: str, active_modules: list[str] | None = None) -> str:
    strategy_txt = str(strategy_name or "").strip()
    module = _strategy_module(strategy_txt)
    tech_type, commercial_name = _module_title(module)

    lines = []
    if strategy_txt:
        lines.append(f"<b>Estrategia (tecnica):</b> {strategy_txt}")
    lines.append(f"<b>Tipo estrategia:</b> {tech_type}")
    lines.append(f"<b>Nombre comercial:</b> {commercial_name}")

    normalized_modules = _normalize_active_modules(active_modules)
    if normalized_modules:
        labels = []
        for mod in normalized_modules:
            mod_tech, mod_name = _module_title(mod)
            labels.append(f"{mod_tech} ({mod_name})")
        lines.append(f"<b>Modulos activos:</b> {', '.join(labels)}")
    return "\n".join(lines)


def send_telegram(message: str, parse_mode: str | None = "HTML") -> bool:
    """Send a Telegram message. Returns True if successful."""
    if not settings.TELEGRAM_ENABLED:
        return False
    token = settings.TELEGRAM_BOT_TOKEN
    chat_id = settings.TELEGRAM_CHAT_ID
    if not token or not chat_id:
        logger.debug("Telegram not configured (missing token or chat_id)")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        payload = {"chat_id": chat_id, "text": message}
        if parse_mode:
            payload["parse_mode"] = parse_mode
        resp = httpx.post(
            url,
            json=payload,
            timeout=10,
        )
        if resp.status_code == 200:
            return True

        # Fallback: retry without parse mode when Telegram rejects HTML entities/tags.
        text_lower = resp.text.lower()
        if parse_mode and resp.status_code == 400 and "can't parse entities" in text_lower:
            logger.warning(
                "Telegram API parse error with parse_mode=%s; retrying without parse_mode",
                parse_mode,
            )
            fallback_resp = httpx.post(
                url,
                json={"chat_id": chat_id, "text": message},
                timeout=10,
            )
            if fallback_resp.status_code == 200:
                return True
            logger.warning(
                "Telegram API error %s after fallback: %s",
                fallback_resp.status_code,
                fallback_resp.text[:200],
            )
            return False

        logger.warning("Telegram API error %s: %s", resp.status_code, resp.text[:200])
        return False
    except Exception as exc:
        logger.warning("Telegram send failed: %s", exc)
        return False


def notify_kill_switch(reason: str, details: str = ""):
    """Alert: kill-switch activated."""
    env_label, _asset = _runtime_context()
    msg = f"{E_STOP} <b>KILL-SWITCH ACTIVATED</b>\n<b>Reason:</b> {reason}"
    if details:
        msg += f"\n<b>Details:</b> {details}"
    if env_label:
        msg += f"\n<b>Env:</b> {env_label}"
    send_telegram(msg)


def notify_trade_opened(
    symbol: str,
    side: str,
    qty: float,
    price: float,
    sl: float | None = None,
    tp: float | None = None,
    leverage: float = 0,
    notional: float = 0,
    equity: float = 0,
    risk_pct: float = 0,
    score: float = 0,
    strategy_name: str = "",
    active_modules: list[str] | None = None,
    entry_reason: str = "",
):
    """Alert: new trade opened."""
    env_label, asset = _runtime_context()
    direction = "LONG" if side == "buy" else "SHORT"
    side_icon = E_UP if side == "buy" else E_DOWN
    msg = (
        f"{E_ROCKET} <b>{direction} Opened</b> {side_icon}\n"
        f"<b>Symbol:</b> {symbol}\n"
        f"<b>Qty:</b> {qty} @ {price:.4f}\n"
        f"<b>Notional:</b> {notional:.2f} {asset}"
    )
    if leverage:
        msg += f" ({leverage:.0f}x)"
    if equity:
        msg += f"\n<b>Equity:</b> {equity:.2f} {asset}"
    if risk_pct:
        msg += f" | <b>Risk:</b> {risk_pct:.1%}"
    if sl is not None:
        sl_dist = abs(sl - price) / price if price else 0
        msg += f"\n<b>SL:</b> {sl:.4f} ({sl_dist:.2%})"
    if tp is not None:
        tp_dist = abs(tp - price) / price if price else 0
        msg += f"\n<b>TP:</b> {tp:.4f} ({tp_dist:.2%})"
    if score:
        msg += f"\n<b>Score:</b> {score:.2f}"
    strategy_txt = _strategy_block(strategy_name, active_modules)
    if strategy_txt:
        msg += f"\n{strategy_txt}"
    entry_reason_txt = str(entry_reason or "").strip()
    if entry_reason_txt:
        msg += f"\n<b>Razon de posicion:</b> {entry_reason_txt}"
    if env_label:
        msg += f"\n<b>Env:</b> {env_label}"
    send_telegram(msg)


def notify_trade_closed(
    symbol: str,
    reason: str,
    pnl_pct: float,
    pnl_abs: float = 0,
    entry_price: float = 0,
    exit_price: float = 0,
    qty: float = 0,
    equity_before: float = 0,
    duration_min: float = 0,
    side: str = "",
    leverage: float = 0,
    strategy_name: str = "",
    active_modules: list[str] | None = None,
):
    """Alert: trade closed (TP/SL/signal_flip)."""
    env_label, asset = _runtime_context()
    direction = "LONG" if side == "buy" else "SHORT" if side == "sell" else ""
    result = "WIN" if pnl_pct >= 0 else "LOSS"
    result_icon = E_TROPHY if pnl_pct >= 0 else E_RED
    reason_labels = {
        "tp": "Take Profit",
        "sl": "Stop Loss",
        "signal_flip": "Signal Flip",
        "trailing_stop": "Trailing Stop",
        "exchange_close": "Exchange Close (SL/TP)",
    }
    reason_display = reason_labels.get(reason, reason)

    msg = f"{result_icon} <b>{direction} Closed</b> [{result}] - {reason_display}\n"
    msg += f"<b>Symbol:</b> {symbol}\n"
    if entry_price and exit_price:
        msg += f"<b>Entry:</b> {entry_price:.4f} -> <b>Exit:</b> {exit_price:.4f}\n"
    if qty:
        msg += f"<b>Qty:</b> {qty}"
        if leverage:
            msg += f" ({leverage:.0f}x)"
        msg += "\n"
    msg += f"<b>PnL:</b> {pnl_pct:+.2%}"
    if pnl_abs:
        msg += f" | <b>{pnl_abs:+.4f} {asset}</b>"
    if equity_before and pnl_abs:
        equity_pct = pnl_abs / equity_before
        msg += f" | {equity_pct:+.2%} equity"
    if duration_min > 0:
        if duration_min >= 60:
            hours = int(duration_min // 60)
            mins = int(duration_min % 60)
            msg += f"\n<b>Duration:</b> {hours}h {mins}m"
        else:
            msg += f"\n<b>Duration:</b> {int(duration_min)}m"
    if equity_before:
        new_equity = equity_before + pnl_abs
        msg += f"\n<b>Equity:</b> {equity_before:.2f} {asset} -> {new_equity:.2f} {asset}"
    strategy_txt = _strategy_block(strategy_name, active_modules)
    if strategy_txt:
        msg += f"\n{strategy_txt}"
    if env_label:
        msg += f"\n<b>Env:</b> {env_label}"
    send_telegram(msg)


def notify_risk_event(kind: str, severity: str, details: str = ""):
    """Alert: risk event detected."""
    env_label, _asset = _runtime_context()
    severity_icon = E_RED if severity.lower() == "critical" else E_WARN
    msg = f"{severity_icon} <b>RISK EVENT</b>\n<b>Type:</b> {kind}\n<b>Severity:</b> {severity}"
    if details:
        msg += f"\n{details}"
    if env_label:
        msg += f"\n<b>Env:</b> {env_label}"
    send_telegram(msg)


def notify_error(context: str, error: str = ""):
    """Alert: system error."""
    env_label, _asset = _runtime_context()
    msg = f"{E_ERROR} <b>ERROR</b>\n<b>Context:</b> {context}"
    if error:
        msg += f"\n<b>Error:</b> {error[:500]}"
    if env_label:
        msg += f"\n<b>Env:</b> {env_label}"
    send_telegram(msg)
