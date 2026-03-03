from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

import httpx
from django.conf import settings

from core.ai_feedback import load_feedback_context_tail, record_ai_feedback_event
from core.api_runtime import (
    build_optimized_context,
    count_tokens,
    get_active_api_config,
    log_token_usage,
)
from core.models import ApiProviderConfig
from execution.ai_entry_gate import (
    _collect_output_text,
    _compact_payload,
    _dir_code,
    _extract_bool,
    _extract_json_blob,
    _extract_payload_value,
    _http_status_error_detail,
    _preview_text,
    _supports_sampling_controls,
    _to_float,
)

logger = logging.getLogger(__name__)


def _build_responses_body(
    *,
    cfg: ApiProviderConfig,
    system_msg: str,
    user_msg: str,
    reserve_out: int,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": cfg.model_name,
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": system_msg}]},
            {"role": "user", "content": [{"type": "input_text", "text": user_msg}]},
        ],
        "max_output_tokens": reserve_out,
    }
    if str(cfg.provider or "").strip().lower() == "openai":
        body.setdefault("reasoning", {"effort": "minimal"})
        body.setdefault(
            "text",
            {
                "format": {
                    "type": "json_schema",
                    "name": "ai_exit_decision",
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "action": {"type": "string", "enum": ["hold", "close"]},
                            "reason": {"type": "string", "maxLength": 160},
                        },
                        "required": ["action", "reason"],
                    },
                    "strict": True,
                }
            },
        )
    if _supports_sampling_controls(cfg.model_name):
        body["temperature"] = float(cfg.temperature)
        body["top_p"] = float(cfg.top_p)

    if isinstance(cfg.extra_params_json, dict):
        body.update(cfg.extra_params_json)
        if not _supports_sampling_controls(cfg.model_name):
            body.pop("temperature", None)
            body.pop("top_p", None)
    return body


def _resolve_config(
    *,
    account_ai_config_id: int | None,
    account_owner_id: int | None,
) -> ApiProviderConfig | None:
    cfg = None
    if account_ai_config_id:
        cfg = (
            ApiProviderConfig.objects.filter(
                id=account_ai_config_id,
                active=True,
            )
            .order_by("-updated_at")
            .first()
        )
        if cfg:
            return cfg
    provider = str(
        getattr(
            settings,
            "AI_EXIT_GATE_DEFAULT_PROVIDER",
            getattr(settings, "AI_ENTRY_GATE_DEFAULT_PROVIDER", "openai"),
        )
        or "openai"
    ).strip().lower()
    return get_active_api_config(
        provider=provider or None,
        owner_id=account_owner_id,
    )


def _normalize_action(txt: str) -> str:
    action_txt = str(txt or "").strip().lower()
    if action_txt in {"close", "exit", "close_now", "full_close", "take_profit_early"}:
        return "close"
    if action_txt in {"hold", "keep", "wait", "no_close"}:
        return "hold"
    return ""


def _parse_ai_exit_decision(raw_text: str) -> tuple[bool, str]:
    payload = _extract_json_blob(raw_text)
    if payload:
        nested = payload.get("decision")
        if isinstance(nested, dict):
            payload = nested
    if payload:
        action_raw = _extract_payload_value(payload, ("action", "decision"), "")
        close_raw = _extract_payload_value(payload, ("close", "should_close", "exit_now"), None)
        reason_raw = _extract_payload_value(payload, ("reason", "rationale", "note", "why"), "")
        action = _normalize_action(str(action_raw))
        if not action and close_raw is not None:
            action = "close" if _extract_bool(close_raw, default=False) else "hold"
        if not action:
            action = "hold"
        reason = str(reason_raw or "").strip() or ("ai_exit_close" if action == "close" else "ai_exit_hold")
        return action == "close", reason[:160]

    txt = str(raw_text or "")
    m_action = re.search(r"(action|decision)\s*[:=]\s*(close|exit|hold|keep|wait)", txt, flags=re.IGNORECASE)
    m_close = re.search(r"(close|should_close|exit_now)\s*[:=]\s*(true|false|yes|no|1|0)", txt, flags=re.IGNORECASE)
    m_reason = re.search(r"(reason|rationale|note|why)\s*[:=]\s*([^\n\r]+)", txt, flags=re.IGNORECASE)
    if not (m_action or m_close or m_reason):
        return False, "ai_exit_parse_failed"
    if m_action:
        action = _normalize_action(m_action.group(2))
        should_close = action == "close"
    elif m_close:
        should_close = _extract_bool(m_close.group(2), default=False)
    else:
        should_close = False
    reason = str(m_reason.group(2) if m_reason else "").strip() or (
        "ai_exit_close_heuristic" if should_close else "ai_exit_hold_heuristic"
    )
    return bool(should_close), reason[:160]


def _build_gate_messages(
    *,
    user_prompt: str,
    ctx_text: str,
    feedback_text: str,
) -> tuple[str, str]:
    system_msg = (
        "Trading exit gate near TP for open positions. Conservative mode. "
        "Close early only if reversal risk is elevated vs remaining upside. "
        "Return JSON only: {\"action\":\"hold|close\",\"reason\":\"short\"}."
    )
    parts = [f"in={user_prompt}"]
    ctx = str(ctx_text or "").strip()
    fb = str(feedback_text or "").strip()
    if ctx:
        parts.append(f"ctx={ctx}")
    if fb:
        parts.append(f"fb={fb}")
    user_msg = "\n".join(parts)
    return system_msg, user_msg


def evaluate_ai_exit_gate(
    *,
    account_ai_enabled: bool,
    account_ai_config_id: int | None,
    account_owner_id: int | None,
    account_alias: str,
    account_service: str,
    symbol: str,
    strategy_name: str,
    position_direction: str,
    sig_score: float,
    atr: float | None,
    spread_bps: float | None,
    tp_pct: float,
    sl_pct: float,
    pnl_pct_gross: float,
    pnl_pct_gate: float,
    r_multiple: float,
    remaining_tp_pct: float,
    position_age_min: float | None,
    session_name: str,
    sig_payload: dict[str, Any],
) -> tuple[bool, str, dict[str, Any]]:
    meta: dict[str, Any] = {
        "account_alias": account_alias,
        "account_service": account_service,
        "symbol": symbol,
        "strategy": strategy_name,
    }
    if not bool(getattr(settings, "AI_EXIT_GATE_ENABLED", True)):
        return False, "ai_exit_disabled_global", meta
    if not account_ai_enabled:
        return False, "ai_exit_disabled_account", meta
    if bool(getattr(settings, "AI_EXIT_GATE_ONLY_ALLOCATOR", True)):
        if not str(strategy_name or "").strip().lower().startswith("alloc_"):
            return False, "ai_exit_skipped_non_allocator", meta

    cfg = _resolve_config(
        account_ai_config_id=account_ai_config_id,
        account_owner_id=account_owner_id,
    )
    if not cfg:
        record_ai_feedback_event(
            event_type="ai_exit_missing_config",
            level="warning",
            account_alias=account_alias,
            account_service=account_service,
            symbol=symbol,
            strategy=strategy_name,
            allow=None,
            reason="ai_exit_no_config",
            payload={},
        )
        return False, "ai_exit_no_config", meta
    if not str(cfg.api_key or "").strip():
        record_ai_feedback_event(
            event_type="ai_exit_missing_api_key",
            level="warning",
            config=cfg,
            account_alias=account_alias,
            account_service=account_service,
            symbol=symbol,
            strategy=strategy_name,
            allow=None,
            reason="ai_exit_no_api_key",
            payload={"cfg_alias": cfg.name_alias},
        )
        return False, "ai_exit_no_api_key", meta

    started = time.perf_counter()
    candidate = {
        "sym": symbol,
        "st": strategy_name,
        "dir": _dir_code(position_direction),
        "sc": round(_to_float(sig_score, 0.0), 6),
        "atr": None if atr is None else round(_to_float(atr, 0.0), 6),
        "spr": None if spread_bps is None else round(_to_float(spread_bps, 0.0), 3),
        "tp": round(_to_float(tp_pct, 0.0), 6),
        "sl": round(_to_float(sl_pct, 0.0), 6),
        "pnl": round(_to_float(pnl_pct_gross, 0.0), 6),
        "pg": round(_to_float(pnl_pct_gate, 0.0), 6),
        "r": round(_to_float(r_multiple, 0.0), 4),
        "rtp": round(max(0.0, _to_float(remaining_tp_pct, 0.0)), 6),
        "age": None if position_age_min is None else round(max(0.0, _to_float(position_age_min, 0.0)), 2),
        "ses": session_name,
        "sig": _compact_payload(sig_payload),
    }
    user_prompt = json.dumps(candidate, ensure_ascii=True, separators=(",", ":"))

    reserve_out = min(
        int(getattr(settings, "AI_EXIT_GATE_MAX_OUTPUT_TOKENS", 96) or 96),
        int(cfg.max_output_tokens or 96),
    )
    ctx = build_optimized_context(
        cfg,
        user_prompt=user_prompt,
        reserve_output_tokens=reserve_out,
    )
    feedback_budget = max(
        0,
        int(getattr(settings, "AI_FEEDBACK_CONTEXT_MAX_TOKENS", 700) or 700),
    )
    remaining_for_feedback = max(
        0,
        int(cfg.max_input_tokens or 0) - reserve_out - int(ctx.total_prompt_tokens),
    )
    feedback_budget = min(feedback_budget, remaining_for_feedback)
    feedback_text, feedback_tokens, feedback_estimated = load_feedback_context_tail(
        model_name=cfg.model_name,
        max_tokens=feedback_budget,
    )
    system_msg, user_msg = _build_gate_messages(
        user_prompt=user_prompt,
        ctx_text=ctx.context_text,
        feedback_text=feedback_text,
    )
    body = _build_responses_body(
        cfg=cfg,
        system_msg=system_msg,
        user_msg=user_msg,
        reserve_out=reserve_out,
    )
    base_url = (str(cfg.base_url or "").strip() or "https://api.openai.com/v1").rstrip("/")
    url = f"{base_url}/responses"
    headers: dict[str, str] = {
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
    }
    if str(cfg.organization_id or "").strip():
        headers["OpenAI-Organization"] = str(cfg.organization_id).strip()
    if str(cfg.project_id or "").strip():
        headers["OpenAI-Project"] = str(cfg.project_id).strip()
    if isinstance(cfg.extra_headers_json, dict):
        for k, v in cfg.extra_headers_json.items():
            if k and v is not None:
                headers[str(k)] = str(v)

    prompt_tokens_est, est_flag = count_tokens(f"{system_msg}\n{user_msg}", cfg.model_name)
    metadata = {
        "symbol": symbol,
        "strategy": strategy_name,
        "direction": position_direction,
        "account_alias": account_alias,
        "account_service": account_service,
    }
    try:
        timeout = max(5.0, float(cfg.timeout_seconds or 30))
        resp = httpx.post(url, headers=headers, json=body, timeout=timeout)
        resp.raise_for_status()
        data = resp.json() if resp.content else {}
        if not isinstance(data, dict):
            data = {}
        output_text = _collect_output_text(data)
        should_close, reason = _parse_ai_exit_decision(output_text)
        latency_ms = max(0, int((time.perf_counter() - started) * 1000))

        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        in_tok = int(usage.get("input_tokens") or prompt_tokens_est)
        out_tok = int(usage.get("output_tokens") or 0)
        est = est_flag or feedback_estimated or not bool(usage)
        log_token_usage(
            config=cfg,
            provider=cfg.provider,
            model_name=str(data.get("model") or cfg.model_name),
            operation="ai_exit_gate",
            prompt_tokens=in_tok,
            completion_tokens=out_tok,
            context_tokens=ctx.used_context_tokens + feedback_tokens,
            estimated=est,
            metadata={
                **metadata,
                "reason": reason,
                "close": should_close,
                "cfg_alias": cfg.name_alias,
                "latency_ms": latency_ms,
            },
        )
        feedback_level = "warning" if should_close else "info"
        if reason == "ai_exit_parse_failed":
            feedback_level = "warning"
        record_ai_feedback_event(
            event_type="ai_exit_decision",
            level=feedback_level,
            config=cfg,
            account_alias=account_alias,
            account_service=account_service,
            symbol=symbol,
            strategy=strategy_name,
            allow=not should_close,
            risk_mult=1.0,
            reason=reason,
            latency_ms=latency_ms,
            payload={
                "direction": position_direction,
                "session": session_name,
                "sig_score": round(_to_float(sig_score, 0.0), 6),
                "pnl_pct_gross": round(_to_float(pnl_pct_gross, 0.0), 6),
                "pnl_pct_gate": round(_to_float(pnl_pct_gate, 0.0), 6),
                "r_multiple": round(_to_float(r_multiple, 0.0), 4),
                "remaining_tp_pct": round(max(0.0, _to_float(remaining_tp_pct, 0.0)), 6),
                "cfg_alias": cfg.name_alias,
                "ai_output_preview": _preview_text(output_text),
                "ai_output_chars": len(str(output_text or "")),
            },
        )
        meta.update(
            {
                "cfg_alias": cfg.name_alias,
                "cfg_id": cfg.id,
                "model": str(data.get("model") or cfg.model_name),
                "ctx_tokens": ctx.used_context_tokens,
                "feedback_tokens": feedback_tokens,
                "prompt_tokens": in_tok,
                "completion_tokens": out_tok,
                "latency_ms": latency_ms,
            }
        )
        return should_close, reason, meta
    except httpx.HTTPStatusError as exc:
        err_detail = _http_status_error_detail(exc)
        logger.warning("AI exit gate failed for %s (%s): %s", symbol, strategy_name, err_detail)
        latency_ms = max(0, int((time.perf_counter() - started) * 1000))
        log_token_usage(
            config=cfg,
            provider=cfg.provider,
            model_name=cfg.model_name,
            operation="ai_exit_gate_error",
            prompt_tokens=prompt_tokens_est,
            completion_tokens=0,
            context_tokens=ctx.used_context_tokens + feedback_tokens,
            estimated=True,
            metadata={**metadata, "error": err_detail, "cfg_alias": cfg.name_alias},
        )
        record_ai_feedback_event(
            event_type="ai_exit_error",
            level="error",
            config=cfg,
            account_alias=account_alias,
            account_service=account_service,
            symbol=symbol,
            strategy=strategy_name,
            allow=None,
            risk_mult=1.0,
            reason=err_detail[:255],
            latency_ms=latency_ms,
            payload={
                "cfg_alias": cfg.name_alias,
                "error_type": type(exc).__name__,
                "status_code": getattr(exc.response, "status_code", None),
            },
        )
        if bool(getattr(settings, "AI_EXIT_GATE_NOTIFY_ERRORS", True)):
            try:
                from risk.notifications import notify_error

                notify_error("AI exit gate", err_detail)
            except Exception:
                pass
        return False, "ai_exit_error", meta
    except Exception as exc:
        logger.warning("AI exit gate failed for %s (%s): %s", symbol, strategy_name, exc)
        latency_ms = max(0, int((time.perf_counter() - started) * 1000))
        log_token_usage(
            config=cfg,
            provider=cfg.provider,
            model_name=cfg.model_name,
            operation="ai_exit_gate_error",
            prompt_tokens=prompt_tokens_est,
            completion_tokens=0,
            context_tokens=ctx.used_context_tokens + feedback_tokens,
            estimated=True,
            metadata={**metadata, "error": str(exc), "cfg_alias": cfg.name_alias},
        )
        record_ai_feedback_event(
            event_type="ai_exit_error",
            level="error",
            config=cfg,
            account_alias=account_alias,
            account_service=account_service,
            symbol=symbol,
            strategy=strategy_name,
            allow=None,
            risk_mult=1.0,
            reason=str(exc)[:255],
            latency_ms=latency_ms,
            payload={
                "cfg_alias": cfg.name_alias,
                "error_type": type(exc).__name__,
            },
        )
        if bool(getattr(settings, "AI_EXIT_GATE_NOTIFY_ERRORS", True)):
            try:
                from risk.notifications import notify_error

                notify_error("AI exit gate", str(exc))
            except Exception:
                pass
        return False, "ai_exit_error", meta
