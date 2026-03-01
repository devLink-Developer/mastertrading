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

logger = logging.getLogger(__name__)


def _dir_code(direction: str) -> str:
    d = str(direction or "").strip().lower()
    if d.startswith("long"):
        return "l"
    if d.startswith("short"):
        return "s"
    return d[:1] if d else "u"


def _to_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except Exception:
        return default


def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(val)))


def _collect_output_text(resp_json: dict[str, Any]) -> str:
    text = str(resp_json.get("output_text") or "").strip()
    if text:
        return text
    chunks: list[str] = []
    for msg in resp_json.get("output", []) or []:
        if not isinstance(msg, dict):
            continue
        for item in msg.get("content", []) or []:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "output_text":
                part = str(item.get("text") or "")
                if part:
                    chunks.append(part)
    return "\n".join(chunks).strip()


def _model_name_lc(model_name: str | None) -> str:
    return str(model_name or "").strip().lower()


def _supports_sampling_controls(model_name: str | None) -> bool:
    """
    Some reasoning-oriented model families reject temperature/top_p in Responses API.
    Keep the request minimal for those families.
    """
    model = _model_name_lc(model_name)
    if not model:
        return True
    unsupported_prefixes = (
        "gpt-5",
        "o1",
        "o3",
        "o4-mini",
    )
    return not any(model.startswith(prefix) for prefix in unsupported_prefixes)


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
    if _supports_sampling_controls(cfg.model_name):
        body["temperature"] = float(cfg.temperature)
        body["top_p"] = float(cfg.top_p)

    if isinstance(cfg.extra_params_json, dict):
        body.update(cfg.extra_params_json)
        # Enforce compatibility if extra params inject unsupported knobs.
        if not _supports_sampling_controls(cfg.model_name):
            body.pop("temperature", None)
            body.pop("top_p", None)
    return body


def _http_status_error_detail(exc: httpx.HTTPStatusError) -> str:
    status = getattr(exc.response, "status_code", None)
    detail = ""
    try:
        payload = exc.response.json()
        if isinstance(payload, dict):
            err = payload.get("error")
            if isinstance(err, dict):
                detail = str(err.get("message") or "")
    except Exception:
        detail = ""
    if not detail:
        try:
            detail = str(exc.response.text or "").strip()
        except Exception:
            detail = ""
    detail = detail[:500] if detail else ""
    if status is None:
        return f"{exc}"
    if detail:
        return f"HTTP {status}: {detail}"
    return f"HTTP {status}: {exc}"


def _extract_json_blob(raw: str) -> dict[str, Any] | None:
    txt = (raw or "").strip()
    if not txt:
        return None
    try:
        parsed = json.loads(txt)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    m = re.search(r"\{.*\}", txt, flags=re.DOTALL)
    if not m:
        return None
    try:
        parsed = json.loads(m.group(0))
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def _parse_ai_decision(raw_text: str) -> tuple[bool, float, str]:
    payload = _extract_json_blob(raw_text)
    if not payload:
        return True, 1.0, "ai_parse_failed"
    allow = bool(payload.get("allow", True))
    risk_mult = _clamp(_to_float(payload.get("risk_mult", 1.0), 1.0), 0.0, 1.0)
    reason = str(payload.get("reason") or "").strip() or "ai_no_reason"
    return allow, risk_mult, reason


def _compact_payload(sig_payload: dict[str, Any]) -> dict[str, Any]:
    payload = sig_payload if isinstance(sig_payload, dict) else {}
    out: dict[str, Any] = {}
    reasons = payload.get("reasons")
    if isinstance(reasons, dict):
        out["ns"] = reasons.get("net_score")
        rows = reasons.get("module_rows")
        compact_rows: list[list[Any]] = []
        if isinstance(rows, list):
            for row in rows[:6]:
                if not isinstance(row, dict):
                    continue
                compact_rows.append(
                    [
                        row.get("module"),
                        _dir_code(str(row.get("direction") or "")),
                        row.get("confidence"),
                        row.get("raw_score"),
                    ]
                )
        if compact_rows:
            out["mr"] = compact_rows
    if "risk_budget_pct" in payload:
        out["rb"] = payload.get("risk_budget_pct")
    if "entry_reason" in payload:
        out["er"] = payload.get("entry_reason")
    if "regime" in payload:
        out["rg"] = payload.get("regime")
    if "session" in payload:
        out["se"] = payload.get("session")
    return out


def _build_gate_messages(
    *,
    user_prompt: str,
    ctx_text: str,
    feedback_text: str,
) -> tuple[str, str]:
    # Keep instruction and payload compact; this path runs on each allocator candidate.
    system_msg = (
        "Trading risk gate. "
        "Input keys: sym,st,dir,sc,atr,spr,sl,ses,sig(ns,mr=[m,d,c,s],rb,er,rg,se). "
        "Return JSON only: {\"allow\":bool,\"risk_mult\":0..1,\"reason\":\"short\"}."
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
    provider = str(getattr(settings, "AI_ENTRY_GATE_DEFAULT_PROVIDER", "openai") or "openai").strip().lower()
    return get_active_api_config(
        provider=provider or None,
        owner_id=account_owner_id,
    )


def evaluate_ai_entry_gate(
    *,
    account_ai_enabled: bool,
    account_ai_config_id: int | None,
    account_owner_id: int | None,
    account_alias: str,
    account_service: str,
    symbol: str,
    strategy_name: str,
    signal_direction: str,
    sig_score: float,
    atr: float | None,
    spread_bps: float | None,
    sl_pct: float,
    session_name: str,
    sig_payload: dict[str, Any],
) -> tuple[bool, float, str, dict[str, Any]]:
    """
    Return (allow, risk_mult, reason, meta).
    - allow: block/open decision
    - risk_mult: [0..1] optional additional risk reduction
    """
    meta: dict[str, Any] = {
        "account_alias": account_alias,
        "account_service": account_service,
        "symbol": symbol,
        "strategy": strategy_name,
    }
    if not bool(getattr(settings, "AI_ENTRY_GATE_ENABLED", True)):
        return True, 1.0, "ai_gate_disabled_global", meta
    if not account_ai_enabled:
        return True, 1.0, "ai_gate_disabled_account", meta
    if bool(getattr(settings, "AI_ENTRY_GATE_ONLY_ALLOCATOR", True)):
        if not str(strategy_name or "").strip().lower().startswith("alloc_"):
            return True, 1.0, "ai_gate_skipped_non_allocator", meta

    cfg = _resolve_config(
        account_ai_config_id=account_ai_config_id,
        account_owner_id=account_owner_id,
    )
    fail_open = bool(getattr(settings, "AI_ENTRY_GATE_FAIL_OPEN", True))
    if not cfg:
        record_ai_feedback_event(
            event_type="ai_gate_missing_config",
            level="warning",
            account_alias=account_alias,
            account_service=account_service,
            symbol=symbol,
            strategy=strategy_name,
            allow=True if fail_open else False,
            risk_mult=1.0 if fail_open else 0.0,
            reason="ai_no_config",
            payload={"fail_open": fail_open},
        )
        return (True, 1.0, "ai_no_config_fail_open", meta) if fail_open else (False, 0.0, "ai_no_config_fail_closed", meta)
    if not str(cfg.api_key or "").strip():
        record_ai_feedback_event(
            event_type="ai_gate_missing_api_key",
            level="warning",
            config=cfg,
            account_alias=account_alias,
            account_service=account_service,
            symbol=symbol,
            strategy=strategy_name,
            allow=True if fail_open else False,
            risk_mult=1.0 if fail_open else 0.0,
            reason="ai_no_api_key",
            payload={"fail_open": fail_open, "cfg_alias": cfg.name_alias},
        )
        return (True, 1.0, "ai_no_api_key_fail_open", meta) if fail_open else (False, 0.0, "ai_no_api_key_fail_closed", meta)

    started = time.perf_counter()
    candidate = {
        "sym": symbol,
        "st": strategy_name,
        "dir": _dir_code(signal_direction),
        "sc": round(_to_float(sig_score, 0.0), 6),
        "atr": None if atr is None else round(_to_float(atr, 0.0), 6),
        "spr": None if spread_bps is None else round(_to_float(spread_bps, 0.0), 3),
        "sl": round(_to_float(sl_pct, 0.0), 6),
        "ses": session_name,
        "sig": _compact_payload(sig_payload),
    }
    user_prompt = json.dumps(candidate, ensure_ascii=True, separators=(",", ":"))

    reserve_out = min(
        int(getattr(settings, "AI_ENTRY_GATE_MAX_OUTPUT_TOKENS", 180) or 180),
        int(cfg.max_output_tokens or 180),
    )
    ctx = build_optimized_context(
        cfg,
        user_prompt=user_prompt,
        reserve_output_tokens=reserve_out,
    )
    feedback_budget = max(
        0,
        int(getattr(settings, "AI_FEEDBACK_CONTEXT_MAX_TOKENS", 800) or 800),
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
        "direction": signal_direction,
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
        allow, risk_mult, reason = _parse_ai_decision(output_text)
        latency_ms = max(0, int((time.perf_counter() - started) * 1000))

        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        in_tok = int(usage.get("input_tokens") or prompt_tokens_est)
        out_tok = int(usage.get("output_tokens") or 0)
        est = est_flag or feedback_estimated or not bool(usage)
        log_token_usage(
            config=cfg,
            provider=cfg.provider,
            model_name=str(data.get("model") or cfg.model_name),
            operation="ai_entry_gate",
            prompt_tokens=in_tok,
            completion_tokens=out_tok,
            context_tokens=ctx.used_context_tokens + feedback_tokens,
            estimated=est,
            metadata={
                **metadata,
                "reason": reason,
                "allow": allow,
                "risk_mult": risk_mult,
                "cfg_alias": cfg.name_alias,
                "latency_ms": latency_ms,
            },
        )
        feedback_level = "warning" if not allow else "info"
        record_ai_feedback_event(
            event_type="ai_gate_decision",
            level=feedback_level,
            config=cfg,
            account_alias=account_alias,
            account_service=account_service,
            symbol=symbol,
            strategy=strategy_name,
            allow=allow,
            risk_mult=risk_mult,
            reason=reason,
            latency_ms=latency_ms,
            payload={
                "sig_score": round(_to_float(sig_score, 0.0), 6),
                "direction": signal_direction,
                "session": session_name,
                "spread_bps": None if spread_bps is None else round(_to_float(spread_bps, 0.0), 3),
                "cfg_alias": cfg.name_alias,
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
        return allow, risk_mult, reason, meta
    except httpx.HTTPStatusError as exc:
        err_detail = _http_status_error_detail(exc)
        logger.warning("AI entry gate failed for %s (%s): %s", symbol, strategy_name, err_detail)
        latency_ms = max(0, int((time.perf_counter() - started) * 1000))
        log_token_usage(
            config=cfg,
            provider=cfg.provider,
            model_name=cfg.model_name,
            operation="ai_entry_gate_error",
            prompt_tokens=prompt_tokens_est,
            completion_tokens=0,
            context_tokens=ctx.used_context_tokens + feedback_tokens,
            estimated=True,
            metadata={**metadata, "error": err_detail, "cfg_alias": cfg.name_alias},
        )
        record_ai_feedback_event(
            event_type="ai_gate_error",
            level="error",
            config=cfg,
            account_alias=account_alias,
            account_service=account_service,
            symbol=symbol,
            strategy=strategy_name,
            allow=True if fail_open else False,
            risk_mult=1.0 if fail_open else 0.0,
            reason=err_detail[:255],
            latency_ms=latency_ms,
            payload={
                "cfg_alias": cfg.name_alias,
                "error_type": type(exc).__name__,
                "fail_open": fail_open,
                "status_code": getattr(exc.response, "status_code", None),
            },
        )
        if bool(getattr(settings, "AI_ENTRY_GATE_NOTIFY_ERRORS", True)):
            try:
                from risk.notifications import notify_error

                notify_error("AI entry gate", err_detail)
            except Exception:
                pass
        if fail_open:
            return True, 1.0, "ai_error_fail_open", meta
        return False, 0.0, "ai_error_fail_closed", meta
    except Exception as exc:
        logger.warning("AI entry gate failed for %s (%s): %s", symbol, strategy_name, exc)
        latency_ms = max(0, int((time.perf_counter() - started) * 1000))
        log_token_usage(
            config=cfg,
            provider=cfg.provider,
            model_name=cfg.model_name,
            operation="ai_entry_gate_error",
            prompt_tokens=prompt_tokens_est,
            completion_tokens=0,
            context_tokens=ctx.used_context_tokens + feedback_tokens,
            estimated=True,
            metadata={**metadata, "error": str(exc), "cfg_alias": cfg.name_alias},
        )
        record_ai_feedback_event(
            event_type="ai_gate_error",
            level="error",
            config=cfg,
            account_alias=account_alias,
            account_service=account_service,
            symbol=symbol,
            strategy=strategy_name,
            allow=True if fail_open else False,
            risk_mult=1.0 if fail_open else 0.0,
            reason=str(exc)[:255],
            latency_ms=latency_ms,
            payload={
                "cfg_alias": cfg.name_alias,
                "error_type": type(exc).__name__,
                "fail_open": fail_open,
            },
        )
        if bool(getattr(settings, "AI_ENTRY_GATE_NOTIFY_ERRORS", True)):
            try:
                from risk.notifications import notify_error

                notify_error("AI entry gate", str(exc))
            except Exception:
                pass
        if fail_open:
            return True, 1.0, "ai_error_fail_open", meta
        return False, 0.0, "ai_error_fail_closed", meta
