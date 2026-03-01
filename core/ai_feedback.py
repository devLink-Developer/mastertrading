from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
from datetime import timezone

from django.conf import settings
from core.api_runtime import count_tokens
from core.models import AiFeedbackEvent, ApiProviderConfig


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _clamp_float(value: Any, low: float, high: float, default: float) -> float:
    val = _to_float(value, default)
    return max(low, min(high, val))


def _stream_path(path_override: str = "") -> Path:
    raw = str(path_override or getattr(settings, "AI_FEEDBACK_JSONL_PATH", "") or "").strip()
    rel = raw or "tmp/ai/feedback_stream.jsonl"
    base_dir = Path(settings.BASE_DIR).resolve()
    candidate = Path(rel)
    target = (base_dir / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
    if base_dir != target and base_dir not in target.parents:
        # Hard fail-safe against writing outside project tree.
        target = (base_dir / "tmp" / "ai" / "feedback_stream.jsonl").resolve()
    return target


def _compact_line(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


def _fingerprint(
    *,
    event_type: str,
    account_alias: str,
    account_service: str,
    symbol: str,
    strategy: str,
    reason: str,
) -> str:
    base = "|".join(
        [
            str(event_type or "").strip().lower(),
            str(account_alias or "").strip().lower(),
            str(account_service or "").strip().lower(),
            str(symbol or "").strip().upper(),
            str(strategy or "").strip().lower(),
            str(reason or "").strip().lower()[:140],
        ]
    )
    return hashlib.sha1(base.encode("utf-8", errors="ignore")).hexdigest()


def _trim_stream_file(path: Path) -> None:
    max_bytes = max(64_000, int(getattr(settings, "AI_FEEDBACK_JSONL_MAX_BYTES", 2_000_000) or 2_000_000))
    keep_ratio = _clamp_float(getattr(settings, "AI_FEEDBACK_JSONL_TRIM_KEEP_RATIO", 0.70), 0.10, 0.95, 0.70)
    try:
        size = path.stat().st_size
    except Exception:
        return
    if size <= max_bytes:
        return
    keep_bytes = max(8_000, int(max_bytes * keep_ratio))
    with path.open("rb") as fh:
        if size > keep_bytes:
            fh.seek(-keep_bytes, 2)
        chunk = fh.read()
    # Align to full line.
    nl = chunk.find(b"\n")
    if nl >= 0 and nl + 1 < len(chunk):
        chunk = chunk[nl + 1 :]
    with path.open("wb") as out:
        out.write(chunk)


def append_feedback_stream_line(payload: dict[str, Any], *, path_override: str = "") -> str:
    path = _stream_path(path_override=path_override)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = _compact_line(payload)
    with path.open("a", encoding="utf-8", newline="\n") as fh:
        fh.write(line)
        fh.write("\n")
    _trim_stream_file(path)
    return path.relative_to(Path(settings.BASE_DIR).resolve()).as_posix()


def feedback_event_to_compact_payload(row: AiFeedbackEvent) -> dict[str, Any]:
    ts = row.created_at.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    return {
        "t": ts,
        "ev": row.event_type,
        "lv": row.level,
        "acc": row.account_alias,
        "svc": row.account_service,
        "sym": row.symbol,
        "st": row.strategy,
        "ok": row.allow,
        "rm": float(row.risk_mult) if row.risk_mult is not None else None,
        "r": row.reason,
        "lat": row.latency_ms,
        "fp": row.fingerprint,
        "p": row.payload_json or {},
    }


def record_ai_feedback_event(
    *,
    event_type: str,
    level: str = AiFeedbackEvent.Level.INFO,
    config: ApiProviderConfig | None = None,
    account_alias: str = "",
    account_service: str = "",
    symbol: str = "",
    strategy: str = "",
    allow: bool | None = None,
    risk_mult: float | None = None,
    reason: str = "",
    latency_ms: int = 0,
    payload: dict[str, Any] | None = None,
    path_override: str = "",
) -> AiFeedbackEvent:
    level_txt = str(level or AiFeedbackEvent.Level.INFO).strip().lower()
    allowed_levels = {k for k, _ in AiFeedbackEvent.Level.choices}
    if level_txt not in allowed_levels:
        level_txt = AiFeedbackEvent.Level.INFO

    risk = None
    if risk_mult is not None:
        risk = _clamp_float(risk_mult, 0.0, 1.0, 1.0)

    reason_txt = str(reason or "").strip()[:255]
    fp = _fingerprint(
        event_type=event_type,
        account_alias=account_alias,
        account_service=account_service,
        symbol=symbol,
        strategy=strategy,
        reason=reason_txt,
    )
    row = AiFeedbackEvent.objects.create(
        config=config,
        account_alias=str(account_alias or "").strip(),
        account_service=str(account_service or "").strip().lower(),
        symbol=str(symbol or "").strip().upper(),
        strategy=str(strategy or "").strip().lower(),
        event_type=str(event_type or "").strip().lower()[:48],
        level=level_txt,
        allow=allow if allow is None else bool(allow),
        risk_mult=risk,
        reason=reason_txt,
        latency_ms=max(0, int(latency_ms or 0)),
        fingerprint=fp,
        payload_json=payload or {},
    )

    if bool(getattr(settings, "AI_FEEDBACK_JSONL_ENABLED", True)):
        compact_payload = feedback_event_to_compact_payload(row)
        append_feedback_stream_line(compact_payload, path_override=path_override)
    return row


def load_feedback_context_tail(
    *,
    model_name: str,
    max_tokens: int,
    path_override: str = "",
) -> tuple[str, int, bool]:
    """
    Read compact JSONL feedback stream and return latest lines that fit token budget.
    """
    budget = max(0, int(max_tokens or 0))
    if budget <= 0:
        return "", 0, True
    path = _stream_path(path_override=path_override)
    if not path.exists():
        return "", 0, True
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return "", 0, True
    if not lines:
        return "", 0, True

    selected: list[str] = []
    used = 0
    estimated = False
    # Newest-first accumulation, then reverse to keep chronological order.
    for raw in reversed(lines):
        txt = str(raw or "").strip()
        if not txt:
            continue
        tok, est = count_tokens(txt, model_name)
        if used + tok > budget:
            break
        selected.append(txt)
        used += tok
        estimated = estimated or est
    if not selected:
        return "", 0, estimated
    selected.reverse()
    return "\n".join(selected), used, estimated
