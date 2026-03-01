from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from django.conf import settings

from core.models import ApiContextFile, ApiProviderConfig, ApiTokenUsageLog

try:  # pragma: no cover - optional dependency
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    tiktoken = None


def get_active_api_config(
    *,
    provider: str | None = None,
    owner_id: int | None = None,
    name_alias: str | None = None,
) -> ApiProviderConfig | None:
    qs = ApiProviderConfig.objects.filter(active=True)
    if provider:
        qs = qs.filter(provider=str(provider).strip().lower())
    if name_alias:
        qs = qs.filter(name_alias=name_alias)
    if owner_id:
        owner_qs = qs.filter(owner_id=owner_id)
        cfg = owner_qs.filter(is_default=True).order_by("-updated_at").first()
        if cfg:
            return cfg
        cfg = owner_qs.order_by("-updated_at").first()
        if cfg:
            return cfg
        qs = qs.filter(owner__isnull=True)
    cfg = qs.filter(is_default=True).order_by("-updated_at").first()
    return cfg or qs.order_by("-updated_at").first()


def _encoding_for_model(model_name: str):
    if tiktoken is None:
        return None
    try:
        return tiktoken.encoding_for_model(model_name or "")
    except Exception:  # pragma: no cover - defensive
        try:
            return tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None


def count_tokens(text: str, model_name: str = "") -> tuple[int, bool]:
    txt = str(text or "")
    if not txt:
        return 0, True
    enc = _encoding_for_model(model_name)
    if enc is not None:
        try:
            return len(enc.encode(txt)), False
        except Exception:
            pass
    # Safe fallback when tokenizer package is unavailable.
    return max(1, int(math.ceil(len(txt) / 4.0))), True


def trim_text_to_token_budget(
    text: str,
    *,
    model_name: str,
    max_tokens: int,
    trim_mode: str = "tail",
) -> tuple[str, int, bool]:
    if max_tokens <= 0:
        return "", 0, True
    raw = str(text or "")
    if not raw:
        return "", 0, True

    tok, estimated = count_tokens(raw, model_name)
    if tok <= max_tokens:
        return raw, tok, estimated

    mode = str(trim_mode or "tail").strip().lower()
    if mode not in {"head", "tail"}:
        mode = "tail"

    lo, hi = 0, len(raw)
    best = ""
    best_tok = 0
    best_est = estimated
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = raw[:mid] if mode == "head" else raw[-mid:]
        c_tok, c_est = count_tokens(candidate, model_name)
        if c_tok <= max_tokens:
            best = candidate
            best_tok = c_tok
            best_est = c_est
            lo = mid + 1
        else:
            hi = mid - 1
    return best, best_tok, best_est


def _resolve_safe_context_path(file_path: str) -> Path:
    base_dir = Path(settings.BASE_DIR).resolve()
    raw = Path(str(file_path or "").strip())
    if not raw:
        raise ValueError("empty path")
    candidate = (base_dir / raw).resolve() if not raw.is_absolute() else raw.resolve()
    if base_dir != candidate and base_dir not in candidate.parents:
        raise ValueError(f"path outside BASE_DIR not allowed: {file_path}")
    return candidate


@dataclass
class ContextFileResult:
    context_file_id: int
    file_path: str
    used_tokens: int
    allocated_tokens: int
    skipped: bool
    reason: str = ""
    estimated: bool = False


@dataclass
class OptimizedContextResult:
    config_id: int
    model_name: str
    user_tokens: int
    reserve_output_tokens: int
    available_input_tokens: int
    used_context_tokens: int
    total_prompt_tokens: int
    estimated: bool
    context_text: str
    files: list[ContextFileResult] = field(default_factory=list)


def build_optimized_context(
    config: ApiProviderConfig,
    *,
    user_prompt: str = "",
    reserve_output_tokens: int | None = None,
) -> OptimizedContextResult:
    model_name = str(config.model_name or "")
    reserve = (
        int(reserve_output_tokens)
        if reserve_output_tokens is not None
        else int(config.max_output_tokens or 0)
    )
    reserve = max(0, reserve)
    max_input = max(1, int(config.max_input_tokens or 0))
    user_tokens, user_est = count_tokens(user_prompt, model_name)
    available = max(0, max_input - reserve - user_tokens)

    files_qs = config.context_files.filter(enabled=True).order_by("priority", "id")
    sections: list[str] = []
    file_results: list[ContextFileResult] = []
    used_ctx_tokens = 0
    estimated_any = user_est

    for source in files_qs:
        remaining = max(0, available - used_ctx_tokens)
        allowed = min(int(source.max_tokens or 0), remaining)
        if allowed <= 0:
            reason = "required_no_budget" if source.required else "no_budget"
            file_results.append(
                ContextFileResult(
                    context_file_id=source.id,
                    file_path=source.file_path,
                    used_tokens=0,
                    allocated_tokens=0,
                    skipped=True,
                    reason=reason,
                )
            )
            continue
        try:
            path = _resolve_safe_context_path(source.file_path)
            text = path.read_text(encoding="utf-8")
        except Exception as exc:
            file_results.append(
                ContextFileResult(
                    context_file_id=source.id,
                    file_path=source.file_path,
                    used_tokens=0,
                    allocated_tokens=max(0, allowed),
                    skipped=True,
                    reason=f"read_error:{type(exc).__name__}",
                )
            )
            continue

        if source.max_chars and source.max_chars > 0:
            limit = int(source.max_chars)
            text = text[:limit] if source.trim_mode == ApiContextFile.TrimMode.HEAD else text[-limit:]

        header = ""
        if source.include_header:
            header_title = (source.name or source.file_path).strip()
            header = f"\n\n### {header_title}\n"
        header_tokens, header_est = count_tokens(header, model_name)
        estimated_any = estimated_any or header_est
        payload_budget = max(0, allowed - header_tokens)
        if payload_budget <= 0:
            file_results.append(
                ContextFileResult(
                    context_file_id=source.id,
                    file_path=source.file_path,
                    used_tokens=0,
                    allocated_tokens=max(0, allowed),
                    skipped=True,
                    reason="header_over_budget",
                    estimated=header_est,
                )
            )
            continue

        trimmed, _tok_count, tok_est = trim_text_to_token_budget(
            text,
            model_name=model_name,
            max_tokens=payload_budget,
            trim_mode=source.trim_mode,
        )
        estimated_any = estimated_any or tok_est
        if not trimmed.strip():
            file_results.append(
                ContextFileResult(
                    context_file_id=source.id,
                    file_path=source.file_path,
                    used_tokens=0,
                    allocated_tokens=max(0, allowed),
                    skipped=True,
                    reason="empty_after_trim",
                    estimated=tok_est,
                )
            )
            continue

        section = f"{header}{trimmed}".strip()
        sec_tokens, sec_est = count_tokens(section, model_name)
        estimated_any = estimated_any or sec_est

        if used_ctx_tokens + sec_tokens > available:
            file_results.append(
                ContextFileResult(
                    context_file_id=source.id,
                    file_path=source.file_path,
                    used_tokens=0,
                    allocated_tokens=max(0, allowed),
                    skipped=True,
                    reason="over_budget",
                    estimated=sec_est,
                )
            )
            continue

        sections.append(section)
        used_ctx_tokens += sec_tokens
        file_results.append(
            ContextFileResult(
                context_file_id=source.id,
                file_path=source.file_path,
                used_tokens=sec_tokens,
                allocated_tokens=max(0, allowed),
                skipped=False,
                estimated=sec_est,
            )
        )

        if used_ctx_tokens >= available:
            break

    context_text = "\n\n".join([s for s in sections if s]).strip()
    total_prompt = user_tokens + used_ctx_tokens
    return OptimizedContextResult(
        config_id=config.id,
        model_name=model_name,
        user_tokens=user_tokens,
        reserve_output_tokens=reserve,
        available_input_tokens=available,
        used_context_tokens=used_ctx_tokens,
        total_prompt_tokens=total_prompt,
        estimated=estimated_any,
        context_text=context_text,
        files=file_results,
    )


def log_token_usage(
    *,
    config: ApiProviderConfig | None,
    provider: str,
    model_name: str,
    operation: str,
    prompt_tokens: int,
    completion_tokens: int,
    context_tokens: int = 0,
    estimated: bool = False,
    metadata: dict[str, Any] | None = None,
) -> ApiTokenUsageLog:
    prompt = max(0, int(prompt_tokens))
    completion = max(0, int(completion_tokens))
    total = prompt + completion
    return ApiTokenUsageLog.objects.create(
        config=config,
        provider=str(provider or ""),
        model_name=str(model_name or ""),
        operation=str(operation or ""),
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        context_tokens=max(0, int(context_tokens)),
        estimated=bool(estimated),
        metadata_json=metadata or {},
    )
