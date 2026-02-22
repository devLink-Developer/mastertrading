from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


FEATURE_NAMES = [
    "score",
    "confidence",
    "raw_score",
    "net_score",
    "risk_budget_pct",
    "active_module_count",
    "has_module_rows",
    "is_alloc",
    "is_strategy_trend",
    "is_strategy_meanrev",
    "is_strategy_carry",
    "is_strategy_smc",
    "direction_long",
    "direction_short",
    "session_asia",
    "session_london",
    "session_ny",
    "session_overlap",
    "session_dead",
    "symbol_is_btc",
    "symbol_is_eth",
    "symbol_is_sol",
    "symbol_is_xrp",
    "symbol_is_doge",
    "symbol_is_ada",
    "symbol_is_link",
    "atr_pct",
    "spread_bps",
]


_CACHED_MODEL_PATH = ""
_CACHED_MODEL_MTIME_NS = -1
_CACHED_MODEL_DATA: dict[str, Any] | None = None


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_lower(value: Any) -> str:
    return str(value or "").strip().lower()


def _infer_direction(strategy_name: str, payload: dict[str, Any]) -> str:
    direction = _safe_lower(payload.get("direction"))
    if direction in {"long", "short"}:
        return direction
    if strategy_name.endswith("_long") or "long" in strategy_name:
        return "long"
    if strategy_name.endswith("_short") or "short" in strategy_name:
        return "short"
    return "flat"


def _extract_session(payload: dict[str, Any]) -> str:
    reasons = payload.get("reasons")
    if isinstance(reasons, dict):
        session = _safe_lower(reasons.get("session"))
        if session:
            return session
    return _safe_lower(payload.get("session"))


def _extract_active_modules(payload: dict[str, Any]) -> float:
    reasons = payload.get("reasons")
    if not isinstance(reasons, dict):
        return 0.0
    active_count = reasons.get("active_module_count")
    if active_count is not None:
        return max(0.0, _to_float(active_count))
    rows = reasons.get("module_rows")
    if isinstance(rows, list):
        return float(len(rows))
    return 0.0


def build_entry_feature_map(
    *,
    strategy_name: str,
    symbol: str,
    sig_score: float,
    payload: dict[str, Any] | None,
    atr_pct: float | None = None,
    spread_bps: float | None = None,
) -> dict[str, float]:
    payload = payload if isinstance(payload, dict) else {}
    strategy = _safe_lower(strategy_name)
    symbol_norm = _safe_lower(symbol).replace("/", "").replace(":", "")
    direction = _infer_direction(strategy, payload)
    session = _extract_session(payload)
    reasons = payload.get("reasons") if isinstance(payload.get("reasons"), dict) else {}
    module_rows = reasons.get("module_rows") if isinstance(reasons, dict) else None

    feature_map = {name: 0.0 for name in FEATURE_NAMES}
    feature_map["score"] = _to_float(sig_score)
    feature_map["confidence"] = _to_float(payload.get("confidence"))
    feature_map["raw_score"] = _to_float(payload.get("raw_score"))
    feature_map["net_score"] = _to_float(payload.get("net_score"))
    feature_map["risk_budget_pct"] = max(0.0, _to_float(payload.get("risk_budget_pct")))
    feature_map["active_module_count"] = _extract_active_modules(payload)
    feature_map["has_module_rows"] = 1.0 if isinstance(module_rows, list) and len(module_rows) > 0 else 0.0
    feature_map["is_alloc"] = 1.0 if strategy.startswith("alloc_") else 0.0
    feature_map["is_strategy_trend"] = 1.0 if "trend" in strategy else 0.0
    feature_map["is_strategy_meanrev"] = 1.0 if "meanrev" in strategy else 0.0
    feature_map["is_strategy_carry"] = 1.0 if "carry" in strategy else 0.0
    feature_map["is_strategy_smc"] = 1.0 if "smc" in strategy else 0.0
    feature_map["direction_long"] = 1.0 if direction == "long" else 0.0
    feature_map["direction_short"] = 1.0 if direction == "short" else 0.0
    feature_map["session_asia"] = 1.0 if session == "asia" else 0.0
    feature_map["session_london"] = 1.0 if session == "london" else 0.0
    feature_map["session_ny"] = 1.0 if session == "ny" else 0.0
    feature_map["session_overlap"] = 1.0 if session == "overlap" else 0.0
    feature_map["session_dead"] = 1.0 if session == "dead" else 0.0
    feature_map["symbol_is_btc"] = 1.0 if symbol_norm.startswith("btcusdt") else 0.0
    feature_map["symbol_is_eth"] = 1.0 if symbol_norm.startswith("ethusdt") else 0.0
    feature_map["symbol_is_sol"] = 1.0 if symbol_norm.startswith("solusdt") else 0.0
    feature_map["symbol_is_xrp"] = 1.0 if symbol_norm.startswith("xrpusdt") else 0.0
    feature_map["symbol_is_doge"] = 1.0 if symbol_norm.startswith("dogeusdt") else 0.0
    feature_map["symbol_is_ada"] = 1.0 if symbol_norm.startswith("adausdt") else 0.0
    feature_map["symbol_is_link"] = 1.0 if symbol_norm.startswith("linkusdt") else 0.0
    feature_map["atr_pct"] = max(0.0, _to_float(atr_pct))
    feature_map["spread_bps"] = max(0.0, _to_float(spread_bps))
    return feature_map


def vectorize_feature_map(
    feature_map: dict[str, float],
    feature_names: list[str] | None = None,
) -> np.ndarray:
    names = feature_names or FEATURE_NAMES
    return np.array([_to_float(feature_map.get(name, 0.0)) for name in names], dtype=np.float64)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def fit_logistic_model(
    x: np.ndarray,
    y: np.ndarray,
    *,
    feature_names: list[str] | None = None,
    epochs: int = 1200,
    learning_rate: float = 0.05,
    l2: float = 1e-3,
) -> dict[str, Any]:
    x_np = np.asarray(x, dtype=np.float64)
    y_np = np.asarray(y, dtype=np.float64).reshape(-1)
    if x_np.ndim != 2:
        raise ValueError("x must be 2D")
    samples, feature_count = x_np.shape
    if samples < 6:
        raise ValueError("not enough samples to fit model")
    if y_np.size != samples:
        raise ValueError("x/y shape mismatch")
    classes = np.unique(y_np)
    if classes.size < 2:
        raise ValueError("target must contain both classes")

    means = x_np.mean(axis=0)
    stds = x_np.std(axis=0)
    stds = np.where(stds < 1e-9, 1.0, stds)
    x_scaled = (x_np - means) / stds

    weights = np.zeros(feature_count, dtype=np.float64)
    bias = 0.0
    pos_count = max(float((y_np >= 0.5).sum()), 1.0)
    neg_count = max(float((y_np < 0.5).sum()), 1.0)
    pos_w = samples / (2.0 * pos_count)
    neg_w = samples / (2.0 * neg_count)
    sample_weights = np.where(y_np >= 0.5, pos_w, neg_w)

    for _ in range(max(1, int(epochs))):
        linear = x_scaled @ weights + bias
        probs = _sigmoid(linear)
        error = (probs - y_np) * sample_weights
        grad_w = (x_scaled.T @ error) / samples + (l2 * weights)
        grad_b = float(error.mean())
        weights -= learning_rate * grad_w
        bias -= learning_rate * grad_b

    fitted_probs = _sigmoid(x_scaled @ weights + bias)
    preds = (fitted_probs >= 0.5).astype(np.float64)
    accuracy = float((preds == y_np).mean())
    eps = 1e-9
    log_loss = float(
        -np.mean(y_np * np.log(fitted_probs + eps) + (1.0 - y_np) * np.log(1.0 - fitted_probs + eps))
    )

    model_feature_names = feature_names or FEATURE_NAMES
    if len(model_feature_names) != feature_count:
        raise ValueError("feature names length mismatch")

    return {
        "version": 1,
        "feature_names": list(model_feature_names),
        "means": means.tolist(),
        "stds": stds.tolist(),
        "weights": weights.tolist(),
        "bias": float(bias),
        "train_metrics": {
            "samples": int(samples),
            "positive_rate": float(y_np.mean()),
            "accuracy_at_05": accuracy,
            "log_loss": log_loss,
        },
    }


def predict_proba_from_model(model: dict[str, Any], feature_map: dict[str, float]) -> float:
    names = [str(name) for name in model.get("feature_names", [])]
    means = np.asarray(model.get("means", []), dtype=np.float64)
    stds = np.asarray(model.get("stds", []), dtype=np.float64)
    weights = np.asarray(model.get("weights", []), dtype=np.float64)
    bias = _to_float(model.get("bias"))
    if len(names) == 0:
        raise ValueError("model has no feature names")
    if not (len(means) == len(stds) == len(weights) == len(names)):
        raise ValueError("invalid model dimensions")

    x = vectorize_feature_map(feature_map, feature_names=names)
    stds_safe = np.where(stds < 1e-9, 1.0, stds)
    x_scaled = (x - means) / stds_safe
    linear = float(np.dot(x_scaled, weights) + bias)
    return float(_sigmoid(np.array([linear], dtype=np.float64))[0])


def save_model(model: dict[str, Any], path: str | Path) -> Path:
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as fh:
        json.dump(model, fh, indent=2, sort_keys=True)
    return target


def load_model(path: str | Path) -> dict[str, Any]:
    model_path = Path(path).expanduser()
    with model_path.open("r", encoding="utf-8") as fh:
        model = json.load(fh)
    _ = predict_proba_from_model(model, {name: 0.0 for name in model.get("feature_names", [])})
    return model


def load_model_cached(path: str | Path) -> dict[str, Any] | None:
    global _CACHED_MODEL_PATH, _CACHED_MODEL_MTIME_NS, _CACHED_MODEL_DATA
    model_path = Path(path).expanduser()
    if not model_path.exists():
        return None
    try:
        mtime_ns = model_path.stat().st_mtime_ns
    except OSError:
        return None
    cache_path = str(model_path.resolve())
    if _CACHED_MODEL_DATA is not None and _CACHED_MODEL_PATH == cache_path and _CACHED_MODEL_MTIME_NS == mtime_ns:
        return _CACHED_MODEL_DATA
    try:
        loaded = load_model(model_path)
    except Exception:
        return None
    _CACHED_MODEL_PATH = cache_path
    _CACHED_MODEL_MTIME_NS = mtime_ns
    _CACHED_MODEL_DATA = loaded
    return loaded


def predict_entry_success_probability(
    *,
    strategy_name: str,
    symbol: str,
    sig_score: float,
    payload: dict[str, Any] | None,
    model_path: str | Path,
    atr_pct: float | None = None,
    spread_bps: float | None = None,
) -> float | None:
    model = load_model_cached(model_path)
    if not model:
        return None
    feature_map = build_entry_feature_map(
        strategy_name=strategy_name,
        symbol=symbol,
        sig_score=sig_score,
        payload=payload,
        atr_pct=atr_pct,
        spread_bps=spread_bps,
    )
    try:
        return predict_proba_from_model(model, feature_map)
    except Exception:
        return None
