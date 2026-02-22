from __future__ import annotations

from datetime import timedelta, timezone
import math
from typing import Any

import numpy as np
from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone as dj_tz

from backtest.models import BacktestTrade
from execution.ml_entry_filter import (
    FEATURE_NAMES,
    build_entry_feature_map,
    fit_logistic_model,
    save_model,
    vectorize_feature_map,
)
from execution.models import OperationReport
from signals.models import Signal
from signals.sessions import get_current_session


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _clamp01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(value, 1.0))


def _parse_symbol_list(raw: str) -> list[str]:
    return sorted({part.strip().upper() for part in str(raw or "").split(",") if part.strip()})


def _parse_strategy_list(raw: str) -> list[str]:
    items = []
    for part in str(raw or "").split(","):
        token = "".join(ch for ch in part.strip().lower() if ch.isalnum() or ch == "_").strip("_")
        if token:
            items.append(token)
    # stable dedup
    seen = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _backtest_direction_and_strategies(side: Any) -> tuple[str, set[str]]:
    side_txt = str(side or "").strip().lower()
    direction = "long" if side_txt in {"buy", "long"} else "short"
    # Backtest engine currently replays SMC detector; keep legacy alias for CLI compatibility.
    return direction, {f"smc_{direction}", f"backtest_{direction}"}


def _parse_run_ids(raw: str) -> list[int]:
    run_ids: set[int] = set()
    for part in str(raw or "").split(","):
        token = part.strip()
        if not token:
            continue
        try:
            run_ids.add(int(token))
        except ValueError as exc:
            raise CommandError(f"Invalid run id '{token}' in --backtest-runs") from exc
    return sorted(run_ids)


def _parse_signal_id(signal_id: Any, correlation_id: str) -> int | None:
    raw = str(signal_id or "").strip()
    if raw.isdigit():
        return int(raw)
    corr = str(correlation_id or "").strip()
    if not corr:
        return None
    head = corr.split("-", 1)[0].split(":", 1)[0].strip()
    if head.isdigit():
        return int(head)
    return None


def _session_from_datetime(value) -> str:
    if value is None:
        return "dead"
    dt = value
    if dj_tz.is_naive(dt):
        dt = dt.replace(tzinfo=timezone.utc)
    dt_utc = dt.astimezone(timezone.utc)
    return get_current_session(dt_utc.hour)


def _analyze_thresholds(model: dict[str, Any], x: np.ndarray, y: np.ndarray, pnl_abs: np.ndarray) -> dict[str, Any]:
    means = np.asarray(model.get("means", []), dtype=np.float64)
    stds = np.asarray(model.get("stds", []), dtype=np.float64)
    weights = np.asarray(model.get("weights", []), dtype=np.float64)
    bias = _to_float(model.get("bias"))
    if not (x.ndim == 2 and x.shape[1] == means.size == stds.size == weights.size):
        raise CommandError("threshold analysis failed: invalid model dimensions")

    stds_safe = np.where(stds < 1e-9, 1.0, stds)
    linear = ((x - means) / stds_safe) @ weights + bias
    probs = 1.0 / (1.0 + np.exp(-np.clip(linear, -35.0, 35.0)))

    baseline_trades = int(y.size)
    baseline_winrate = float(y.mean()) if baseline_trades > 0 else 0.0
    baseline_pnl = float(pnl_abs.sum()) if baseline_trades > 0 else 0.0

    min_selected = max(5, int(math.ceil(baseline_trades * 0.03)))
    best: dict[str, Any] | None = None
    for threshold in range(35, 91):
        thr = threshold / 100.0
        mask = probs >= thr
        selected = int(mask.sum())
        if selected < min_selected:
            continue
        selected_pnl = float(pnl_abs[mask].sum())
        selected_winrate = float(y[mask].mean()) if selected > 0 else 0.0
        candidate = {
            "recommended_threshold": thr,
            "recommended_selected_trades": selected,
            "recommended_pnl_abs": selected_pnl,
            "recommended_winrate": selected_winrate,
        }
        if best is None:
            best = candidate
            continue
        if selected_pnl > best["recommended_pnl_abs"] + 1e-9:
            best = candidate
            continue
        if abs(selected_pnl - best["recommended_pnl_abs"]) <= 1e-9:
            if selected_winrate > best["recommended_winrate"] + 1e-9:
                best = candidate
                continue
            if abs(selected_winrate - best["recommended_winrate"]) <= 1e-9 and selected > best["recommended_selected_trades"]:
                best = candidate

    if best is None:
        fallback_threshold = 0.5
        mask = probs >= fallback_threshold
        selected = int(mask.sum())
        if selected > 0:
            best = {
                "recommended_threshold": fallback_threshold,
                "recommended_selected_trades": selected,
                "recommended_pnl_abs": float(pnl_abs[mask].sum()),
                "recommended_winrate": float(y[mask].mean()),
            }
        else:
            best = {
                "recommended_threshold": fallback_threshold,
                "recommended_selected_trades": baseline_trades,
                "recommended_pnl_abs": baseline_pnl,
                "recommended_winrate": baseline_winrate,
            }

    return {
        "baseline_trades": baseline_trades,
        "baseline_winrate": baseline_winrate,
        "baseline_pnl_abs": baseline_pnl,
        **best,
    }


def _choose_holdout_split(y: np.ndarray) -> int | None:
    """
    Time-ordered holdout split for threshold selection.
    Returns split index for x[:split]=train and x[split:]=holdout, or None.
    """
    n = int(y.size)
    if n < 80:
        return None
    holdout_n = max(20, int(math.ceil(n * 0.2)))
    holdout_n = min(holdout_n, n - 6)
    if holdout_n < 20:
        return None
    split_idx = n - holdout_n
    y_train = y[:split_idx]
    y_holdout = y[split_idx:]
    if np.unique(y_train).size < 2:
        return None
    if np.unique(y_holdout).size < 2:
        return None
    return split_idx


class Command(BaseCommand):
    help = "Train ML entry filter model from live operations, backtest trades, or both."

    def add_arguments(self, parser):
        parser.add_argument(
            "--source",
            type=str,
            choices=["live", "backtest", "mixed"],
            default="mixed",
            help="Training dataset source (default: mixed).",
        )
        parser.add_argument(
            "--days",
            type=int,
            default=21,
            help="Lookback window in days for live OperationReport data.",
        )
        parser.add_argument(
            "--backtest-days",
            type=int,
            default=180,
            help="Lookback window in days for BacktestTrade data.",
        )
        parser.add_argument(
            "--backtest-runs",
            type=str,
            default="",
            help="Optional comma-separated BacktestRun ids to include.",
        )
        parser.add_argument(
            "--symbols",
            type=str,
            default="",
            help="Optional comma-separated symbols (e.g. BTCUSDT,ADAUSDT).",
        )
        parser.add_argument(
            "--strategies",
            type=str,
            default="",
            help="Optional comma-separated strategies (e.g. alloc_long,smc_short).",
        )
        parser.add_argument(
            "--min-samples",
            type=int,
            default=120,
            help="Minimum sample count required to train.",
        )
        parser.add_argument(
            "--epochs",
            type=int,
            default=1200,
            help="Epochs for logistic training.",
        )
        parser.add_argument(
            "--lr",
            type=float,
            default=0.05,
            help="Learning rate for logistic training.",
        )
        parser.add_argument(
            "--l2",
            type=float,
            default=0.001,
            help="L2 regularization strength.",
        )
        parser.add_argument(
            "--output",
            type=str,
            default=str(getattr(settings, "ML_ENTRY_FILTER_MODEL_PATH", "")),
            help="Destination JSON path for trained model.",
        )

    def handle(self, *args, **options):
        source = str(options["source"]).strip().lower()
        days = max(1, int(options["days"] or 21))
        backtest_days = max(1, int(options["backtest_days"] or 180))
        min_samples = int(options["min_samples"] or 0)
        if min_samples < 6:
            raise CommandError("--min-samples must be >= 6")
        epochs = max(1, int(options["epochs"] or 1200))
        lr = max(1e-5, float(options["lr"] or 0.05))
        l2 = max(0.0, float(options["l2"] or 0.001))
        output_path = str(options["output"] or "").strip()
        if not output_path:
            raise CommandError("--output is required")

        symbols = _parse_symbol_list(options.get("symbols", ""))
        strategies = _parse_strategy_list(options.get("strategies", ""))
        strategies_set = {s.lower() for s in strategies}
        backtest_run_ids = _parse_run_ids(options.get("backtest_runs", ""))
        now = dj_tz.now()
        from_live_dt = now - timedelta(days=days)
        from_backtest_dt = now - timedelta(days=backtest_days)

        rows: list[np.ndarray] = []
        labels: list[float] = []
        pnl_values: list[float] = []
        grouped_operations = 0
        skipped_missing_signal = 0
        live_samples = 0
        backtest_samples = 0
        backtest_trades = 0

        if source in {"live", "mixed"}:
            op_qs = OperationReport.objects.select_related("instrument").filter(closed_at__gte=from_live_dt)
            if symbols:
                op_qs = op_qs.filter(instrument__symbol__in=symbols)
            operations = list(op_qs.order_by("closed_at", "id"))

            grouped: dict[tuple[int, str], dict[str, Any]] = {}
            for op in operations:
                corr = str(op.correlation_id or "").strip()
                key = (op.instrument_id, corr if corr else f"op:{op.id}")
                if key not in grouped:
                    grouped[key] = {"op": op, "pnl_abs": 0.0}
                grouped[key]["pnl_abs"] += _to_float(op.pnl_abs)
                current_op = grouped[key]["op"]
                current_has_signal = str(current_op.signal_id or "").strip() != ""
                op_has_signal = str(op.signal_id or "").strip() != ""
                if (not current_has_signal and op_has_signal) or op.closed_at >= current_op.closed_at:
                    grouped[key]["op"] = op

            grouped_operations = len(grouped)
            parsed_ids: dict[tuple[int, str], int | None] = {}
            signal_ids: set[int] = set()
            for key, item in grouped.items():
                op = item["op"]
                sig_id = _parse_signal_id(op.signal_id, op.correlation_id)
                parsed_ids[key] = sig_id
                if sig_id is not None:
                    signal_ids.add(sig_id)
            signal_map = Signal.objects.in_bulk(signal_ids) if signal_ids else {}

            for key, item in grouped.items():
                op = item["op"]
                signal = signal_map.get(parsed_ids[key])
                if signal is None:
                    skipped_missing_signal += 1
                    continue
                if strategies_set:
                    sig_strategy = str(getattr(signal, "strategy", "") or "").strip().lower()
                    if sig_strategy not in strategies_set:
                        continue
                payload = signal.payload_json if isinstance(signal.payload_json, dict) else {}
                feature_map = build_entry_feature_map(
                    strategy_name=str(signal.strategy or ""),
                    symbol=op.instrument.symbol,
                    sig_score=_to_float(signal.score),
                    payload=payload,
                )
                rows.append(vectorize_feature_map(feature_map, feature_names=FEATURE_NAMES))
                labels.append(1.0 if _to_float(item["pnl_abs"]) > 0 else 0.0)
                pnl_values.append(_to_float(item["pnl_abs"]))
            live_samples = len(rows)

        if source in {"backtest", "mixed"}:
            bt_qs = BacktestTrade.objects.select_related("instrument", "run").filter(entry_ts__gte=from_backtest_dt)
            if symbols:
                bt_qs = bt_qs.filter(instrument__symbol__in=symbols)
            if backtest_run_ids:
                bt_qs = bt_qs.filter(run_id__in=backtest_run_ids)
            bt_rows = list(bt_qs.order_by("entry_ts", "id"))
            backtest_trades = len(bt_rows)

            for trade in bt_rows:
                direction, inferred_strategies = _backtest_direction_and_strategies(trade.side)
                if strategies_set and strategies_set.isdisjoint(inferred_strategies):
                    continue
                session = _session_from_datetime(trade.entry_ts)
                run_settings = trade.run.settings_json if isinstance(trade.run.settings_json, dict) else {}
                risk_budget = _to_float(run_settings.get("RISK_PER_TRADE_PCT"))
                if risk_budget <= 0:
                    risk_budget = _to_float(getattr(settings, "RISK_PER_TRADE_PCT", 0.0))
                score = _to_float(trade.score)
                payload = {
                    "direction": direction,
                    "session": session,
                    "confidence": _clamp01(score),
                    "raw_score": score,
                    "net_score": score,
                    "risk_budget_pct": max(0.0, risk_budget),
                    "reasons": {
                        "session": session,
                        "active_module_count": 0,
                        "module_rows": [],
                    },
                }
                feature_map = build_entry_feature_map(
                    strategy_name=f"smc_{direction}",
                    symbol=trade.instrument.symbol,
                    sig_score=score,
                    payload=payload,
                )
                rows.append(vectorize_feature_map(feature_map, feature_names=FEATURE_NAMES))
                labels.append(1.0 if _to_float(trade.pnl_abs) > 0 else 0.0)
                pnl_values.append(_to_float(trade.pnl_abs))
            backtest_samples = len(rows) - live_samples

        total_samples = len(rows)
        if total_samples < min_samples:
            raise CommandError(
                f"Not enough samples ({total_samples}) for min-samples={min_samples}. "
                f"live={live_samples}, backtest={backtest_samples}."
            )

        x = np.vstack(rows).astype(np.float64)
        y = np.asarray(labels, dtype=np.float64)
        pnl_arr = np.asarray(pnl_values, dtype=np.float64)
        if np.unique(y).size < 2:
            raise CommandError("Training set must contain both winning and losing samples")

        threshold_analysis: dict[str, Any] | None = None
        threshold_meta: dict[str, Any] = {
            "selection_scope": "in_sample",
            "selection_train_samples": int(total_samples),
            "selection_eval_samples": int(total_samples),
        }
        split_idx = _choose_holdout_split(y)
        if split_idx is not None:
            x_train = x[:split_idx]
            y_train = y[:split_idx]
            x_holdout = x[split_idx:]
            y_holdout = y[split_idx:]
            pnl_holdout = pnl_arr[split_idx:]
            selection_epochs = max(200, min(epochs, 600))
            try:
                selection_model = fit_logistic_model(
                    x_train,
                    y_train,
                    feature_names=FEATURE_NAMES,
                    epochs=selection_epochs,
                    learning_rate=lr,
                    l2=l2,
                )
                threshold_analysis = _analyze_thresholds(
                    selection_model,
                    x_holdout,
                    y_holdout,
                    pnl_holdout,
                )
                threshold_meta = {
                    "selection_scope": "holdout_tail",
                    "selection_train_samples": int(y_train.size),
                    "selection_eval_samples": int(y_holdout.size),
                }
            except Exception as exc:
                threshold_meta = {
                    "selection_scope": "in_sample_fallback",
                    "selection_train_samples": int(total_samples),
                    "selection_eval_samples": int(total_samples),
                    "selection_error": str(exc)[:160],
                }

        model = fit_logistic_model(
            x,
            y,
            feature_names=FEATURE_NAMES,
            epochs=epochs,
            learning_rate=lr,
            l2=l2,
        )
        if threshold_analysis is None:
            threshold_analysis = _analyze_thresholds(model, x, y, pnl_arr)
        threshold_analysis.update(threshold_meta)
        model["threshold_analysis"] = threshold_analysis

        if source == "live":
            window_from = from_live_dt
        elif source == "backtest":
            window_from = from_backtest_dt
        else:
            window_from = min(from_live_dt, from_backtest_dt)

        training_window: dict[str, Any] = {
            "source": source,
            "days": days,
            "backtest_days": backtest_days,
            "from": window_from.isoformat(),
            "from_live": from_live_dt.isoformat(),
            "from_backtest": from_backtest_dt.isoformat(),
            "to": now.isoformat(),
            "symbols": symbols,
            "strategies": strategies,
            "grouped_operations": grouped_operations,
            "skipped_missing_signal": skipped_missing_signal,
            "backtest_trades": backtest_trades,
            "samples_live": live_samples,
            "samples_backtest": backtest_samples,
            "samples_used": total_samples,
            "backtest_run_ids": backtest_run_ids,
        }
        model["training_window"] = training_window

        saved_path = save_model(model, output_path)
        rec_thr = _to_float((model.get("threshold_analysis") or {}).get("recommended_threshold"))
        rec_trades = int(_to_float((model.get("threshold_analysis") or {}).get("recommended_selected_trades")))
        self.stdout.write(
            self.style.SUCCESS(
                f"Trained entry filter model: source={source} samples={total_samples} "
                f"(live={live_samples}, backtest={backtest_samples}) recommended_threshold={rec_thr:.2f} "
                f"selected_trades={rec_trades} path={saved_path}"
            )
        )
