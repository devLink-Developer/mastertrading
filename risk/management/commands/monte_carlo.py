"""
Monte Carlo simulation for drawdown distribution and risk-of-ruin estimation.

Usage:
    python manage.py monte_carlo
    python manage.py monte_carlo --days 60 --sims 50000
    python manage.py monte_carlo --symbol BTCUSDT --regime-aware --stress-profile bear
    python manage.py monte_carlo --json reports/mc.json
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
from django.core.management.base import BaseCommand
from django.utils import timezone as dj_tz

from execution.models import OperationReport
from signals.models import Signal


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


class Command(BaseCommand):
    help = "Monte Carlo equity-curve simulation & risk-of-ruin from live trades"

    def add_arguments(self, parser):
        parser.add_argument("--days", type=int, default=30, help="Lookback window (days)")
        parser.add_argument("--symbol", type=str, default="", help="Filter by instrument (e.g. BTCUSDT)")
        parser.add_argument("--sims", type=int, default=10_000, help="Number of MC paths")
        parser.add_argument(
            "--ruin-threshold",
            type=float,
            default=20.0,
            help="Ruin defined as equity drop >= X%% from starting capital",
        )
        parser.add_argument("--horizon", type=int, default=0, help="Trades per path (0 = observed N)")
        parser.add_argument("--seed", type=int, default=42, help="RNG seed")
        parser.add_argument("--json", type=str, default="", help="Write results to JSON file")

        parser.add_argument(
            "--regime-aware",
            action="store_true",
            help="Use Markov regime transitions + regime-conditioned return sampling.",
        )
        parser.add_argument(
            "--stress-profile",
            type=str,
            default="none",
            choices=["none", "balanced", "bear"],
            help="Preset stress profile for regime-aware mode.",
        )
        parser.add_argument("--stress-vol-mult", type=float, default=0.0, help="Override abs-return multiplier.")
        parser.add_argument(
            "--stress-loss-cluster-mult",
            type=float,
            default=0.0,
            help="Override loss-after-loss multiplier.",
        )
        parser.add_argument(
            "--stress-corr-shock-prob",
            type=float,
            default=0.0,
            help="Override correlation shock probability per step.",
        )
        parser.add_argument(
            "--stress-corr-shock-size",
            type=float,
            default=0.0,
            help="Override additive negative shock size (fraction, e.g. 0.005 = -0.5%).",
        )
        parser.add_argument(
            "--stress-bear-bias",
            type=float,
            default=-1.0,
            help="Override transition/start bias toward bear-like states.",
        )
        parser.add_argument("--markov-alpha", type=float, default=1.0, help="Transition smoothing alpha.")
        parser.add_argument("--min-state-trades", type=int, default=8, help="Min trades per regime state.")

    def handle(self, **opts):
        rows = self._load_trade_rows(opts["days"], opts["symbol"])
        n_trades = len(rows)
        if n_trades < 10:
            self.stderr.write(self.style.ERROR(f"Only {n_trades} trades found - need >=10 for MC."))
            return

        returns = np.array([_safe_float(r.get("pnl_pct"), 0.0) for r in rows], dtype=float)
        horizon = opts["horizon"] or n_trades
        n_sims = int(opts["sims"])
        ruin_thr = float(opts["ruin_threshold"]) / 100.0
        rng = np.random.default_rng(int(opts["seed"]))

        regime_aware = bool(opts.get("regime_aware", False))
        stress = self._resolve_stress_params(opts)

        sym_label = f" ({opts['symbol']})" if opts["symbol"] else ""
        self.stdout.write(
            self.style.NOTICE(
                "\n"
                + "=" * 62
                + f"\n  Monte Carlo - {n_sims:,} paths x {horizon} trades\n"
                + f"  Source: {n_trades} live trades{sym_label}, last {opts['days']}d\n"
                + f"  Ruin threshold: {opts['ruin_threshold']:.1f}%\n"
                + f"  Regime aware: {'yes' if regime_aware else 'no'}"
                + f"\n  Stress profile: {stress['profile']}"
                + "\n"
                + "=" * 62
            )
        )

        if regime_aware:
            max_dds, final_pnls, ruin_flags, regime_summary = self._simulate_regime_aware(
                rows=rows,
                n_sims=n_sims,
                horizon=horizon,
                ruin_thr=ruin_thr,
                rng=rng,
                stress=stress,
                markov_alpha=max(0.0, float(opts["markov_alpha"])),
                min_state_trades=max(1, int(opts["min_state_trades"])),
            )
        else:
            max_dds, final_pnls, ruin_flags = self._simulate(
                returns=returns,
                n_sims=n_sims,
                horizon=horizon,
                ruin_thr=ruin_thr,
                rng=rng,
            )
            regime_summary = {}

        obs = self._observed_stats(list(returns))
        mc = self._mc_stats(max_dds, final_pnls, ruin_flags, ruin_thr)

        self._print_observed(obs)
        self._print_mc(mc, opts["ruin_threshold"], stress, regime_summary)

        if opts["json"]:
            self._write_json(
                path=opts["json"],
                obs=obs,
                mc=mc,
                opts=opts,
                stress=stress,
                regime_summary=regime_summary,
            )

    def _resolve_stress_params(self, opts: dict[str, Any]) -> dict[str, Any]:
        profile = str(opts.get("stress_profile") or "none").strip().lower()
        if profile not in {"none", "balanced", "bear"}:
            profile = "none"
        presets = {
            "none": {
                "vol_mult": 1.0,
                "loss_cluster_mult": 1.0,
                "corr_shock_prob": 0.0,
                "corr_shock_size": 0.0,
                "bear_bias": 0.0,
            },
            "balanced": {
                "vol_mult": 1.15,
                "loss_cluster_mult": 1.20,
                "corr_shock_prob": 0.01,
                "corr_shock_size": 0.004,
                "bear_bias": 0.05,
            },
            "bear": {
                "vol_mult": 1.35,
                "loss_cluster_mult": 1.45,
                "corr_shock_prob": 0.03,
                "corr_shock_size": 0.008,
                "bear_bias": 0.12,
            },
        }
        stress = dict(presets[profile])
        if _safe_float(opts.get("stress_vol_mult"), 0.0) > 0:
            stress["vol_mult"] = max(0.1, _safe_float(opts.get("stress_vol_mult"), 1.0))
        if _safe_float(opts.get("stress_loss_cluster_mult"), 0.0) > 0:
            stress["loss_cluster_mult"] = max(0.1, _safe_float(opts.get("stress_loss_cluster_mult"), 1.0))
        if _safe_float(opts.get("stress_corr_shock_prob"), 0.0) >= 0:
            raw = _safe_float(opts.get("stress_corr_shock_prob"), 0.0)
            if raw > 0:
                stress["corr_shock_prob"] = min(1.0, raw)
        if _safe_float(opts.get("stress_corr_shock_size"), 0.0) > 0:
            stress["corr_shock_size"] = min(0.30, _safe_float(opts.get("stress_corr_shock_size"), 0.0))
        bear_bias = _safe_float(opts.get("stress_bear_bias"), -1.0)
        if bear_bias >= 0:
            stress["bear_bias"] = min(1.0, bear_bias)
        stress["profile"] = profile
        return stress

    def _extract_regime(self, sig_payload: dict[str, Any]) -> str:
        if not isinstance(sig_payload, dict):
            return "unknown"
        if isinstance(sig_payload.get("regime"), str):
            txt = str(sig_payload.get("regime") or "").strip().lower()
            if txt:
                return txt
        reasons = sig_payload.get("reasons")
        if isinstance(reasons, dict):
            txt = str(reasons.get("regime") or "").strip().lower()
            if txt:
                return txt
        return "unknown"

    def _load_trade_rows(self, days: int, symbol: str) -> list[dict[str, Any]]:
        cutoff = dj_tz.now() - timedelta(days=max(1, int(days)))
        qs = OperationReport.objects.filter(closed_at__gte=cutoff).order_by("closed_at")
        if symbol:
            qs = qs.filter(instrument__symbol__iexact=symbol)
        rows = list(qs.values("pnl_pct", "closed_at", "signal_id"))
        if not rows:
            return []

        sig_ids: set[int] = set()
        for row in rows:
            sid = str(row.get("signal_id") or "").strip()
            if sid.isdigit():
                sig_ids.add(int(sid))
        sig_map: dict[int, dict[str, Any]] = {}
        if sig_ids:
            for sig in Signal.objects.filter(id__in=sig_ids).values("id", "payload_json"):
                sig_map[int(sig["id"])] = sig.get("payload_json") or {}

        out: list[dict[str, Any]] = []
        for row in rows:
            sid = str(row.get("signal_id") or "").strip()
            payload = sig_map.get(int(sid), {}) if sid.isdigit() else {}
            out.append(
                {
                    "pnl_pct": _safe_float(row.get("pnl_pct"), 0.0),
                    "closed_at": row.get("closed_at"),
                    "regime": self._extract_regime(payload),
                }
            )
        return out

    @staticmethod
    def _simulate(
        returns: np.ndarray,
        n_sims: int,
        horizon: int,
        ruin_thr: float,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        idx = rng.integers(0, len(returns), size=(n_sims, horizon))
        sampled = returns[idx]
        equity = np.cumprod(1.0 + sampled, axis=1)
        peak = np.maximum.accumulate(equity, axis=1)
        dd = (equity - peak) / peak
        max_dds = dd.min(axis=1)
        final_pnls = equity[:, -1] - 1.0
        ruin_flags = (max_dds <= -ruin_thr).astype(int)
        return max_dds, final_pnls, ruin_flags

    def _simulate_regime_aware(
        self,
        *,
        rows: list[dict[str, Any]],
        n_sims: int,
        horizon: int,
        ruin_thr: float,
        rng: np.random.Generator,
        stress: dict[str, Any],
        markov_alpha: float,
        min_state_trades: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        global_returns = np.array([_safe_float(r["pnl_pct"], 0.0) for r in rows], dtype=float)
        regimes = [str(r.get("regime") or "unknown").strip().lower() or "unknown" for r in rows]

        by_regime: dict[str, list[float]] = defaultdict(list)
        for r, reg in zip(global_returns, regimes):
            by_regime[reg].append(float(r))

        major_states = {k for k, vals in by_regime.items() if len(vals) >= min_state_trades}
        states = sorted(major_states) if major_states else ["unknown"]
        if "unknown" not in states:
            states.append("unknown")

        prepared: list[str] = []
        for reg in regimes:
            prepared.append(reg if reg in major_states else "unknown")

        returns_by_state: dict[str, np.ndarray] = {}
        for st in states:
            arr = np.array(by_regime.get(st, []), dtype=float)
            if len(arr) == 0:
                arr = global_returns
            returns_by_state[st] = arr

        state_idx = {s: i for i, s in enumerate(states)}
        n_states = len(states)
        trans = np.zeros((n_states, n_states), dtype=float)
        for i in range(len(prepared) - 1):
            a = state_idx[prepared[i]]
            b = state_idx[prepared[i + 1]]
            trans[a, b] += 1.0
        trans = trans + max(0.0, markov_alpha)
        trans = trans / trans.sum(axis=1, keepdims=True)

        start_counts = Counter(prepared)
        start = np.array([start_counts.get(s, 0) for s in states], dtype=float)
        if float(start.sum()) <= 0:
            start = np.ones_like(start)
        start = start / start.sum()

        bear_states = [s for s in states if ("bear" in s or "choppy" in s)]
        bear_bias = max(0.0, _safe_float(stress.get("bear_bias"), 0.0))
        if bear_states and bear_bias > 0:
            bear_idx = [state_idx[s] for s in bear_states]
            start[bear_idx] += bear_bias
            start = start / start.sum()
            for i in range(n_states):
                trans[i, bear_idx] += bear_bias
                trans[i, :] = trans[i, :] / trans[i, :].sum()

        loss_after_loss = 0
        loss_total = 0
        for i in range(1, len(global_returns)):
            if global_returns[i - 1] < 0:
                loss_total += 1
                if global_returns[i] < 0:
                    loss_after_loss += 1
        base_ll_prob = (loss_after_loss / loss_total) if loss_total > 0 else 0.0
        ll_prob = min(0.95, base_ll_prob * max(0.1, _safe_float(stress.get("loss_cluster_mult"), 1.0)))

        vol_mult = max(0.1, _safe_float(stress.get("vol_mult"), 1.0))
        corr_shock_prob = _safe_float(stress.get("corr_shock_prob"), 0.0)
        corr_shock_size = max(0.0, _safe_float(stress.get("corr_shock_size"), 0.0))

        max_dds = np.zeros(n_sims, dtype=float)
        final_pnls = np.zeros(n_sims, dtype=float)
        ruin_flags = np.zeros(n_sims, dtype=int)

        for path in range(n_sims):
            eq = 1.0
            peak = 1.0
            min_dd = 0.0
            st_i = int(rng.choice(np.arange(n_states), p=start))
            prev_ret = 0.0

            for _ in range(horizon):
                st_name = states[st_i]
                pool = returns_by_state.get(st_name, global_returns)
                ret = float(rng.choice(pool))
                if prev_ret < 0 and ll_prob > 0 and rng.random() < ll_prob:
                    neg_pool = pool[pool < 0]
                    if len(neg_pool) > 0:
                        ret = float(rng.choice(neg_pool))

                ret = float(np.sign(ret) * min(0.95, abs(ret) * vol_mult))
                if corr_shock_prob > 0 and corr_shock_size > 0 and rng.random() < corr_shock_prob:
                    ret -= corr_shock_size
                ret = max(-0.95, ret)

                eq *= (1.0 + ret)
                if eq > peak:
                    peak = eq
                dd = (eq - peak) / peak
                if dd < min_dd:
                    min_dd = dd
                prev_ret = ret

                st_i = int(rng.choice(np.arange(n_states), p=trans[st_i]))

            max_dds[path] = min_dd
            final_pnls[path] = eq - 1.0
            ruin_flags[path] = 1 if min_dd <= -ruin_thr else 0

        summary = {
            "states": states,
            "state_counts": dict(Counter(prepared)),
            "loss_after_loss_base_prob": round(base_ll_prob, 6),
            "loss_after_loss_stress_prob": round(ll_prob, 6),
        }
        return max_dds, final_pnls, ruin_flags, summary

    @staticmethod
    def _observed_stats(pnl_pcts: list[float]) -> dict[str, Any]:
        arr = np.array(pnl_pcts, dtype=float)
        wins = arr[arr > 0]
        losses = arr[arr < 0]
        wr = len(wins) / len(arr) if len(arr) else 0
        avg_w = float(wins.mean()) if len(wins) else 0
        avg_l = float(losses.mean()) if len(losses) else 0
        payoff = abs(avg_w / avg_l) if avg_l != 0 else 0
        expectancy = wr * avg_w + (1 - wr) * avg_l
        kelly = (wr - (1 - wr) / payoff) if payoff > 0 else 0.0
        eq = np.cumprod(1.0 + arr)
        peak = np.maximum.accumulate(eq)
        obs_max_dd = float(((eq - peak) / peak).min())
        return {
            "n_trades": len(arr),
            "win_rate": round(wr, 4),
            "avg_win_pct": round(avg_w * 100, 4),
            "avg_loss_pct": round(avg_l * 100, 4),
            "payoff_ratio": round(payoff, 4),
            "expectancy_pct": round(expectancy * 100, 4),
            "kelly_fraction": round(kelly, 4),
            "total_return_pct": round(float((eq[-1] - 1) * 100), 4) if len(eq) else 0,
            "observed_max_dd_pct": round(obs_max_dd * 100, 2),
        }

    @staticmethod
    def _mc_stats(
        max_dds: np.ndarray,
        final_pnls: np.ndarray,
        ruin_flags: np.ndarray,
        ruin_thr: float,
    ) -> dict[str, Any]:
        dd_pcts = max_dds * 100
        ret_pcts = final_pnls * 100
        pctiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        return {
            "risk_of_ruin_pct": round(float(ruin_flags.mean()) * 100, 2),
            "ruin_threshold_pct": round(ruin_thr * 100, 1),
            "max_dd_percentiles": {f"p{p}": round(float(np.percentile(dd_pcts, p)), 2) for p in pctiles},
            "final_return_percentiles": {f"p{p}": round(float(np.percentile(ret_pcts, p)), 2) for p in pctiles},
            "mean_max_dd_pct": round(float(dd_pcts.mean()), 2),
            "std_max_dd_pct": round(float(dd_pcts.std()), 2),
            "mean_final_return_pct": round(float(ret_pcts.mean()), 2),
            "std_final_return_pct": round(float(ret_pcts.std()), 2),
        }

    def _print_observed(self, obs: dict[str, Any]) -> None:
        self.stdout.write(self.style.SUCCESS("\n--- Observed Trade Stats ---"))
        self.stdout.write(f"  Trades:          {obs['n_trades']}")
        self.stdout.write(f"  Win rate:        {obs['win_rate']:.1%}")
        self.stdout.write(f"  Avg win:         {obs['avg_win_pct']:+.3f}%")
        self.stdout.write(f"  Avg loss:        {obs['avg_loss_pct']:+.3f}%")
        self.stdout.write(f"  Payoff ratio:    {obs['payoff_ratio']:.3f}")
        self.stdout.write(f"  Expectancy:      {obs['expectancy_pct']:+.4f}% per trade")
        self.stdout.write(f"  Kelly fraction:  {obs['kelly_fraction']:+.4f}")
        self.stdout.write(f"  Total return:    {obs['total_return_pct']:+.2f}%")
        self.stdout.write(f"  Max drawdown:    {obs['observed_max_dd_pct']:.2f}%")

    def _print_mc(
        self,
        mc: dict[str, Any],
        ruin_thr: float,
        stress: dict[str, Any],
        regime_summary: dict[str, Any],
    ) -> None:
        self.stdout.write(self.style.SUCCESS("\n--- Monte Carlo Results ---"))
        self.stdout.write(f"  Risk of ruin ({ruin_thr:.0f}% DD): {mc['risk_of_ruin_pct']:.2f}%")
        self.stdout.write(
            "  Stress: "
            f"profile={stress.get('profile')} vol_mult={_safe_float(stress.get('vol_mult'),1.0):.2f} "
            f"loss_cluster_mult={_safe_float(stress.get('loss_cluster_mult'),1.0):.2f} "
            f"corr_shock_prob={_safe_float(stress.get('corr_shock_prob'),0.0):.3f}"
        )

        self.stdout.write("\n  Max Drawdown distribution:")
        for k, v in mc["max_dd_percentiles"].items():
            self.stdout.write(f"    {k:>4}: {v:+.2f}%")

        self.stdout.write("\n  Final Return distribution:")
        for k, v in mc["final_return_percentiles"].items():
            self.stdout.write(f"    {k:>4}: {v:+.2f}%")

        if regime_summary:
            self.stdout.write("\n  Regime summary:")
            self.stdout.write(f"    states: {', '.join(regime_summary.get('states', []))}")
            self.stdout.write(
                "    loss-after-loss base/stress: "
                f"{regime_summary.get('loss_after_loss_base_prob', 0):.3f}/"
                f"{regime_summary.get('loss_after_loss_stress_prob', 0):.3f}"
            )

        self.stdout.write(
            f"\n  Mean max DD:  {mc['mean_max_dd_pct']:+.2f}% +/- {mc['std_max_dd_pct']:.2f}%"
        )
        self.stdout.write(
            f"  Mean return:  {mc['mean_final_return_pct']:+.2f}% +/- {mc['std_final_return_pct']:.2f}%"
        )
        self.stdout.write("")

    def _write_json(
        self,
        *,
        path: str,
        obs: dict[str, Any],
        mc: dict[str, Any],
        opts: dict[str, Any],
        stress: dict[str, Any],
        regime_summary: dict[str, Any],
    ) -> None:
        out = {
            "params": {
                "days": opts["days"],
                "symbol": opts["symbol"] or "ALL",
                "sims": opts["sims"],
                "horizon": opts["horizon"] or obs["n_trades"],
                "ruin_threshold_pct": opts["ruin_threshold"],
                "seed": opts["seed"],
                "regime_aware": bool(opts.get("regime_aware", False)),
            },
            "stress": stress,
            "regime_summary": regime_summary,
            "observed": obs,
            "monte_carlo": mc,
        }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=2), encoding="utf-8")
        self.stdout.write(self.style.SUCCESS(f"Results saved -> {p}"))
