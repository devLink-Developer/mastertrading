"""
Backtest engine – replays historical candles through the signal engine
and simulates execution with all risk-management rules.

Usage:
    python manage.py backtest --start 2025-01-01 --end 2025-06-01

Architecture:
    1. Load ALL candles for the period into memory (pandas DataFrames).
    2. Walk forward bar-by-bar on the LTF (default 5m, supports 1m for
       higher-fidelity signal_flip simulation matching live 60s cadence).
    3. At each bar, slice the candle history up to "now" and run the
       same _detect_signal() function used in production.
    4. A SimulatedPosition tracks open trades with TP/SL/trailing stop.
    5. At the end, compute summary metrics.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from django.conf import settings
from django.utils import timezone as dj_tz

from core.models import Instrument
from marketdata.models import Candle, FundingRate
from signals.sessions import (
    get_current_session,
    get_session_risk_mult,
    get_session_score_min,
    is_dead_session,
)
from signals.direction_policy import (
    get_direction_mode,
    is_direction_allowed,
)
from signals.tasks import (
    _detect_signal,
    _trend_from_swings,
)
from signals.modules.trend import detect as trend_detect
from signals.modules.meanrev import detect as meanrev_detect
from signals.modules.carry import detect as carry_detect
from signals.allocator import (
    default_weight_map,
    default_risk_budget_map,
    resolve_symbol_allocation,
)
from signals.modules.common import compute_adx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers ported from execution/tasks.py for backtest parity
# ---------------------------------------------------------------------------

def _max_daily_trades_for_adx(htf_adx: float | None) -> int:
    """Regime-adaptive trade throttle (mirrors execution/tasks.py)."""
    low_adx_limit = int(getattr(settings, "MAX_DAILY_TRADES_LOW_ADX", 3))
    normal_limit = int(getattr(settings, "MAX_DAILY_TRADES", 6))
    high_adx_limit = int(getattr(settings, "MAX_DAILY_TRADES_HIGH_ADX", 10))
    if htf_adx is None:
        return normal_limit
    if htf_adx < 20:
        return low_adx_limit
    if htf_adx > 25:
        return high_adx_limit
    return normal_limit


def _volatility_adjusted_risk(
    symbol: str, atr_pct: float | None, base_risk: float,
) -> float:
    """Scale risk inversely with vol (mirrors execution/tasks.py)."""
    per_inst = getattr(settings, "PER_INSTRUMENT_RISK", {})
    if symbol in per_inst:
        return float(per_inst[symbol])

    effective_base = base_risk
    if getattr(settings, "INSTRUMENT_RISK_TIERS_ENABLED", False):
        tier_map = getattr(settings, "INSTRUMENT_TIER_MAP", {})
        tiers = getattr(settings, "INSTRUMENT_RISK_TIERS", {})
        tier_name = tier_map.get(symbol, "")
        if tier_name and tier_name in tiers:
            effective_base = float(tiers[tier_name])

    if atr_pct is None or atr_pct <= 0:
        return effective_base

    low_vol = 0.008
    high_vol = 0.015
    min_scale = 0.6
    if atr_pct <= low_vol:
        return effective_base
    if atr_pct >= high_vol:
        return effective_base * min_scale
    ratio = (atr_pct - low_vol) / (high_vol - low_vol)
    scale = 1.0 - ratio * (1.0 - min_scale)
    return effective_base * scale

# ---------------------------------------------------------------------------
# Fee model
# ---------------------------------------------------------------------------
TAKER_FEE_BPS = 4  # 0.04 % taker fee (VIP/BNB discount tier)
MAKER_FEE_BPS = 2  # 0.02 %


def _fee_pct() -> float:
    """Return one-way taker fee as a fraction (e.g. 0.0006)."""
    return TAKER_FEE_BPS / 10_000


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_candles(
    instrument: Instrument,
    tf: str,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """Load candles from DB into a pandas DataFrame sorted chronologically."""
    qs = (
        Candle.objects.filter(
            instrument=instrument,
            timeframe=tf,
            ts__gte=start,
            ts__lte=end,
        )
        .order_by("ts")
        .values("ts", "open", "high", "low", "close", "volume")
    )
    if not qs.exists():
        return pd.DataFrame()
    df = pd.DataFrame(list(qs))
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)
    df.set_index("ts", inplace=True)
    return df


def load_funding_rates(
    instrument: Instrument,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    qs = (
        FundingRate.objects.filter(
            instrument=instrument,
            ts__gte=start,
            ts__lte=end,
        )
        .order_by("ts")
        .values("ts", "rate")
    )
    if not qs.exists():
        return pd.DataFrame()
    df = pd.DataFrame(list(qs))
    df["rate"] = df["rate"].astype(float)
    df.set_index("ts", inplace=True)
    return df


# ---------------------------------------------------------------------------
# ATR helper (pure pandas, no DB calls)
# ---------------------------------------------------------------------------

def _atr_pct_df(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    """Compute ATR % from a candle DataFrame (must be chronologically sorted)."""
    if len(df) < period + 1:
        return None
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    trs = []
    for i in range(1, len(df)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)
    atr = np.mean(trs[-period:])
    last_close = closes[-1]
    if last_close == 0:
        return None
    return float(atr / last_close)


# ---------------------------------------------------------------------------
# Simulated position
# ---------------------------------------------------------------------------

@dataclass
class SimTrade:
    """Record of a completed simulated trade."""
    instrument_id: int
    side: str
    qty: float
    entry_price: float
    exit_price: float
    entry_ts: datetime
    exit_ts: datetime
    pnl_abs: float
    pnl_pct: float
    fee_paid: float
    reason: str
    score: float


@dataclass
class SimPosition:
    """Tracks an open simulated position."""
    instrument_id: int
    side: str  # buy / sell
    qty: float
    entry_price: float
    entry_ts: datetime
    sl_price: float
    tp_price: float
    sl_pct: float
    tp_pct: float
    score: float
    partial_closed: bool = False
    trailing_activated: bool = False
    max_favorable: float = 0.0  # track high water mark for trailing

    def check_exit(self, bar: pd.Series, bar_ts: datetime, trailing_enabled: bool) -> list[SimTrade]:
        """
        Check if current bar triggers exit (partial close, SL, TP, trailing stop).
        Returns list of SimTrade (empty=no exit, 1=partial or full, 2=partial+full).
        After a full close self.qty is set to 0.
        """
        results: list[SimTrade] = []
        high = bar["high"]
        low = bar["low"]
        close = bar["close"]
        is_long = self.side == "buy"

        # Track high water mark (max favorable excursion)
        if is_long:
            current_fav = (high - self.entry_price) / self.entry_price
        else:
            current_fav = (self.entry_price - low) / self.entry_price
        self.max_favorable = max(self.max_favorable, current_fav)

        # -- Partial close: take profit on a fraction at R threshold --
        if not self.partial_closed and self.sl_pct > 0:
            partial_r = float(getattr(settings, "PARTIAL_CLOSE_AT_R", 0.8))
            partial_pct = float(getattr(settings, "PARTIAL_CLOSE_PCT", 0.5))
            r_mult = self.max_favorable / self.sl_pct
            if r_mult >= partial_r and self.qty > 0:
                close_qty = round(self.qty * partial_pct, 6)
                remaining = round(self.qty - close_qty, 6)
                if close_qty > 0 and remaining > 0:
                    # Fill at the R-threshold price (conservative estimate)
                    r_dist = partial_r * self.sl_pct
                    partial_price = (
                        self.entry_price * (1 + r_dist) if is_long
                        else self.entry_price * (1 - r_dist)
                    )
                    results.append(self._close(partial_price, bar_ts, "partial_close", qty=close_qty))
                    self.qty = remaining
                    self.partial_closed = True

        # -- Breakeven stop: after X R in profit, move SL to entry (or slight buffer) --
        if getattr(settings, "BREAKEVEN_STOP_ENABLED", True) and self.sl_pct > 0:
            be_at_r = float(getattr(settings, "BREAKEVEN_STOP_AT_R", 1.0) or 0.0)
            if be_at_r > 0:
                be_window_min = int(getattr(settings, "BREAKEVEN_WINDOW_MINUTES", 0) or 0)
                be_allowed = True
                if be_window_min > 0:
                    try:
                        age_min = (bar_ts - self.entry_ts).total_seconds() / 60.0
                        be_allowed = age_min <= be_window_min
                    except Exception:
                        be_allowed = False

                if be_allowed:
                    r_mult = self.max_favorable / self.sl_pct
                    if r_mult >= be_at_r:
                        be_offset = float(getattr(settings, "BREAKEVEN_STOP_OFFSET_PCT", 0.0) or 0.0)
                        be_offset = max(0.0, be_offset)
                        be_price = self.entry_price * (1 + be_offset) if is_long else self.entry_price * (1 - be_offset)
                        if is_long and be_price > self.sl_price:
                            self.sl_price = be_price
                        if (not is_long) and be_price < self.sl_price:
                            self.sl_price = be_price

        # -- Trailing stop logic: activate at R threshold, then lock a fraction of the HWM profit --
        if trailing_enabled:
            if not self.trailing_activated:
                r_mult = self.max_favorable / self.sl_pct if self.sl_pct > 0 else 0
                if r_mult >= settings.TRAILING_STOP_ACTIVATION_R:
                    self.trailing_activated = True

            if self.trailing_activated:
                lock_in = float(getattr(settings, "TRAILING_STOP_LOCK_IN_PCT", 0.5) or 0.5)
                lock_in = max(0.0, min(lock_in, 1.0))
                if is_long:
                    new_sl = self.entry_price * (1 + self.max_favorable * lock_in)
                    if new_sl > self.sl_price:
                        self.sl_price = new_sl
                else:
                    new_sl = self.entry_price * (1 - self.max_favorable * lock_in)
                    if new_sl < self.sl_price:
                        self.sl_price = new_sl

        # -- SL hit check (intra-bar: use low for longs, high for shorts) --
        if is_long and low <= self.sl_price:
            exit_price = self.sl_price  # assume fill at SL level
            reason = "trailing_stop" if self.trailing_activated else "sl"
            results.append(self._close(exit_price, bar_ts, reason))
            self.qty = 0
            return results

        if not is_long and high >= self.sl_price:
            exit_price = self.sl_price
            reason = "trailing_stop" if self.trailing_activated else "sl"
            results.append(self._close(exit_price, bar_ts, reason))
            self.qty = 0
            return results

        # -- TP hit check --
        if is_long and high >= self.tp_price:
            results.append(self._close(self.tp_price, bar_ts, "tp"))
            self.qty = 0
            return results

        if not is_long and low <= self.tp_price:
            results.append(self._close(self.tp_price, bar_ts, "tp"))
            self.qty = 0
            return results

        return results

    def force_close(self, price: float, ts: datetime, reason: str = "end_of_data") -> SimTrade:
        """Force-close the position at the given price."""
        trade = self._close(price, ts, reason)
        self.qty = 0
        return trade

    def _close(self, exit_price: float, ts: datetime, reason: str, qty: Optional[float] = None) -> SimTrade:
        qty = qty if qty is not None else self.qty
        is_long = self.side == "buy"
        direction = 1 if is_long else -1
        gross_pnl = (exit_price - self.entry_price) * qty * direction
        # Fee: entry + exit
        fee = (self.entry_price * qty + exit_price * qty) * _fee_pct()
        net_pnl = gross_pnl - fee
        pnl_pct = (exit_price - self.entry_price) / self.entry_price * direction
        return SimTrade(
            instrument_id=self.instrument_id,
            side=self.side,
            qty=qty,
            entry_price=self.entry_price,
            exit_price=exit_price,
            entry_ts=self.entry_ts,
            exit_ts=ts,
            pnl_abs=net_pnl,
            pnl_pct=pnl_pct,
            fee_paid=fee,
            reason=reason,
            score=self.score,
        )


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def compute_metrics(trades: list[SimTrade], initial_equity: float, **kwargs) -> dict:
    """Compute strategy performance metrics from a list of simulated trades.

    Keyword args:
        bars_per_day: int — used for Sharpe annualization (288 for 5m, 1440 for 1m).
    """
    if not trades:
        return {
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "total_pnl_pct": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "best_trade": 0.0,
            "worst_trade": 0.0,
            "profit_factor": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "max_drawdown_abs": 0.0,
            "avg_trade_duration_min": 0.0,
            "total_fees": 0.0,
            "expectancy": 0.0,
            "final_equity": initial_equity,
            "return_pct": 0.0,
        }

    pnls = [t.pnl_abs for t in trades]
    pnl_pcts = [t.pnl_pct for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    total_pnl = sum(pnls)
    total_fees = sum(t.fee_paid for t in trades)

    # Equity curve for drawdown
    equity_curve = [initial_equity]
    for pnl in pnls:
        equity_curve.append(equity_curve[-1] + pnl)
    equity_arr = np.array(equity_curve)
    running_max = np.maximum.accumulate(equity_arr)
    drawdowns = (equity_arr - running_max)
    max_dd_abs = float(np.min(drawdowns))
    max_dd_pct = float(np.min(drawdowns / running_max)) if running_max.max() > 0 else 0.0

    # Sharpe (daily approximation: bars_per_day based on ltf)
    bars_per_day = kwargs.get("bars_per_day", 288)  # default 288 for 5m
    if len(pnl_pcts) > 1:
        arr = np.array(pnl_pcts)
        sharpe = float(np.mean(arr) / np.std(arr) * np.sqrt(bars_per_day)) if np.std(arr) > 0 else 0.0
    else:
        sharpe = 0.0

    # Profit factor
    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0

    # Average trade duration
    durations = [(t.exit_ts - t.entry_ts).total_seconds() / 60 for t in trades]
    avg_duration = np.mean(durations) if durations else 0.0

    final_equity = equity_curve[-1]

    return {
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "breakeven": len(trades) - len(wins) - len(losses),
        "win_rate": round(len(wins) / len(trades) * 100, 2),
        "total_pnl": round(total_pnl, 4),
        "total_pnl_pct": round(total_pnl / initial_equity * 100, 2) if initial_equity > 0 else 0.0,
        "avg_win": round(np.mean(wins), 4) if wins else 0.0,
        "avg_loss": round(np.mean(losses), 4) if losses else 0.0,
        "best_trade": round(max(pnls), 4),
        "worst_trade": round(min(pnls), 4),
        "profit_factor": round(pf, 3) if pf != float("inf") else "inf",
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown_pct": round(abs(max_dd_pct) * 100, 2),
        "max_drawdown_abs": round(abs(max_dd_abs), 4),
        "avg_trade_duration_min": round(float(avg_duration), 1),
        "total_fees": round(total_fees, 4),
        "expectancy": round(total_pnl / len(trades), 4),
        "final_equity": round(final_equity, 2),
        "return_pct": round((final_equity - initial_equity) / initial_equity * 100, 2) if initial_equity > 0 else 0.0,
        "initial_equity": round(initial_equity, 2),
    }


# ---------------------------------------------------------------------------
# Main backtest runner
# ---------------------------------------------------------------------------

def run_backtest(
    instruments: list[Instrument],
    start: datetime,
    end: datetime,
    initial_equity: float = 1000.0,
    ltf: str = "5m",
    htf: str = "4h",
    lookback_bars: int = 240,
    trailing_stop: bool = True,
    verbose: bool = False,
) -> tuple[list[SimTrade], dict]:
    """
    Run a walk-forward backtest over the given instruments and date range.

    For each 5m bar:
        1. Build the "known so far" slice of LTF and HTF candles
        2. Run _detect_signal() from the signal engine
        3. If we have an open position, check TP/SL/trailing
        4. If we get a new signal, open a position (risk-based sizing)

    Returns (trades, metrics).
    """
    t0 = time.time()

    # Preload all data into memory
    candle_cache: dict[int, dict[str, pd.DataFrame]] = {}
    funding_cache: dict[int, pd.DataFrame] = {}

    # We need candles BEFORE `start` for lookback context
    lookback_start = start - timedelta(days=30)  # generous buffer

    for inst in instruments:
        candle_cache[inst.id] = {}
        for tf in (ltf, htf, "1h"):  # Always load 1h for trend/meanrev modules
            if tf in candle_cache[inst.id]:
                continue  # avoid duplicate loads if htf == "1h"
            df = load_candles(inst, tf, lookback_start, end)
            candle_cache[inst.id][tf] = df
            if verbose:
                logger.info("Loaded %d %s candles for %s", len(df), tf, inst.symbol)

        funding_cache[inst.id] = load_funding_rates(inst, lookback_start, end)

    # Build the bar timeline from LTF candles across all instruments
    all_timestamps = set()
    for inst in instruments:
        df_ltf = candle_cache[inst.id].get(ltf, pd.DataFrame())
        if not df_ltf.empty:
            # Only bars within the test window
            mask = df_ltf.index >= start
            all_timestamps.update(df_ltf.index[mask].tolist())

    if not all_timestamps:
        logger.warning("No candle data found for the given date range")
        return [], compute_metrics([], initial_equity)

    timeline = sorted(all_timestamps)
    logger.info(
        "Backtest timeline: %d bars from %s to %s (%d instruments)",
        len(timeline), timeline[0], timeline[-1], len(instruments),
    )

    # State
    equity = initial_equity
    positions: dict[int, SimPosition] = {}  # instrument_id → SimPosition
    trades: list[SimTrade] = []
    session_skips_dead = 0
    session_skips_score = 0
    direction_skips = 0
    session_open_counts: dict[str, int] = {}
    session_policy_enabled = bool(getattr(settings, "SESSION_POLICY_ENABLED", False))
    session_dead_zone_block = bool(getattr(settings, "SESSION_DEAD_ZONE_BLOCK", True))
    session_score_overrides = getattr(settings, "SESSION_SCORE_MIN", {})
    session_risk_overrides = getattr(settings, "SESSION_RISK_MULTIPLIER", {})
    signal_cooldown: dict[int, datetime] = {}  # inst_id → earliest next entry time

    # -- Regime ADX gate (per-instrument, mirrors live) --
    regime_adx_min = float(getattr(settings, "MARKET_REGIME_ADX_MIN", 0))
    regime_gate_skips = 0

    # -- Daily trade throttle state --
    daily_trade_counts: dict[str, int] = {}  # date_str → count
    daily_throttle_skips = 0

    # -- HMM regime: recompute every N bars (default ~6h) --
    hmm_enabled = bool(getattr(settings, "HMM_REGIME_ENABLED", False))
    hmm_regime_cache: dict[int, dict] = {}  # inst_id → regime dict
    hmm_refit_interval = 360  # bars between refits (360×1m = 6h, 72×5m = 6h)
    hmm_risk_skips = 0

    # -- GARCH blended vol: recompute every N bars --
    garch_enabled = bool(getattr(settings, "GARCH_ENABLED", False))
    garch_vol_cache: dict[int, float] = {}  # inst_id → cond_vol
    garch_refit_interval = hmm_refit_interval

    # Allocator weights (computed once, same as live)
    alloc_weights = default_weight_map()
    alloc_risk_budgets = default_risk_budget_map()

    # Module signal cache: persists between bars (mirrors live DB behaviour)
    # Key: (inst_id, module_name) → {"signal": dict, "ts": bar_ts}
    module_signal_cache: dict[tuple[int, str], dict] = {}
    # In live, modules run every ~60s and allocator reads within ALLOCATOR_WINDOW_SECONDS.
    # In backtest, bars are 5m apart, so TTL must bridge adjacent bars.
    # TTL = 2 bars allows signal from bar N to coexist with bar N+1 signals,
    # approximating live behaviour where independently-fired modules overlap
    # within the 130s allocator window.
    _bar_seconds = {"1m": 60, "3m": 180, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400}
    _ltf_bar_sec = _bar_seconds.get(ltf, 300)
    signal_ttl_seconds = _ltf_bar_sec * 2  # 2 bars (600s for 5m)

    daily_start_equity: dict[str, float] = {}  # date_str → equity at day start
    weekly_start_equity: dict[str, float] = {}  # week_str → equity at week start

    # Pre-compute positional index maps for fast slicing
    ltf_idx_map: dict[int, dict] = {}
    htf_idx_map: dict[int, dict] = {}
    for inst in instruments:
        df_ltf_full = candle_cache[inst.id].get(ltf, pd.DataFrame())
        if not df_ltf_full.empty:
            # Convert to tz-naive numpy array for searchsorted compatibility
            idx = df_ltf_full.index
            if hasattr(idx, 'tz') and idx.tz is not None:
                ts_array = idx.tz_localize(None).values
            else:
                ts_array = idx.values
            ltf_idx_map[inst.id] = {"ts": ts_array, "df": df_ltf_full}
        df_htf_full = candle_cache[inst.id].get(htf, pd.DataFrame())
        if not df_htf_full.empty:
            idx = df_htf_full.index
            if hasattr(idx, 'tz') and idx.tz is not None:
                ts_array = idx.tz_localize(None).values
            else:
                ts_array = idx.values
            htf_idx_map[inst.id] = {"ts": ts_array, "df": df_htf_full}

    # 1h index map for trend/meanrev/carry modules (mirrors live module_engine using 1h)
    htf_1h_idx_map: dict[int, dict] = {}
    for inst in instruments:
        df_1h = candle_cache[inst.id].get("1h", pd.DataFrame())
        if not df_1h.empty:
            idx = df_1h.index
            if hasattr(idx, 'tz') and idx.tz is not None:
                ts_array = idx.tz_localize(None).values
            else:
                ts_array = idx.values
            htf_1h_idx_map[inst.id] = {"ts": ts_array, "df": df_1h}

    total_bars = len(timeline)
    log_every = max(1, total_bars // 20)  # progress log ~20 times

    for bar_idx, bar_ts in enumerate(timeline):
        if bar_idx % log_every == 0:
            logger.info("  Progress: %d/%d bars (%.0f%%), equity=$%.2f, trades=%d",
                        bar_idx, total_bars, bar_idx / total_bars * 100, equity, len(trades))

        # Convert bar_ts to tz-naive numpy datetime64 for searchsorted
        bar_ts_naive = np.datetime64(bar_ts.replace(tzinfo=None) if hasattr(bar_ts, 'replace') else bar_ts)

        # Drawdown tracking
        day_key = bar_ts.strftime("%Y-%m-%d") if hasattr(bar_ts, "strftime") else str(bar_ts)[:10]
        if day_key not in daily_start_equity:
            daily_start_equity[day_key] = equity

        iso_cal = bar_ts.isocalendar() if hasattr(bar_ts, "isocalendar") else (0, 0, 0)
        week_key = f"{iso_cal[0]}-W{iso_cal[1]:02d}"
        if week_key not in weekly_start_equity:
            weekly_start_equity[week_key] = equity

        # Daily DD check
        day_start = daily_start_equity[day_key]
        daily_dd = (equity - day_start) / day_start if day_start > 0 else 0
        dd_blocked = daily_dd <= -settings.DAILY_DD_LIMIT

        # Weekly DD check
        week_start = weekly_start_equity[week_key]
        weekly_dd = (equity - week_start) / week_start if week_start > 0 else 0
        if weekly_dd <= -settings.WEEKLY_DD_LIMIT:
            dd_blocked = True

        # -- Periodic HMM regime refit (every ~6h worth of bars) --
        if hmm_enabled and bar_idx % hmm_refit_interval == 0:
            for _ri in instruments:
                htf_1h_ri = htf_1h_idx_map.get(_ri.id)
                if htf_1h_ri is None:
                    continue
                ts_1h_r = htf_1h_ri["ts"]
                h1_pos_r = np.searchsorted(ts_1h_r, bar_ts_naive, side="right")
                start_1h_r = max(0, h1_pos_r - 500)
                df_1h_r = htf_1h_ri["df"].iloc[start_1h_r:h1_pos_r]
                if len(df_1h_r) >= 100:
                    try:
                        from signals.regime import predict_regime_from_df
                        regime = predict_regime_from_df(df_1h_r)
                        if regime:
                            hmm_regime_cache[_ri.id] = regime
                    except Exception:
                        pass

        # -- Periodic GARCH refit (every ~6h worth of bars) --
        if garch_enabled and bar_idx % garch_refit_interval == 0:
            for _gi in instruments:
                htf_1h_gi = htf_1h_idx_map.get(_gi.id)
                if htf_1h_gi is None:
                    continue
                ts_1h_gv = htf_1h_gi["ts"]
                h1_pos_gv = np.searchsorted(ts_1h_gv, bar_ts_naive, side="right")
                start_1h_gv = max(0, h1_pos_gv - 500)
                df_1h_gv = htf_1h_gi["df"].iloc[start_1h_gv:h1_pos_gv]
                if len(df_1h_gv) >= 60:
                    try:
                        from signals.garch import forecast_vol_from_df
                        gvol = forecast_vol_from_df(df_1h_gv)
                        if gvol is not None:
                            garch_vol_cache[_gi.id] = gvol
                    except Exception:
                        pass

        for inst in instruments:
            ltf_info = ltf_idx_map.get(inst.id)
            if ltf_info is None:
                continue

            df_ltf_full = ltf_info["df"]
            ts_ltf = ltf_info["ts"]

            # Fast positional lookup
            pos_idx = np.searchsorted(ts_ltf, bar_ts_naive, side="right")
            if pos_idx == 0 or ts_ltf[pos_idx - 1] != bar_ts_naive:
                continue  # bar not found for this instrument

            bar = df_ltf_full.iloc[pos_idx - 1]
            close_price = float(bar["close"])

            # -- 1. Check existing position (cheap — always runs) --
            has_position = inst.id in positions
            position_just_closed = False

            if has_position:
                pos = positions[inst.id]
                exit_results = pos.check_exit(bar, bar_ts, trailing_stop)
                if exit_results:
                    trades.extend(exit_results)
                    equity += sum(r.pnl_abs for r in exit_results)
                    if pos.qty <= 0:
                        del positions[inst.id]
                        has_position = False
                        position_just_closed = True
                    if verbose:
                        for result in exit_results:
                            tag = "PARTIAL" if result.reason == "partial_close" else "CLOSE"
                            logger.info(
                                "  %s %s %s reason=%s pnl=%.4f equity=%.2f",
                                tag, inst.symbol, result.side, result.reason,
                                result.pnl_abs, equity,
                            )
                    continue  # exit or partial — move to next instrument

            # -- 2. Signal detection --
            # Skip if in cooldown (only when no position — signal flip ignores cooldown)
            if not has_position and not position_just_closed:
                cooldown_until = signal_cooldown.get(inst.id)
                if cooldown_until and bar_ts < cooldown_until:
                    continue

            # Skip if drawdown-blocked
            if dd_blocked:
                continue

            # -- Regime ADX gate (per-instrument, mirrors live) --
            if regime_adx_min > 0 and not has_position:
                htf_1h_info_gate = htf_1h_idx_map.get(inst.id)
                if htf_1h_info_gate is not None:
                    ts_1h_g = htf_1h_info_gate["ts"]
                    h1_pos_g = np.searchsorted(ts_1h_g, bar_ts_naive, side="right")
                    start_1h_g = max(0, h1_pos_g - 100)
                    df_1h_gate = htf_1h_info_gate["df"].iloc[start_1h_g:h1_pos_g]
                    if len(df_1h_gate) >= 30:
                        _inst_adx_1h = compute_adx(df_1h_gate, period=14)
                        if _inst_adx_1h is not None and _inst_adx_1h < regime_adx_min:
                            regime_gate_skips += 1
                            continue

            # -- Daily trade throttle (per-instrument ADX adaptive) --
            if not has_position:
                _day_k = bar_ts.strftime("%Y-%m-%d") if hasattr(bar_ts, "strftime") else str(bar_ts)[:10]
                _today_count = daily_trade_counts.get(_day_k, 0)
                # Compute 1h ADX for this instrument's throttle tier
                _throttle_adx = None
                htf_1h_info_thr = htf_1h_idx_map.get(inst.id)
                if htf_1h_info_thr is not None:
                    ts_1h_t = htf_1h_info_thr["ts"]
                    h1_pos_t = np.searchsorted(ts_1h_t, bar_ts_naive, side="right")
                    start_1h_t = max(0, h1_pos_t - 100)
                    df_1h_thr = htf_1h_info_thr["df"].iloc[start_1h_t:h1_pos_t]
                    if len(df_1h_thr) >= 30:
                        _throttle_adx = compute_adx(df_1h_thr, period=14)
                _daily_limit = _max_daily_trades_for_adx(_throttle_adx)
                if _today_count >= _daily_limit:
                    daily_throttle_skips += 1
                    continue

            session = get_current_session(bar_ts.hour)
            if session_policy_enabled and session_dead_zone_block and is_dead_session(session):
                session_skips_dead += 1
                continue
            session_min_score = (
                get_session_score_min(session, session_score_overrides)
                if session_policy_enabled
                else None
            )

            # Slice candles ONLY when we actually need signal detection
            start_ltf = max(0, pos_idx - lookback_bars)
            df_ltf_slice = df_ltf_full.iloc[start_ltf:pos_idx]

            htf_info = htf_idx_map.get(inst.id)
            if htf_info is not None:
                ts_htf = htf_info["ts"]
                htf_pos = np.searchsorted(ts_htf, bar_ts_naive, side="right")
                start_htf = max(0, htf_pos - lookback_bars)
                df_htf_slice = htf_info["df"].iloc[start_htf:htf_pos]
            else:
                df_htf_slice = pd.DataFrame()

            # 1h HTF slice for trend/meanrev/carry (mirrors live module_engine)
            htf_1h_info = htf_1h_idx_map.get(inst.id)
            if htf_1h_info is not None:
                ts_1h = htf_1h_info["ts"]
                h1_pos = np.searchsorted(ts_1h, bar_ts_naive, side="right")
                start_1h = max(0, h1_pos - lookback_bars)
                df_htf_1h_slice = htf_1h_info["df"].iloc[start_1h:h1_pos]
            else:
                df_htf_1h_slice = df_htf_slice  # fallback to primary HTF

            # Funding rates up to now
            funding_df = funding_cache.get(inst.id, pd.DataFrame())
            if not funding_df.empty:
                funding_slice = funding_df.loc[:bar_ts]
                funding_rates = funding_slice["rate"].tolist()
            else:
                funding_rates = []

            if len(df_ltf_slice) < 30 or (len(df_htf_slice) < 30 and len(df_htf_1h_slice) < 30):
                continue

            # ── Run all 4 modules (mirrors live: each fires independently) ──
            session_str = session if session_policy_enabled else get_current_session(bar_ts.hour)

            # 1) Trend module (uses 1h HTF like live)
            try:
                tr = trend_detect(df_ltf_slice, df_htf_1h_slice, funding_rates, session_str)
                if tr and tr.get("direction") in {"long", "short"}:
                    module_signal_cache[(inst.id, "trend")] = {"signal": {"module": "trend", **tr}, "ts": bar_ts}
            except Exception:
                pass

            # 2) Mean-reversion module (uses 1h HTF like live)
            try:
                mr = meanrev_detect(df_ltf_slice, df_htf_1h_slice, funding_rates, session_str)
                if mr and mr.get("direction") in {"long", "short"}:
                    module_signal_cache[(inst.id, "meanrev")] = {"signal": {"module": "meanrev", **mr}, "ts": bar_ts}
            except Exception:
                pass

            # 3) Carry module (uses 1h HTF like live)
            try:
                ca = carry_detect(df_ltf_slice, df_htf_1h_slice, funding_rates, session_str)
                if ca and ca.get("direction") in {"long", "short"}:
                    module_signal_cache[(inst.id, "carry")] = {"signal": {"module": "carry", **ca}, "ts": bar_ts}
            except Exception:
                pass

            # 4) SMC module (existing _detect_signal, uses 4h HTF)
            smc_ok, smc_dir, smc_explain, smc_score = _detect_signal(
                df_ltf_slice,
                df_htf_slice,
                funding_rates,
                min_score=session_min_score,
                session=session_str if session_policy_enabled else None,
            )
            if smc_ok and smc_dir in {"long", "short"}:
                module_signal_cache[(inst.id, "smc")] = {"signal": {
                    "module": "smc",
                    "direction": smc_dir,
                    "confidence": float(smc_score),
                    "raw_score": float(smc_score),
                    "smc_confluence": bool(
                        isinstance(smc_explain, dict)
                        and smc_explain.get("conditions", {}).get("liquidity_sweep")
                        and smc_explain.get("conditions", {}).get("structure_break")
                    ),
                }, "ts": bar_ts}
            # No pop-on-None: let TTL-based expiry handle stale signals.
            # In live, unfired modules simply have no new signal — old ones age
            # out naturally within the ALLOCATOR_WINDOW_SECONDS.

            # ── Collect cached signals within TTL window (mirrors live DB read) ──
            module_signals: list[dict] = []
            for mod_name in ("trend", "meanrev", "carry", "smc"):
                cached = module_signal_cache.get((inst.id, mod_name))
                if cached is None:
                    continue
                try:
                    age_sec = (bar_ts - cached["ts"]).total_seconds()
                except Exception:
                    age_sec = 9999
                if age_sec <= signal_ttl_seconds:
                    module_signals.append(cached["signal"])

            # ── Allocator: combine modules ──
            min_modules = max(1, int(getattr(settings, "ALLOCATOR_MIN_MODULES_ACTIVE", 2)))
            if len(module_signals) < min_modules:
                continue  # not enough conviction

            alloc = resolve_symbol_allocation(
                module_signals,
                threshold=float(getattr(settings, "ALLOCATOR_NET_THRESHOLD", 0.20)),
                base_risk_pct=float(settings.RISK_PER_TRADE_PCT),
                session_risk_mult=(
                    get_session_risk_mult(session, session_risk_overrides)
                    if session_policy_enabled
                    else 1.0
                ),
                weights=alloc_weights,
                risk_budgets=alloc_risk_budgets,
            )
            direction = alloc["direction"]
            score = float(alloc["confidence"])

            if direction not in {"long", "short"}:
                continue

            if not is_direction_allowed(direction, inst.symbol):
                direction_skips += 1
                if verbose:
                    logger.info(
                        "  Direction policy blocked %s %s (mode=%s)",
                        inst.symbol,
                        direction,
                        get_direction_mode(inst.symbol),
                    )
                continue

            # -- 2b. Signal-flip exit: close open position if direction opposes --
            new_side = "buy" if direction == "long" else "sell"
            if has_position:
                pos = positions[inst.id]
                # Check SIGNAL_FLIP_MIN_AGE: skip flip if position too young
                flip_min_age = float(getattr(settings, "SIGNAL_FLIP_MIN_AGE_MINUTES", 0) or 0)
                flip_age_enabled = bool(getattr(settings, "SIGNAL_FLIP_MIN_AGE_ENABLED", False))
                if flip_age_enabled and flip_min_age > 0:
                    try:
                        age_min = (bar_ts - pos.entry_ts).total_seconds() / 60.0
                    except Exception:
                        age_min = 999
                    if age_min < flip_min_age:
                        continue  # position too young to flip

                if pos.side != new_side:
                    # Direction flipped — close at bar close
                    result = pos.force_close(close_price, bar_ts, reason="signal_flip")
                    trades.append(result)
                    equity += result.pnl_abs
                    del positions[inst.id]
                    has_position = False
                    if verbose:
                        logger.info(
                            "  FLIP %s %s→%s pnl=%.4f equity=%.2f",
                            inst.symbol, pos.side, new_side, result.pnl_abs, equity,
                        )
                    # Fall through to open new position in flipped direction
                else:
                    # Same direction — keep position, no new entry
                    continue

            # -- 3. Open new position --
            # ATR-based sizing (with GARCH blend if enabled)
            atr = _atr_pct_df(df_ltf_slice)
            side = new_side

            # GARCH blended vol: replaces pure ATR for TP/SL computation
            vol_for_sizing = atr
            if garch_enabled and inst.id in garch_vol_cache and atr is not None and atr > 0:
                garch_w = float(getattr(settings, "GARCH_BLEND_WEIGHT", 0.6))
                garch_w = max(0.0, min(1.0, garch_w))
                vol_for_sizing = garch_w * garch_vol_cache[inst.id] + (1.0 - garch_w) * atr

            tp_pct = settings.TAKE_PROFIT_PCT
            sl_pct = settings.STOP_LOSS_PCT
            if vol_for_sizing and vol_for_sizing > 0:
                tp_pct = max(tp_pct, vol_for_sizing * settings.ATR_MULT_TP)
                sl_pct = max(sl_pct, vol_for_sizing * settings.ATR_MULT_SL)
            # Match runtime: enforce absolute minimum SL to avoid noise-driven stop-outs.
            min_sl = float(getattr(settings, "MIN_SL_PCT", 0.0) or 0.0)
            if min_sl > 0:
                sl_pct = max(sl_pct, min_sl)

            # Risk-based qty (vol-adjusted risk + HMM regime mult)
            session_risk_mult = (
                get_session_risk_mult(session, session_risk_overrides)
                if session_policy_enabled
                else 1.0
            )
            if session_risk_mult <= 0:
                session_skips_dead += 1
                continue

            # HMM regime risk multiplier (mirrors live)
            _hmm_risk_mult = 1.0
            if hmm_enabled and inst.id in hmm_regime_cache:
                _hmm_risk_mult = float(hmm_regime_cache[inst.id].get("risk_mult", 1.0))
                if _hmm_risk_mult < 1.0:
                    hmm_risk_skips += 1  # count risk reductions

            # Volatility-adjusted risk (mirrors execution/tasks.py)
            inst_risk = _volatility_adjusted_risk(
                inst.symbol, vol_for_sizing, settings.RISK_PER_TRADE_PCT,
            )
            stop_dist = sl_pct
            if stop_dist <= 0:
                continue
            risk_amount = inst_risk * equity
            qty = (risk_amount / (stop_dist * close_price)) * session_risk_mult * _hmm_risk_mult
            # Round to reasonable precision (3 decimals for BTC, no floor to 1)
            qty = round(qty, 3)
            if qty <= 0:
                continue

            # Margin check
            notional = close_price * qty
            leverage = settings.MAX_EFF_LEVERAGE
            required_margin = notional / leverage if leverage > 0 else notional
            if required_margin > equity * 0.95:  # leave 5% buffer
                # Try reducing qty to fit margin
                max_notional = equity * 0.95 * leverage
                qty = round(max_notional / close_price, 3)
                notional = close_price * qty
                if qty <= 0:
                    continue

            # Per-instrument exposure cap
            max_exposure = equity * settings.MAX_EXPOSURE_PER_INSTRUMENT_PCT * leverage
            if notional > max_exposure:
                qty = round(max_exposure / close_price, 3)
                notional = close_price * qty
                if qty <= 0:
                    continue

            # TP / SL prices
            if side == "buy":
                tp_price = close_price * (1 + tp_pct)
                sl_price = close_price * (1 - sl_pct)
            else:
                tp_price = close_price * (1 - tp_pct)
                sl_price = close_price * (1 + sl_pct)

            pos = SimPosition(
                instrument_id=inst.id,
                side=side,
                qty=qty,
                entry_price=close_price,
                entry_ts=bar_ts,
                sl_price=sl_price,
                tp_price=tp_price,
                sl_pct=sl_pct,
                tp_pct=tp_pct,
                score=score,
            )
            positions[inst.id] = pos
            session_open_counts[session] = session_open_counts.get(session, 0) + 1

            # Increment daily trade count (for throttle)
            _open_day = bar_ts.strftime("%Y-%m-%d") if hasattr(bar_ts, "strftime") else str(bar_ts)[:10]
            daily_trade_counts[_open_day] = daily_trade_counts.get(_open_day, 0) + 1

            # Cooldown between trades per instrument (asymmetric per symbol)
            inst_cooldown = settings.PER_INSTRUMENT_COOLDOWN.get(inst.symbol, settings.SIGNAL_COOLDOWN_MINUTES)
            signal_cooldown[inst.id] = bar_ts + timedelta(minutes=inst_cooldown)

            if verbose:
                logger.info(
                    "  OPEN %s %s qty=%d price=%.4f sl=%.4f tp=%.4f score=%.3f session=%s risk_mult=%.2f",
                    inst.symbol, side, qty, close_price, sl_price, tp_price, score, session, session_risk_mult,
                )

    # Close any remaining open positions at last known price
    for inst_id, pos in list(positions.items()):
        inst = next((i for i in instruments if i.id == inst_id), None)
        if inst:
            df_ltf_full = candle_cache[inst.id].get(ltf, pd.DataFrame())
            if not df_ltf_full.empty:
                last_close = float(df_ltf_full["close"].iloc[-1])
                last_ts = df_ltf_full.index[-1]
                trade = pos.force_close(last_close, last_ts, reason="end_of_data")
                trades.append(trade)
                equity += trade.pnl_abs

    elapsed = time.time() - t0
    _bpd = {"1m": 1440, "3m": 480, "5m": 288, "15m": 96, "1h": 24, "4h": 6}
    metrics = compute_metrics(trades, initial_equity, bars_per_day=_bpd.get(ltf, 288))
    metrics["elapsed_seconds"] = round(elapsed, 2)
    metrics["bars_processed"] = len(timeline)
    metrics["instruments_tested"] = len(instruments)
    metrics["session_skips_dead"] = int(session_skips_dead)
    metrics["session_skips_score"] = int(session_skips_score)
    metrics["direction_skips"] = int(direction_skips)
    metrics["session_open_counts"] = dict(session_open_counts)
    # Signal flip stats
    signal_flip_trades = [t for t in trades if t.reason == "signal_flip"]
    metrics["signal_flip_count"] = len(signal_flip_trades)
    metrics["signal_flip_pnl"] = round(sum(t.pnl_abs for t in signal_flip_trades), 4)
    # Breakdown by close reason
    reason_counts = {}
    for t in trades:
        reason_counts[t.reason] = reason_counts.get(t.reason, 0) + 1
    metrics["close_reason_counts"] = reason_counts
    # New filter counters (parity with live)
    metrics["regime_gate_skips"] = int(regime_gate_skips)
    metrics["daily_throttle_skips"] = int(daily_throttle_skips)
    metrics["hmm_risk_reductions"] = int(hmm_risk_skips)
    metrics["garch_enabled"] = garch_enabled
    metrics["hmm_enabled"] = hmm_enabled
    metrics["daily_trade_counts"] = dict(daily_trade_counts)

    return trades, metrics
