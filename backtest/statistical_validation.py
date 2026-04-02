"""
Purged K-Fold Cross-Validation and White's Reality Check (SPA test).

Purged CV prevents data leakage in time-series by:
  - Splitting chronologically (no random shuffle)
  - Purging observations near the train/test boundary within an embargo window
  - Ensuring the model never trains on data adjacent to the test fold

White's Reality Check / Hansen's SPA test:
  - Tests H0: the best strategy is no better than a benchmark (zero return)
  - Uses stationary bootstrap to generate p-values
  - Guards against data-snooping bias when comparing multiple strategies/configs
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Purged K-Fold CV
# ---------------------------------------------------------------------------

@dataclass
class FoldResult:
    """Metrics for a single fold of cross-validation."""
    fold: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    n_train_trades: int
    n_test_trades: int
    train_pnl_pct: float
    test_pnl_pct: float
    train_win_rate: float
    test_win_rate: float
    train_sharpe: float
    test_sharpe: float
    train_max_dd: float
    test_max_dd: float
    test_expectancy: float


@dataclass
class PurgedCVResult:
    """Aggregated results across all folds."""
    n_folds: int
    embargo_days: int
    folds: list[FoldResult] = field(default_factory=list)
    mean_test_pnl: float = 0.0
    std_test_pnl: float = 0.0
    mean_test_sharpe: float = 0.0
    mean_test_win_rate: float = 0.0
    mean_test_expectancy: float = 0.0
    mean_train_test_gap: float = 0.0  # overfitting indicator
    deflated_sharpe_pvalue: float = 1.0

    def to_dict(self) -> dict:
        return {
            "n_folds": self.n_folds,
            "embargo_days": self.embargo_days,
            "mean_test_pnl_pct": round(self.mean_test_pnl, 4),
            "std_test_pnl_pct": round(self.std_test_pnl, 4),
            "mean_test_sharpe": round(self.mean_test_sharpe, 4),
            "mean_test_win_rate": round(self.mean_test_win_rate, 4),
            "mean_test_expectancy": round(self.mean_test_expectancy, 6),
            "mean_train_test_gap_pnl": round(self.mean_train_test_gap, 4),
            "deflated_sharpe_pvalue": round(self.deflated_sharpe_pvalue, 6),
            "folds": [
                {
                    "fold": f.fold,
                    "train_period": f"{f.train_start:%Y-%m-%d} → {f.train_end:%Y-%m-%d}",
                    "test_period": f"{f.test_start:%Y-%m-%d} → {f.test_end:%Y-%m-%d}",
                    "n_train": f.n_train_trades,
                    "n_test": f.n_test_trades,
                    "train_pnl_pct": round(f.train_pnl_pct, 4),
                    "test_pnl_pct": round(f.test_pnl_pct, 4),
                    "train_wr": round(f.train_win_rate, 4),
                    "test_wr": round(f.test_win_rate, 4),
                    "train_sharpe": round(f.train_sharpe, 4),
                    "test_sharpe": round(f.test_sharpe, 4),
                    "test_max_dd": round(f.test_max_dd, 4),
                    "test_expectancy": round(f.test_expectancy, 6),
                }
                for f in self.folds
            ],
        }


def _compute_fold_metrics(pnl_pcts: list[float]) -> dict:
    """Compute metrics for a list of trade PnL percentages."""
    if not pnl_pcts:
        return {"pnl": 0.0, "wr": 0.0, "sharpe": 0.0, "max_dd": 0.0, "expectancy": 0.0}
    arr = np.array(pnl_pcts, dtype=np.float64)
    wins = arr[arr > 0]
    losses = arr[arr <= 0]
    wr = len(wins) / len(arr) if len(arr) > 0 else 0.0
    mean_r = float(arr.mean())
    std_r = float(arr.std(ddof=1)) if len(arr) > 1 else 1e-9
    sharpe = (mean_r / std_r) * np.sqrt(252) if std_r > 1e-12 else 0.0

    # Max drawdown from cumulative PnL
    cum = np.cumsum(arr)
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    max_dd = float(dd.min()) if len(dd) > 0 else 0.0

    avg_win = float(wins.mean()) if len(wins) > 0 else 0.0
    avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0
    expectancy = wr * avg_win + (1 - wr) * avg_loss

    return {
        "pnl": float(arr.sum()),
        "wr": wr,
        "sharpe": float(sharpe),
        "max_dd": max_dd,
        "expectancy": expectancy,
    }


def purged_kfold_cv(
    trades: list[dict],
    n_folds: int = 5,
    embargo_pct: float = 0.05,
    embargo_days: int = 0,
) -> PurgedCVResult:
    """Run Purged K-Fold Cross-Validation on a chronological trade list.

    Parameters
    ----------
    trades : list of dict
        Each dict must have 'entry_ts' (datetime), 'exit_ts' (datetime),
        'pnl_pct' (float). Can come from SimTrade or OperationReport.
    n_folds : int
        Number of CV folds (default 5).
    embargo_pct : float
        Fraction of the training set duration to purge at the boundary (default 5%).
    embargo_days : int
        Explicit embargo in days (overrides embargo_pct if > 0).

    Returns
    -------
    PurgedCVResult with per-fold and aggregated metrics.
    """
    if len(trades) < n_folds * 2:
        return PurgedCVResult(n_folds=n_folds, embargo_days=embargo_days)

    # Sort chronologically by entry
    sorted_trades = sorted(trades, key=lambda t: t["entry_ts"])
    n = len(sorted_trades)
    fold_size = n // n_folds

    folds = []
    for k in range(n_folds):
        test_start_idx = k * fold_size
        test_end_idx = (k + 1) * fold_size if k < n_folds - 1 else n

        test_trades = sorted_trades[test_start_idx:test_end_idx]
        if not test_trades:
            continue

        test_start = test_trades[0]["entry_ts"]
        test_end = test_trades[-1]["exit_ts"]

        # Compute embargo window
        if embargo_days > 0:
            emb = timedelta(days=embargo_days)
        else:
            total_duration = (sorted_trades[-1]["exit_ts"] - sorted_trades[0]["entry_ts"])
            emb_seconds = total_duration.total_seconds() * embargo_pct / n_folds
            emb = timedelta(seconds=max(emb_seconds, 3600))

        # Train = everything outside test + embargo
        train_trades = []
        for t in sorted_trades:
            t_exit = t["exit_ts"]
            t_entry = t["entry_ts"]
            # Purge: exclude trades whose exit is within embargo before test start
            # or whose entry is within embargo after test end
            if t_exit < (test_start - emb) or t_entry > (test_end + emb):
                train_trades.append(t)

        if not train_trades:
            continue

        train_pnls = [t["pnl_pct"] for t in train_trades]
        test_pnls = [t["pnl_pct"] for t in test_trades]
        train_m = _compute_fold_metrics(train_pnls)
        test_m = _compute_fold_metrics(test_pnls)

        folds.append(FoldResult(
            fold=k + 1,
            train_start=train_trades[0]["entry_ts"],
            train_end=train_trades[-1]["exit_ts"],
            test_start=test_start,
            test_end=test_end,
            n_train_trades=len(train_trades),
            n_test_trades=len(test_trades),
            train_pnl_pct=train_m["pnl"],
            test_pnl_pct=test_m["pnl"],
            train_win_rate=train_m["wr"],
            test_win_rate=test_m["wr"],
            train_sharpe=train_m["sharpe"],
            test_sharpe=test_m["sharpe"],
            train_max_dd=train_m["max_dd"],
            test_max_dd=test_m["max_dd"],
            test_expectancy=test_m["expectancy"],
        ))

    if not folds:
        return PurgedCVResult(n_folds=n_folds, embargo_days=embargo_days)

    result = PurgedCVResult(
        n_folds=n_folds,
        embargo_days=embargo_days,
        folds=folds,
        mean_test_pnl=float(np.mean([f.test_pnl_pct for f in folds])),
        std_test_pnl=float(np.std([f.test_pnl_pct for f in folds], ddof=1)) if len(folds) > 1 else 0.0,
        mean_test_sharpe=float(np.mean([f.test_sharpe for f in folds])),
        mean_test_win_rate=float(np.mean([f.test_win_rate for f in folds])),
        mean_test_expectancy=float(np.mean([f.test_expectancy for f in folds])),
        mean_train_test_gap=float(np.mean([
            f.train_pnl_pct - f.test_pnl_pct for f in folds
        ])),
    )

    # Deflated Sharpe: probability that observed Sharpe is due to multiple testing
    test_sharpes = [f.test_sharpe for f in folds]
    if len(test_sharpes) >= 2:
        result.deflated_sharpe_pvalue = _deflated_sharpe_ratio(test_sharpes)

    return result


def _deflated_sharpe_ratio(sharpes: list[float]) -> float:
    """Simplified Deflated Sharpe Ratio p-value (Bailey & López de Prado).

    Tests whether the best observed Sharpe is distinguishable from
    the expected maximum of N i.i.d. Sharpe trials under H0.
    """
    from scipy.stats import norm

    n = len(sharpes)
    if n < 2:
        return 1.0
    arr = np.array(sharpes, dtype=np.float64)
    sr_max = float(arr.max())
    sr_mean = float(arr.mean())
    sr_std = float(arr.std(ddof=1))
    if sr_std < 1e-12:
        return 1.0

    # Expected max under H0 (Euler-Mascheroni approximation)
    gamma = 0.5772156649
    e_max = sr_mean + sr_std * ((2 * np.log(n)) ** 0.5 - (np.log(np.pi) + gamma) / (2 * (2 * np.log(n)) ** 0.5))

    # PSR(SR*) = Φ((SR* - E[max SR]) / σ[SR])
    z = (sr_max - e_max) / sr_std
    return float(1.0 - norm.cdf(z))


# ---------------------------------------------------------------------------
# White's Reality Check / Hansen's SPA Test
# ---------------------------------------------------------------------------

@dataclass
class SPAResult:
    """Result of the Stepwise Superior Predictive Ability test."""
    n_strategies: int
    n_bootstrap: int
    block_size: int
    best_strategy: str
    best_mean_return: float
    spa_pvalue: float
    consistent_pvalue: float  # Hansen's consistent p-value
    significant: bool  # at 5%
    strategy_stats: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_strategies": self.n_strategies,
            "n_bootstrap": self.n_bootstrap,
            "block_size": self.block_size,
            "best_strategy": self.best_strategy,
            "best_mean_return": round(self.best_mean_return, 6),
            "spa_pvalue": round(self.spa_pvalue, 6),
            "consistent_pvalue": round(self.consistent_pvalue, 6),
            "significant_at_5pct": self.significant,
            "strategies": self.strategy_stats,
        }


def _stationary_bootstrap_indices(n: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Generate bootstrap indices using Politis & Romano's stationary bootstrap.

    Each index has probability 1/block_size of starting a new block.
    """
    prob_new_block = 1.0 / max(1, block_size)
    indices = np.empty(n, dtype=np.int64)
    indices[0] = rng.integers(0, n)
    for i in range(1, n):
        if rng.random() < prob_new_block:
            indices[i] = rng.integers(0, n)
        else:
            indices[i] = (indices[i - 1] + 1) % n
    return indices


def whites_reality_check(
    strategy_returns: dict[str, list[float]],
    n_bootstrap: int = 5000,
    block_size: int = 10,
    seed: int = 42,
) -> SPAResult:
    """White's Reality Check / Hansen's SPA test.

    Tests H0: no strategy has a positive expected return (vs. zero benchmark).

    Parameters
    ----------
    strategy_returns : dict
        {strategy_name: [trade_pnl_pct, ...]}
        All lists must have the same length (aligned by trade index or time).
    n_bootstrap : int
        Number of bootstrap replications.
    block_size : int
        Expected block length for stationary bootstrap.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    SPAResult with p-values and per-strategy diagnostics.
    """
    if not strategy_returns:
        return SPAResult(0, n_bootstrap, block_size, "", 0.0, 1.0, 1.0, False)

    names = list(strategy_returns.keys())
    # Align to common length (trim to shortest)
    min_len = min(len(v) for v in strategy_returns.values())
    if min_len < 5:
        return SPAResult(len(names), n_bootstrap, block_size, "", 0.0, 1.0, 1.0, False)

    # d_k = excess return of strategy k vs. benchmark (zero)
    d_matrix = np.array([strategy_returns[k][:min_len] for k in names], dtype=np.float64)
    # d_matrix shape: (n_strategies, n_obs)
    n_strat, n_obs = d_matrix.shape

    # Observed mean excess returns
    d_bar = d_matrix.mean(axis=1)  # (n_strat,)
    best_idx = int(np.argmax(d_bar))
    best_name = names[best_idx]
    best_mean = float(d_bar[best_idx])

    # Test statistic: max of studentized means
    d_std = d_matrix.std(axis=1, ddof=1)
    d_std[d_std < 1e-12] = 1e-12
    t_obs = float((d_bar * np.sqrt(n_obs) / d_std).max())

    # Bootstrap under H0: center the returns (remove mean)
    d_centered = d_matrix - d_bar[:, None]

    rng = np.random.default_rng(seed)
    boot_max_t = np.empty(n_bootstrap, dtype=np.float64)
    # For consistent p-value (Hansen SPA): use max(0, d_bar) centering
    d_bar_plus = np.maximum(d_bar, 0.0)
    d_centered_consistent = d_matrix - d_bar_plus[:, None]

    boot_max_t_consistent = np.empty(n_bootstrap, dtype=np.float64)

    for b in range(n_bootstrap):
        idx = _stationary_bootstrap_indices(n_obs, block_size, rng)

        # RC p-value (center at d_bar)
        boot_sample = d_centered[:, idx]
        boot_mean = boot_sample.mean(axis=1)
        boot_std = boot_sample.std(axis=1, ddof=1)
        boot_std[boot_std < 1e-12] = 1e-12
        boot_t = (boot_mean * np.sqrt(n_obs)) / boot_std
        boot_max_t[b] = float(boot_t.max())

        # Consistent p-value (center at max(0, d_bar))
        boot_sample_c = d_centered_consistent[:, idx]
        boot_mean_c = boot_sample_c.mean(axis=1)
        boot_std_c = boot_sample_c.std(axis=1, ddof=1)
        boot_std_c[boot_std_c < 1e-12] = 1e-12
        boot_t_c = (boot_mean_c * np.sqrt(n_obs)) / boot_std_c
        boot_max_t_consistent[b] = float(boot_t_c.max())

    # p-value = fraction of bootstrap max-t >= observed max-t
    spa_pvalue = float(np.mean(boot_max_t >= t_obs))
    consistent_pvalue = float(np.mean(boot_max_t_consistent >= t_obs))

    strategy_stats = []
    for i, name in enumerate(names):
        strategy_stats.append({
            "name": name,
            "mean_return": round(float(d_bar[i]), 6),
            "std_return": round(float(d_std[i]), 6),
            "t_stat": round(float(d_bar[i] * np.sqrt(n_obs) / d_std[i]), 4),
            "n_trades": min_len,
        })

    return SPAResult(
        n_strategies=n_strat,
        n_bootstrap=n_bootstrap,
        block_size=block_size,
        best_strategy=best_name,
        best_mean_return=best_mean,
        spa_pvalue=spa_pvalue,
        consistent_pvalue=consistent_pvalue,
        significant=consistent_pvalue < 0.05,
        strategy_stats=strategy_stats,
    )
