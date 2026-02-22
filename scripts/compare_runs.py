"""Compare all backtest runs side by side."""
import os, django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

from backtest.models import BacktestRun, BacktestTrade
from django.db.models import Count, Avg, Sum, Q

print("=" * 80)
print("  BACKTEST RUNS COMPARISON")
print("=" * 80)

for r in BacktestRun.objects.all().order_by("id"):
    m = r.metrics_json
    print(f"\n--- Run #{r.id}: {r.name} ({r.start_date:%Y-%m-%d} to {r.end_date:%Y-%m-%d}) ---")
    print(f"  Trades       : {m.get('total_trades', 0)}")
    print(f"  Win Rate     : {m.get('win_rate', 0)}%")
    print(f"  Total PnL    : ${m.get('total_pnl', 0):+,.4f} ({m.get('total_pnl_pct', 0):+.2f}%)")
    print(f"  Final Equity : ${m.get('final_equity', 0):,.2f}")
    print(f"  Profit Factor: {m.get('profit_factor', 0)}")
    print(f"  Sharpe       : {m.get('sharpe_ratio', 0)}")
    print(f"  Max Drawdown : {m.get('max_drawdown_pct', 0):.2f}%")
    print(f"  Avg Win      : ${m.get('avg_win', 0):+,.4f}")
    print(f"  Avg Loss     : ${m.get('avg_loss', 0):+,.4f}")
    print(f"  Expectancy   : ${m.get('expectancy', 0):+,.4f}")
    print(f"  Total Fees   : ${m.get('total_fees', 0):,.4f}")
    print(f"  Avg Duration : {m.get('avg_trade_duration_min', 0):.1f} min")
    
    # Compute R:R
    avg_w = m.get('avg_win', 0)
    avg_l = m.get('avg_loss', 0)
    rr = abs(avg_w / avg_l) if avg_l and avg_l != 0 else 0
    print(f"  R:R Ratio    : {rr:.2f}")

    # Per-reason breakdown
    trades = BacktestTrade.objects.filter(run=r)
    if trades.exists():
        print(f"\n  Exit Reasons:")
        for row in trades.values("reason").annotate(
            c=Count("id"), avg_pnl=Avg("pnl_abs"), total_pnl=Sum("pnl_abs")
        ).order_by("-c"):
            wins = trades.filter(reason=row["reason"], outcome="win").count()
            wr = wins / row["c"] * 100 if row["c"] else 0
            print(f"    {row['reason']:<16} {row['c']:>4} trades  WR={wr:5.1f}%  avg=${row['avg_pnl']:+8.4f}  total=${row['total_pnl']:+10.4f}")

        # Score breakdown
        print(f"\n  Score Bands:")
        for label, filt in [("<0.55", Q(score__lt=0.55)), ("0.55-0.7", Q(score__gte=0.55, score__lt=0.7)), (">=0.7", Q(score__gte=0.7))]:
            qs = trades.filter(filt)
            if qs.exists():
                agg = qs.aggregate(cnt=Count("id"), avg_pnl=Avg("pnl_abs"), total_pnl=Sum("pnl_abs"))
                wins = qs.filter(outcome="win").count()
                wr = wins / agg["cnt"] * 100
                print(f"    {label:<10} {agg['cnt']:>4} trades  WR={wr:5.1f}%  avg=${agg['avg_pnl']:+8.4f}  total=${agg['total_pnl']:+10.4f}")

        # Per instrument
        print(f"\n  Per Instrument:")
        for row in trades.values("instrument__symbol").annotate(
            c=Count("id"), total_pnl=Sum("pnl_abs"), avg_pnl=Avg("pnl_abs")
        ).order_by("instrument__symbol"):
            wins = trades.filter(instrument__symbol=row["instrument__symbol"], outcome="win").count()
            wr = wins / row["c"] * 100 if row["c"] else 0
            print(f"    {row['instrument__symbol']:<12} {row['c']:>4} trades  WR={wr:5.1f}%  total=${row['total_pnl']:+10.4f}")

print("\n" + "=" * 80)
