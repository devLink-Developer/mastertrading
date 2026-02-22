"""Quick analysis of backtest results."""
import os, django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

from backtest.models import BacktestTrade
from django.db.models import Count, Avg, Sum, Q

trades = BacktestTrade.objects.filter(run_id=2)
total = trades.count()
print(f"=== BACKTEST #2 ANALYSIS ({total} trades) ===\n")

# Direction
buys = trades.filter(side="buy").count()
sells = trades.filter(side="sell").count()
print(f"Direction: {buys} buys, {sells} sells")

# Buy PnL vs Sell PnL
for side in ("buy", "sell"):
    qs = trades.filter(side=side)
    if qs.exists():
        agg = qs.aggregate(avg=Avg("pnl_abs"), total=Sum("pnl_abs"), cnt=Count("id"))
        wins = qs.filter(outcome="win").count()
        wr = wins / agg["cnt"] * 100 if agg["cnt"] else 0
        print(f"  {side}: {agg['cnt']} trades, WR={wr:.1f}%, avg={agg['avg']:.4f}, total={agg['total']:.4f}")

# Instrument
print("\nInstruments:")
for row in trades.values("instrument__symbol").annotate(c=Count("id"), total=Sum("pnl_abs")):
    print(f"  {row['instrument__symbol']}: {row['c']} trades, PnL={row['total']:.4f}")

# Exit reason breakdown
print("\nExit reasons:")
for row in trades.values("reason").annotate(c=Count("id"), avg=Avg("pnl_abs"), total=Sum("pnl_abs")).order_by("-c"):
    print(f"  {row['reason']}: {row['c']} trades, avg={row['avg']:.4f}, total={row['total']:.4f}")

# Score distribution
print("\nScore distribution:")
for label, filt in [("<0.5", Q(score__lt=0.5)), ("0.5-0.7", Q(score__gte=0.5, score__lt=0.7)), (">=0.7", Q(score__gte=0.7))]:
    qs = trades.filter(filt)
    if qs.exists():
        agg = qs.aggregate(avg=Avg("pnl_abs"), total=Sum("pnl_abs"), cnt=Count("id"))
        wins = qs.filter(outcome="win").count()
        wr = wins / agg["cnt"] * 100
        print(f"  Score {label}: {agg['cnt']} trades, WR={wr:.1f}%, avg={agg['avg']:.4f}, total={agg['total']:.4f}")

# Avg win vs avg loss
wins = trades.filter(outcome="win")
losses = trades.filter(outcome="loss")
if wins.exists() and losses.exists():
    avg_w = wins.aggregate(a=Avg("pnl_abs"))["a"]
    avg_l = losses.aggregate(a=Avg("pnl_abs"))["a"]
    rr = abs(avg_w / avg_l) if avg_l else 0
    print(f"\nRisk/Reward: avg_win={avg_w:.4f}, avg_loss={avg_l:.4f}, R:R={rr:.2f}")

# Fees impact
total_fees = trades.aggregate(f=Sum("fee_paid"))["f"] or 0
total_pnl = trades.aggregate(p=Sum("pnl_abs"))["p"] or 0
gross_pnl = total_pnl + total_fees
print(f"\nFees impact: gross_pnl={gross_pnl:.4f}, fees={total_fees:.4f}, net_pnl={total_pnl:.4f}")
print(f"  Fees as % of gross loss: {total_fees/abs(total_pnl)*100:.1f}%")
