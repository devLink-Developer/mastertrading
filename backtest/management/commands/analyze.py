"""Analyze a backtest run. Usage: python manage.py analyze --run 2"""
from django.core.management.base import BaseCommand
from backtest.models import BacktestTrade
from django.db.models import Count, Avg, Sum, Q


class Command(BaseCommand):
    help = "Analyze a backtest run"

    def add_arguments(self, parser):
        parser.add_argument("--run", type=int, default=2)

    def handle(self, *args, **options):
        run_id = options["run"]
        trades = BacktestTrade.objects.filter(run_id=run_id)
        total = trades.count()
        self.stdout.write(f"=== BACKTEST #{run_id} ANALYSIS ({total} trades) ===\n")

        buys = trades.filter(side="buy").count()
        sells = trades.filter(side="sell").count()
        self.stdout.write(f"Direction: {buys} buys, {sells} sells")

        for side in ("buy", "sell"):
            qs = trades.filter(side=side)
            if qs.exists():
                agg = qs.aggregate(avg=Avg("pnl_abs"), total=Sum("pnl_abs"), cnt=Count("id"))
                wins = qs.filter(outcome="win").count()
                wr = wins / agg["cnt"] * 100 if agg["cnt"] else 0
                self.stdout.write(f"  {side}: {agg['cnt']} trades, WR={wr:.1f}%, avg_pnl={agg['avg']:.4f}, total_pnl={agg['total']:.4f}")

        self.stdout.write("\nExit reasons:")
        for row in trades.values("reason").annotate(c=Count("id"), avg=Avg("pnl_abs"), total=Sum("pnl_abs")).order_by("-c"):
            self.stdout.write(f"  {row['reason']}: {row['c']} trades, avg={row['avg']:.4f}, total={row['total']:.4f}")

        self.stdout.write("\nScore distribution:")
        for label, filt in [("<0.5", Q(score__lt=0.5)), ("0.5-0.7", Q(score__gte=0.5, score__lt=0.7)), (">=0.7", Q(score__gte=0.7))]:
            qs = trades.filter(filt)
            if qs.exists():
                agg = qs.aggregate(avg=Avg("pnl_abs"), total=Sum("pnl_abs"), cnt=Count("id"))
                wins = qs.filter(outcome="win").count()
                wr = wins / agg["cnt"] * 100
                self.stdout.write(f"  Score {label}: {agg['cnt']} trades, WR={wr:.1f}%, avg={agg['avg']:.4f}, total={agg['total']:.4f}")

        wins_qs = trades.filter(outcome="win")
        losses_qs = trades.filter(outcome="loss")
        if wins_qs.exists() and losses_qs.exists():
            avg_w = wins_qs.aggregate(a=Avg("pnl_abs"))["a"]
            avg_l = losses_qs.aggregate(a=Avg("pnl_abs"))["a"]
            rr = abs(avg_w / avg_l) if avg_l else 0
            self.stdout.write(f"\nRisk/Reward: avg_win={avg_w:.4f}, avg_loss={avg_l:.4f}, R:R={rr:.2f}")

        total_fees = trades.aggregate(f=Sum("fee_paid"))["f"] or 0
        total_pnl = trades.aggregate(p=Sum("pnl_abs"))["p"] or 0
        gross_pnl = total_pnl + total_fees
        self.stdout.write(f"\nFees: gross_pnl={gross_pnl:.2f}, fees={total_fees:.2f}, net={total_pnl:.2f}")
