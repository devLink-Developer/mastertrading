"""Quick BTC-only backtest to diagnose why it doesn't trade."""
import os, sys, django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
django.setup()

from datetime import datetime
from django.utils import timezone as dj_tz
from core.models import Instrument
from backtest.engine import run_backtest

# Test BTC only, short period
inst = [Instrument.objects.get(symbol="BTCUSDT")]
start = dj_tz.make_aware(datetime(2025, 7, 1))
end = dj_tz.make_aware(datetime(2025, 8, 1))

trades, metrics = run_backtest(inst, start, end, initial_equity=1000, ltf="5m", htf="4h", trailing_stop=True, verbose=True)
print(f"\nTrades: {metrics['total_trades']}, PnL: {metrics['total_pnl_pct']}%")
for t in trades:
    print(f"  {t.side} entry={t.entry_price:.2f} exit={t.exit_price:.2f} pnl={t.pnl_abs:.2f} reason={t.reason} score={t.score}")
