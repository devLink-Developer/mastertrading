"""Quick check of recent orders and signals."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
import django
django.setup()

from execution.models import Order, OperationReport
from signals.models import Signal
from django.utils import timezone as tz
from datetime import timedelta

now = tz.now()
cutoff = now - timedelta(hours=3)

print("=== ULTIMAS ORDENES (3h) ===")
for o in Order.objects.filter(opened_at__gte=cutoff).order_by('-opened_at')[:10]:
    print(f"  {o.opened_at}  {o.instrument.symbol:12s}  side={o.side:4s}  qty={o.quantity}  status={o.status}")

print()
print("=== ULTIMAS SENALES EMITIDAS (3h) ===")
for s in Signal.objects.filter(created_at__gte=cutoff).order_by('-created_at')[:10]:
    p = s.payload or {}
    conds = p.get("conditions", {})
    print(f"  {s.created_at}  {s.instrument.symbol:12s}  dir={s.direction:5s}  score={s.score:.3f}")
    print(f"    conditions: {conds}")
    if p.get("short_score_penalty"):
        print(f"    SHORT PENALTY applied: -{p['short_score_penalty']}")
    if p.get("ema_score_adjustment"):
        print(f"    ema_adj: {p['ema_score_adjustment']}")
    if p.get("htf_adx"):
        print(f"    htf_adx: {p['htf_adx']}")
    print()

print("=== CONFIG ACTUAL ===")
from django.conf import settings
print(f"  SIGNAL_DIRECTION_MODE: {settings.SIGNAL_DIRECTION_MODE}")
print(f"  SHORT_SCORE_PENALTY: {getattr(settings, 'SHORT_SCORE_PENALTY', 'NOT SET')}")
print(f"  MIN_SIGNAL_SCORE: {settings.MIN_SIGNAL_SCORE}")
print(f"  MAX_DAILY_TRADES: {getattr(settings, 'MAX_DAILY_TRADES', 'NOT SET')}")
print(f"  STALE_POSITION_ENABLED: {getattr(settings, 'STALE_POSITION_ENABLED', 'NOT SET')}")
print(f"  UPTREND_SHORT_KILLER_ENABLED: {getattr(settings, 'UPTREND_SHORT_KILLER_ENABLED', 'NOT SET')}")
print(f"  PYRAMIDING_ENABLED: {settings.PYRAMIDING_ENABLED}")
