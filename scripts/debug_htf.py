import os, sys, django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
django.setup()

import pandas as pd, numpy as np
from backtest.engine import load_candles
from datetime import datetime, timedelta
from django.utils import timezone as dj_tz
from core.models import Instrument

inst = Instrument.objects.get(symbol="BTCUSDT")
start = dj_tz.make_aware(datetime(2025, 7, 1))
lookback_start = start - timedelta(days=30)
end = dj_tz.make_aware(datetime(2025, 8, 1))

df5 = load_candles(inst, "5m", lookback_start, end)
df4 = load_candles(inst, "4h", lookback_start, end)
print(f"LTF: {len(df5)}, HTF: {len(df4)}")
print(f"HTF range: {df4.index[0]} to {df4.index[-1]}")

idx4 = df4.index
ts4 = idx4.tz_localize(None).values if hasattr(idx4, "tz") and idx4.tz else idx4.values

idx5 = df5.index
ts5 = idx5.tz_localize(None).values if hasattr(idx5, "tz") and idx5.tz else idx5.values

mask = df5.index >= start
timeline = df5.index[mask].tolist()

for label, bar_ts in [("Jul 1", timeline[0]), ("Jul 5", timeline[1152]), ("Jul 15", timeline[4000])]:
    bar_naive = np.datetime64(bar_ts.replace(tzinfo=None))
    
    pos5 = np.searchsorted(ts5, bar_naive, side="right")
    start5 = max(0, pos5 - 200)
    ltf_slice = df5.iloc[start5:pos5]
    
    htf_pos = np.searchsorted(ts4, bar_naive, side="right")
    start4 = max(0, htf_pos - 200)
    htf_slice = df4.iloc[start4:htf_pos]
    
    print(f"\n{label} ({bar_ts}):")
    print(f"  LTF: pos={pos5}, slice_len={len(ltf_slice)}")
    print(f"  HTF: pos={htf_pos}, slice_len={len(htf_slice)}")
    if len(htf_slice) > 0:
        print(f"  HTF first: {htf_slice.index[0]}, last: {htf_slice.index[-1]}")
