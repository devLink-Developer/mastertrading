"""Debug: inject logging into the engine loop to find why BTC signals
are detected but positions aren't opened."""
import os, sys, django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
django.setup()

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from django.conf import settings
from django.utils import timezone as dj_tz
from core.models import Instrument
from backtest.engine import load_candles, load_funding_rates, _atr_pct_df
from signals.tasks import _detect_signal

inst = Instrument.objects.get(symbol="BTCUSDT")
start = dj_tz.make_aware(datetime(2025, 7, 1))
end = dj_tz.make_aware(datetime(2025, 7, 5))
lookback_start = start - timedelta(days=30)
lookback_bars = 200

df_ltf = load_candles(inst, "5m", lookback_start, end)
df_htf = load_candles(inst, "4h", lookback_start, end)
print(f"LTF: {len(df_ltf)} candles, HTF: {len(df_htf)} candles")

# Build tz-naive arrays like engine
idx_ltf = df_ltf.index
if hasattr(idx_ltf, 'tz') and idx_ltf.tz:
    ts_ltf = idx_ltf.tz_localize(None).values
else:
    ts_ltf = idx_ltf.values

idx_htf = df_htf.index
if hasattr(idx_htf, 'tz') and idx_htf.tz:
    ts_htf = idx_htf.tz_localize(None).values
else:
    ts_htf = idx_htf.values

# Timeline
mask = df_ltf.index >= start
bar_timestamps = df_ltf.index[mask].tolist()
print(f"Timeline: {len(bar_timestamps)} bars")

equity = 1000.0
signal_count = 0
skip_reasons = {"cooldown": 0, "dd": 0, "ltf_short": 0, "htf_short": 0,
                "no_signal": 0, "margin": 0, "exposure": 0, "opened": 0, "qty_zero": 0}

from datetime import timedelta as td
cooldown_until = None

for bar_ts in bar_timestamps[:2000]:
    bar_ts_naive = np.datetime64(bar_ts.replace(tzinfo=None))
    
    pos_idx = np.searchsorted(ts_ltf, bar_ts_naive, side="right")
    if pos_idx == 0 or ts_ltf[pos_idx - 1] != bar_ts_naive:
        continue

    bar = df_ltf.iloc[pos_idx - 1]
    close_price = float(bar["close"])

    # Cooldown check
    if cooldown_until and bar_ts < cooldown_until:
        skip_reasons["cooldown"] += 1
        continue

    # Slice
    start_ltf = max(0, pos_idx - lookback_bars)
    df_ltf_slice = df_ltf.iloc[start_ltf:pos_idx]

    htf_pos = np.searchsorted(ts_htf, bar_ts_naive, side="right")
    start_htf = max(0, htf_pos - lookback_bars)
    df_htf_slice = df_htf.iloc[start_htf:htf_pos]

    if len(df_ltf_slice) < 30:
        skip_reasons["ltf_short"] += 1
        continue
    if len(df_htf_slice) < 30:
        skip_reasons["htf_short"] += 1
        continue

    # Funding
    funding_rates = []

    ok, direction, explain, score = _detect_signal(df_ltf_slice, df_htf_slice, funding_rates)
    if not ok:
        skip_reasons["no_signal"] += 1
        continue

    signal_count += 1

    # Sizing
    atr = _atr_pct_df(df_ltf_slice)
    side = "buy" if direction == "long" else "sell"
    tp_pct = settings.TAKE_PROFIT_PCT
    sl_pct = settings.STOP_LOSS_PCT
    if atr and atr > 0:
        tp_pct = max(tp_pct, atr * settings.ATR_MULT_TP)
        sl_pct = max(sl_pct, atr * settings.ATR_MULT_SL)

    stop_dist = sl_pct
    risk_amount = settings.RISK_PER_TRADE_PCT * equity
    qty = risk_amount / (stop_dist * close_price)
    qty = round(qty, 3)

    notional = close_price * qty
    leverage = settings.MAX_EFF_LEVERAGE
    required_margin = notional / leverage if leverage > 0 else notional
    max_exposure = equity * settings.MAX_EXPOSURE_PER_INSTRUMENT_PCT * leverage

    if signal_count <= 5:
        print(f"\nSignal #{signal_count} at {bar_ts}: {direction} score={score}")
        print(f"  close={close_price:.2f}, atr={atr:.6f}, sl_pct={sl_pct:.4f}, tp_pct={tp_pct:.4f}")
        print(f"  risk_amount={risk_amount:.2f}, qty={qty:.6f}, notional={notional:.2f}")
        print(f"  required_margin={required_margin:.2f}, max_equity_margin={equity*0.95:.2f}")
        print(f"  max_exposure={max_exposure:.2f}")

    if required_margin > equity * 0.95:
        max_notional = equity * 0.95 * leverage
        qty = round(max_notional / close_price, 3)
        notional = close_price * qty
        if qty <= 0:
            skip_reasons["margin"] += 1
            continue
        if signal_count <= 5:
            print(f"  -> adjusted qty={qty:.6f}, notional={notional:.2f} (margin cap)")

    if notional > max_exposure:
        qty = round(max_exposure / close_price, 3)
        notional = close_price * qty
        if qty <= 0:
            skip_reasons["exposure"] += 1
            continue
        if signal_count <= 5:
            print(f"  -> adjusted qty={qty:.6f}, notional={notional:.2f} (exposure cap)")

    if qty <= 0:
        skip_reasons["qty_zero"] += 1
        continue

    skip_reasons["opened"] += 1
    cooldown_until = bar_ts + td(minutes=120)
    if signal_count <= 5:
        print(f"  -> TRADE OPENED! qty={qty}, side={side}")

print(f"\nTotal signals: {signal_count}")
print(f"Skip reasons: {skip_reasons}")
print(f"Settings: TP={settings.TAKE_PROFIT_PCT}, SL={settings.STOP_LOSS_PCT}, ATR_TP={settings.ATR_MULT_TP}, ATR_SL={settings.ATR_MULT_SL}, leverage={settings.MAX_EFF_LEVERAGE}, exposure_cap={settings.MAX_EXPOSURE_PER_INSTRUMENT_PCT}")
