"""Quick script to check open stop orders on exchange."""
import os, django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

from adapters import get_default_adapter

adapter = get_default_adapter()
orders = adapter.fetch_open_orders()
print(f"Open orders on exchange: {len(orders or [])}")
for o in (orders or []):
    info = o.get("info", {})
    trigger = o.get("triggerPrice") or info.get("stopPrice") or info.get("triggerPrice") or "none"
    reduce = o.get("reduceOnly") or info.get("reduceOnly")
    print(f"  {o.get('symbol')} {o.get('side')} type={o.get('type')} "
          f"trigger={trigger} reduce={reduce} status={o.get('status')} id={o.get('id')}")

# Also check positions
positions = adapter.fetch_positions()
for pos in (positions or []):
    qty = pos.get("contracts") or pos.get("size") or 0
    if float(qty) == 0:
        continue
    print(f"\nPosition: {pos.get('symbol')} side={pos.get('side')} qty={qty} "
          f"entry={pos.get('entryPrice')} mark={pos.get('markPrice')}")
