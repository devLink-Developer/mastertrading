from adapters import get_default_adapter

adapter = get_default_adapter()
orders = adapter.fetch_open_orders()
print(f"Open orders on exchange: {len(orders or [])}")
for o in (orders or []):
    info = o.get("info", {})
    trigger = o.get("triggerPrice") or info.get("stopPrice") or info.get("triggerPrice") or "none"
    reduce_only = o.get("reduceOnly") or info.get("reduceOnly")
    print(f"  sym={o.get('symbol')} side={o.get('side')} type={o.get('type')} trigger={trigger} reduce={reduce_only} id={o.get('id')}")

positions = adapter.fetch_positions()
for pos in (positions or []):
    qty = pos.get("contracts") or pos.get("size") or 0
    if float(qty) == 0:
        continue
    print(f"Position: {pos.get('symbol')} side={pos.get('side')} qty={qty} entry={pos.get('entryPrice')} mark={pos.get('markPrice')}")
