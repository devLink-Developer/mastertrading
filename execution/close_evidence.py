"""Bounded, read-only close execution evidence for linear perpetual contracts.

No Django/application imports or order submission. Unified CCXT fields are used
only as execution evidence; requested amount/price and ticker are never fills.
Fees are costs: a positive fee is paid and a negative fee is a rebate. This
module resolves the close fee only; the caller owns separate entry-fee handling.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import re
from typing import Any


SUPPORTED_ASSETS = frozenset({"USDT", "USDC", "VST"})
ZERO = Decimal("0")
QUANTITY_TOLERANCE = Decimal("0.0000000001")


@dataclass(frozen=True)
class CloseEvidence:
    status: str = "unknown"
    source: str = "none"
    order_id: str | None = None
    filled_qty: Decimal | None = None
    average_price: Decimal | None = None
    fee: Decimal | None = None
    fee_currency: str | None = None
    fee_known: bool = False
    filled_timestamp_ms: int | None = None
    is_full_close: bool = False
    reason: str = "missing_order_evidence"


def _number(value: Any) -> Decimal | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return None
    return number if number.is_finite() else None


def _positive(value: Any) -> Decimal | None:
    number = _number(value)
    return number if number is not None and number > ZERO else None


def _symbol(value: Any) -> str:
    return re.sub(r"[-_/\s]", "", str(value or "").split(":")[0]).upper()


def _timestamp(value: Any) -> int | None:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return None
        delta = value.astimezone(timezone.utc) - datetime(1970, 1, 1, tzinfo=timezone.utc)
        return delta.days * 86400000 + delta.seconds * 1000 + delta.microseconds // 1000
    if isinstance(value, str) and not value.isdigit():
        try:
            return _timestamp(datetime.fromisoformat(value.replace("Z", "+00:00")))
        except ValueError:
            return None
    number = _positive(value)
    # Exchange timestamps are milliseconds. Reject unlabelled Unix seconds.
    return int(number) if number is not None and number >= Decimal("100000000000") else None


def _raw_order(order: dict) -> dict:
    raw = order.get("info")
    if not isinstance(raw, dict):
        return {}
    for key in ("newOrderResponse", "orderOpenResponse"):
        if isinstance(raw.get(key), dict):
            return raw[key]
    return raw


def _first(mapping: dict, *keys: str) -> Any:
    for key in keys:
        if mapping.get(key) is not None:
            return mapping[key]
    return None


def _order_id(order: dict) -> str | None:
    value = order.get("id") or _first(_raw_order(order), "orderId", "i", "mainOrderId")
    return str(value) if value is not None and str(value).strip() else None


def _filled_time(order: dict) -> int | None:
    # timestamp/datetime commonly mean order creation, so never use them as fill time.
    value = _first(order, "lastTradeTimestamp", "lastUpdateTimestamp")
    if value is None:
        value = _first(_raw_order(order), "updateTime", "T", "dealTime", "filledTime")
    return _timestamp(value)


def _fee_evidence(order: dict, asset: str) -> tuple[Decimal | None, bool, str]:
    if asset not in SUPPORTED_ASSETS:
        return None, False, "unsupported_account_asset"
    fees = order.get("fees")
    # CCXT fee is usually a summary/duplicate of fees, never an extra charge.
    if not isinstance(fees, list) or not fees:
        fee = order.get("fee")
        fees = [fee] if isinstance(fee, dict) else []
    if not fees:
        return None, False, "missing_fee"
    total, identified, anonymous = ZERO, {}, set()
    for fee in fees:
        if not isinstance(fee, dict):
            return None, False, "invalid_fee"
        amount = _number(fee.get("cost"))
        currency = str(fee.get("currency") or "").upper()
        if amount is None or currency != asset:
            return None, False, "missing_or_different_fee_currency"
        ident = _first(fee, "id", "transactionId", "tranId", "tradeId")
        signature = (currency, amount)
        if ident is not None:
            ident = str(ident)
            if ident in identified:
                if identified[ident] != signature:
                    return None, False, "conflicting_fee_identity"
                continue
            identified[ident] = signature
        else:
            # Equal anonymous components may be two fills OR an accidental
            # duplicate. Their identity cannot be established from amounts.
            if signature in anonymous:
                return None, False, "ambiguous_duplicate_fee"
            anonymous.add(signature)
        total += amount
    return total, True, "fee_verified"


def _position_side(order: dict) -> str:
    value = order.get("positionSide") or _first(_raw_order(order), "positionSide", "ps")
    return str(value or "").lower()


def _reduce_only(order: dict) -> bool:
    value = order.get("reduceOnly")
    if value is None:
        value = _first(_raw_order(order), "reduceOnly", "ro")
    return value is True or str(value).strip().lower() in {"true", "1"}


def _normalize(order: Any, *, symbol: str, expected_qty: Decimal, account_asset: str,
               contract_size: Decimal, expected_order_id: str | None, close_side: str | None,
               position_side: str | None, source: str) -> CloseEvidence:
    if not isinstance(order, dict):
        return CloseEvidence(source=source, reason="missing_order_evidence")
    ident = _order_id(order)
    if not ident or (expected_order_id is not None and ident != expected_order_id):
        return CloseEvidence(source=source, reason="missing_or_mismatched_order_id")
    raw = _raw_order(order)
    raw_id = _first(raw, "orderId", "i", "mainOrderId")
    if raw_id is not None and str(raw_id) != ident:
        return CloseEvidence(source=source, order_id=ident, reason="missing_or_mismatched_order_id")
    actual_symbol = order.get("symbol") or _first(raw, "symbol", "s")
    actual_side = str(order.get("side") or _first(raw, "side", "S") or "").lower()
    raw_symbol = _first(raw, "symbol", "s")
    raw_side = str(_first(raw, "side", "S") or "").lower()
    # Missing identity fields are tolerated for an exact-ID response only;
    # historical candidate selection below requires every identifying field.
    if actual_symbol and _symbol(actual_symbol) != _symbol(symbol):
        return CloseEvidence(source=source, order_id=ident, reason="symbol_mismatch")
    if raw_symbol and _symbol(raw_symbol) != _symbol(symbol):
        return CloseEvidence(source=source, order_id=ident, reason="symbol_mismatch")
    if close_side and actual_side and actual_side != close_side.lower():
        return CloseEvidence(source=source, order_id=ident, reason="side_mismatch")
    if raw_side and actual_side and raw_side != actual_side:
        return CloseEvidence(source=source, order_id=ident, reason="side_mismatch")
    actual_position_side = _position_side(order)
    if position_side and actual_position_side not in ("", "both", position_side.lower()):
        return CloseEvidence(source=source, order_id=ident, reason="position_side_mismatch")

    has_raw_order = any(key in raw for key in ("executedQty", "origQty", "avgPrice", "orderId", "price", "status",
                                               "i", "q", "X", "ap", "z", "p"))
    executed = _first(raw, "executedQty", "z", "dealSize", "filledSize")
    # CCXT can infer filled=amount from status=closed. If the retained raw
    # payload exists, require an executed quantity rather than that inference.
    filled = _number(executed if executed is not None else None if has_raw_order else order.get("filled"))
    if filled is not None and filled < ZERO:
        filled = None
    raw_average = _first(raw, "avgPrice", "ap", "averagePrice", "dealPrice")
    average = _positive(raw_average if raw_average is not None else order.get("average"))
    if filled is None or filled <= ZERO:
        average = None
    elif average is None:
        raw_cost = _first(raw, "cumQuote", "cummulativeQuoteQty", "executedQuoteQty", "dealFunds")
        # safe_order may synthesize unified cost from the requested price. A
        # raw exchange payload must carry executed quote evidence for fallback.
        cost = _positive(raw_cost if raw_cost is not None else None if has_raw_order else order.get("cost"))
        if cost is not None:
            average = cost / (filled * contract_size)
    fee, fee_known, fee_reason = _fee_evidence(order, account_asset)
    status = str(order.get("status") or _first(raw, "status", "X", "orderStatus") or "").lower()
    remaining = _number(order.get("remaining"))
    quantity_matches = filled is not None and abs(filled - expected_qty) <= QUANTITY_TOLERANCE
    terminal = status in {"closed", "filled"}
    full = bool(quantity_matches and filled > ZERO and average is not None and terminal
                and (remaining is None or abs(remaining) <= QUANTITY_TOLERANCE))
    if full:
        reason = "verified_fill" if fee_known else fee_reason
        evidence_status = "filled"
    elif filled is not None and ZERO < filled < expected_qty - QUANTITY_TOLERANCE:
        reason, evidence_status = "partial_fill", "partial"
    elif filled is not None and not quantity_matches:
        reason, evidence_status = "filled_quantity_mismatch", "unknown"
    elif average is None:
        reason, evidence_status = "missing_executed_average", "unknown"
    else:
        reason, evidence_status = "order_not_fully_closed", "unknown"
    return CloseEvidence(status=evidence_status, source=source, order_id=ident,
                         filled_qty=filled, average_price=average, fee=fee,
                         fee_currency=account_asset if fee_known else None, fee_known=fee_known,
                         filled_timestamp_ms=_filled_time(order), is_full_close=full, reason=reason)


def _parameters(expected_qty: Any, contract_size: Any, account_asset: str):
    qty, size = _positive(expected_qty), _positive(contract_size)
    asset = str(account_asset or "").upper()
    if qty is None or size is None or asset not in SUPPORTED_ASSETS:
        return None
    return qty, size, asset


def resolve_close_evidence(adapter: Any, symbol: str, expected_qty: Any, *,
                           order_response: dict | None = None, order_id: Any = None,
                           account_asset: str = "USDT", contract_size: Any = 1,
                           close_side: str | None = None,
                           position_side: str | None = None) -> CloseEvidence:
    """Normalize a close response; if incomplete read its exact order ID once.

    `is_full_close` verifies quantity, execution average and terminal status.
    It does NOT imply `fee_known`; callers must check both before declaring net
    accounting complete. No entry fee is included in this evidence.
    """
    params = _parameters(expected_qty, contract_size, account_asset)
    if params is None or not _symbol(symbol):
        return CloseEvidence(reason="invalid_quantity_contract_or_account_asset")
    qty, size, asset = params
    ident = str(order_id) if order_id is not None else _order_id(order_response or {})
    options = dict(symbol=symbol, expected_qty=qty, account_asset=asset, contract_size=size,
                   expected_order_id=ident, close_side=close_side, position_side=position_side)
    evidence = _normalize(order_response, source="order_response", **options)
    if evidence.is_full_close and evidence.fee_known:
        return evidence
    fetch = getattr(getattr(adapter, "client", None), "fetch_order", None)
    if ident is None or not callable(fetch):
        return evidence
    try:
        mapper = getattr(adapter, "_map_symbol", None)
        mapped = mapper(symbol) if callable(mapper) else symbol
        fetched = fetch(ident, mapped)
    except Exception as exc:
        # Never include raw API exception text: it may contain request secrets.
        return replace(evidence, reason=f"{evidence.reason};fetch_order_failed:{type(exc).__name__}")
    if not isinstance(fetched, dict) or not _order_id(fetched):
        return replace(evidence, order_id=ident, reason=f"{evidence.reason};fetch_order_missing_evidence")
    resolved = _normalize(fetched, source="fetch_order", **options)
    # Do not join two snapshots' fee/fill fields: one could be partial/stale.
    # A read returning an explicit identity mismatch cannot corroborate a fill.
    if resolved.reason in {"missing_or_mismatched_order_id", "symbol_mismatch", "side_mismatch", "position_side_mismatch"}:
        return replace(resolved, order_id=ident)
    if (evidence.filled_qty is not None and evidence.filled_qty > ZERO
            and (resolved.filled_qty is None or resolved.filled_qty < evidence.filled_qty)):
        return evidence
    return resolved if resolved.is_full_close or not evidence.is_full_close else evidence


def resolve_exchange_close_evidence(adapter: Any, symbol: str, expected_qty: Any, *,
                                    close_side: str, position_side: str,
                                    opened_at: Any, closed_at: Any, account_asset: str = "USDT",
                                    contract_size: Any = 1, tolerance_seconds: int = 120) -> CloseEvidence:
    """Find one unambiguous fully filled reduce-only close in bounded history.

    Every candidate needs an explicit order ID, exact pair/opposite side,
    matching LONG/SHORT positionSide (or BOTH for a reduce-only one-way close),
    full expected quantity, fill/update timestamp after entry and within the
    close-time tolerance. More than one candidate or a full page is unknown.
    """
    params = _parameters(expected_qty, contract_size, account_asset)
    opened, closed = _timestamp(opened_at), _timestamp(closed_at)
    close_side, position_side = str(close_side or "").lower(), str(position_side or "").lower()
    expected_side = {"long": "sell", "short": "buy", "buy": "sell", "sell": "buy"}.get(position_side)
    position_side = {"buy": "long", "sell": "short"}.get(position_side, position_side)
    if (params is None or opened is None or closed is None or opened >= closed
            or not isinstance(tolerance_seconds, int) or not 0 <= tolerance_seconds <= 120
            or expected_side != close_side):
        return CloseEvidence(reason="missing_or_invalid_position_identity")
    qty, size, asset = params
    tolerance = tolerance_seconds * 1000
    fetch = getattr(getattr(adapter, "client", None), "fetch_closed_orders", None)
    if not callable(fetch):
        return CloseEvidence(reason="closed_order_history_unavailable")
    try:
        mapper = getattr(adapter, "_map_symbol", None)
        mapped = mapper(symbol) if callable(mapper) else symbol
        # CCXT's `since` filters creation timestamps. A protective stop can be
        # created at entry and fill much later; filter its fill time below.
        orders = fetch(mapped, since=opened, limit=50)
    except Exception as exc:
        return CloseEvidence(source="fetch_closed_orders", reason=f"history_fetch_failed:{type(exc).__name__}")
    if not isinstance(orders, list) or len(orders) >= 50:
        return CloseEvidence(source="fetch_closed_orders", reason="missing_or_truncated_order_history")
    candidates = {}
    for order in orders:
        if not isinstance(order, dict):
            continue
        raw = _raw_order(order)
        if (_symbol(order.get("symbol") or _first(raw, "symbol", "s")) != _symbol(symbol)
                or str(order.get("side") or _first(raw, "side", "S") or "").lower() != close_side
                or not _reduce_only(order) or _position_side(order) not in {position_side, "both"}):
            continue
        stamp = _filled_time(order)
        if stamp is None or not opened < stamp or abs(stamp - closed) > tolerance:
            continue
        evidence = _normalize(order, symbol=symbol, expected_qty=qty, account_asset=asset,
                              contract_size=size, expected_order_id=None, close_side=close_side,
                              position_side=position_side, source="fetch_closed_orders")
        if not evidence.is_full_close:
            continue
        if evidence.order_id in candidates and candidates[evidence.order_id] != evidence:
            return CloseEvidence(status="ambiguous", source="fetch_closed_orders", reason="conflicting_order_identity")
        candidates[evidence.order_id] = evidence
    if len(candidates) != 1:
        return CloseEvidence(status="ambiguous" if candidates else "unknown", source="fetch_closed_orders",
                             reason="multiple_matching_closes" if candidates else "no_matching_close")
    return next(iter(candidates.values()))
