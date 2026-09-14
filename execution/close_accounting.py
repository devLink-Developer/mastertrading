"""Durable close evidence and bounded reconciliation, without order submission.

One Order stores one exchange close order's cumulative execution evidence.
Compatibility numeric Order fields may default to zero when evidence is absent;
the close_accounting evidence/known flags are authoritative for accounting.
No TradeFill is synthesized without an exchange trade identity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta, timezone as datetime_timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import uuid
from typing import Any

from django.db import transaction
from django.db.models.fields.json import KeyTextTransform, KeyTransform
from django.utils import timezone

from core.exchange_runtime import get_runtime_exchange_context
from core.models import Instrument
from execution.close_evidence import CloseEvidence, resolve_close_evidence, resolve_exchange_close_evidence
from execution.models import OperationReport, Order


VERSION = 1
TOLERANCE = Decimal("0.0000000001")


@dataclass(frozen=True)
class PositionCloseSummary:
    status: str = "pending"
    reason: str = "missing_close_evidence"
    position_key: str = ""
    filled_qty: Decimal | None = None
    average_price: Decimal | None = None
    exit_fee: Decimal | None = None
    closed_at: datetime | None = None
    final_order_id: int | None = None
    context: dict = field(default_factory=dict)
    leg_order_ids: tuple[int, ...] = ()


def _decimal(value):
    try:
        number = Decimal(str(value))
        return number if number.is_finite() else None
    except (ValueError, TypeError, InvalidOperation):
        return None


def _date(value):
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    if not isinstance(value, datetime) or timezone.is_naive(value):
        return None
    return value.astimezone(datetime_timezone.utc)


def _json(value):
    return json.loads(json.dumps(value, default=lambda item: item.isoformat() if isinstance(item, datetime) else str(item)))


def _metadata(order):
    raw = order.raw_response or {}
    return raw.get("close_accounting", {}) if isinstance(raw, dict) else {}


def fee_asset_for_context(context: dict) -> str:
    """Explicit BingX demo virtual-settlement mapping, verified against ledger.

    CCXT labels the commission USDT in the verified BingX sandbox response,
    while that sandbox's ledger settles the same numeric amount in VST.
    This is not a general conversion among stablecoins or accounts.
    """
    runtime = context.get("runtime", context)
    asset = str(runtime.get("primary_asset") or context.get("account_asset") or "").upper()
    if (runtime.get("service") == "bingx" and runtime.get("sandbox") is True
            and runtime.get("mode") == "demo" and asset == "VST"):
        return "USDT"
    return asset


def _position_key(inst, context):
    if context.get("lifecycle_conflict_key"):
        return context["lifecycle_conflict_key"]
    identity = [context["namespace"], inst.pk, inst.symbol, context["side"]]
    if context["correlation_id"]:
        identity.extend(["correlation", context["correlation_id"]])
    elif context["opened_at"]:
        identity.extend(["opened_at", context["opened_at"]])
    else:
        identity.extend(["unidentified_intent", context.get("intent_token")])
    return hashlib.sha256(json.dumps(identity, separators=(",", ":")).encode()).hexdigest()


def _resolve(adapter, inst, context, response=None, exact_id=None):
    side = "sell" if context["side"] == "buy" else "buy"
    position_side = "long" if context["side"] == "buy" else "short"
    options = dict(account_asset=context["exchange_fee_asset"], contract_size=context["contract_size"])
    if exact_id or response:
        evidence = resolve_close_evidence(adapter, inst.symbol, context["requested_qty"],
                    order_response=response, order_id=exact_id, close_side=side, position_side=position_side, **options)
        if evidence.order_id or exact_id:
            return evidence
    # A missing acknowledgement ID can be recovered only from one strict,
    # fully identified historical candidate in the persisted submission window.
    return resolve_exchange_close_evidence(adapter, inst.symbol, context["requested_qty"],
                close_side=side, position_side=position_side, opened_at=context["opened_at"],
                closed_at=context["closed_at"], **options)


def _context_valid(context):
    entry = _decimal(context.get("entry_price"))
    return bool(context.get("namespace") and context.get("side") in {"buy", "sell"}
                and _date(context.get("opened_at")) is not None and entry is not None and entry > 0
                and context.get("mode") and context.get("account_asset"))


def _leg_timestamp(context, stamp):
    if stamp is None:
        return None
    try:
        value = datetime.fromtimestamp(int(stamp) / 1000, datetime_timezone.utc)
    except (TypeError, ValueError, OverflowError):
        return None
    opened, intent = _date(context.get("opened_at")), _date(context.get("closed_at"))
    # Some exchange responses round their timestamps to the second while the
    # database preserves microseconds. Allow only that same-instant rounding.
    if (opened is None or intent is None or value < opened.replace(microsecond=0)
            or value > intent + timedelta(seconds=120)):
        return None
    return value


def _save_evidence(inst, context, evidence, response):
    key = _position_key(inst, context)
    with transaction.atomic():
        Instrument.objects.select_for_update().get(pk=inst.pk)
        qs = Order.objects.filter(instrument=inst, reduce_only=True,
                    raw_response__close_accounting__namespace=context["namespace"])
        exact = list(qs.filter(exchange_order_id=evidence.order_id).order_by("pk")[:2]) if evidence.order_id else []
        if len(exact) > 1:
            raise ValueError("Conflicting durable close rows share an exchange order ID")
        order = exact[0] if exact else None
        if order is not None and context["correlation_id"]:
            previous_context = _metadata(order).get("context", {})
            old_open, new_open = _date(previous_context.get("opened_at")), _date(context["opened_at"])
            if old_open and new_open and abs(old_open - new_open) > timedelta(seconds=120):
                raise ValueError("correlation_lifecycle_conflict: exact order cannot change lifecycle")
            if previous_context.get("lifecycle_conflict_key"):
                context, key = previous_context, _metadata(order)["position_key"]
        if order is None and context["correlation_id"] and not context.get("lifecycle_conflict"):
            anchor = qs.filter(raw_response__close_accounting__position_key=key).order_by("pk").first()
            if anchor is not None:
                anchored_open = _date(_metadata(anchor).get("context", {}).get("opened_at"))
                incoming_open = _date(context["opened_at"])
                if (anchored_open is None or incoming_open is None
                        or abs(anchored_open - incoming_open) > timedelta(seconds=120)):
                    conflict_identity = [key, context["opened_at"], evidence.order_id or "unidentified",
                                         context["is_partial"]]
                    conflict_key = hashlib.sha256(json.dumps(conflict_identity).encode()).hexdigest()
                    context = dict(context, lifecycle_conflict=True, lifecycle_conflict_key=conflict_key,
                                   canonical_position_key=key)
                    key = conflict_key
        if order is not None and _metadata(order).get("position_key") != key:
            raise ValueError("An exchange order ID cannot be assigned to two positions")
        if order is None:
            pending = list(qs.filter(exchange_order_id="", raw_response__close_accounting__position_key=key,
                                raw_response__close_accounting__is_partial=context["is_partial"]).order_by("pk")[:2])
            if len(pending) > 1:
                raise ValueError("Multiple unidentified close legs require explicit reconciliation")
            order = pending[0] if pending else None
        previous = _metadata(order) if order is not None else {}
        if previous:
            old_context = previous.get("context", {})
            if (_decimal(old_context.get("requested_qty")) != _decimal(context["requested_qty"])
                    or old_context.get("is_partial") != context["is_partial"]):
                raise ValueError("The same exchange order cannot change its requested close-leg identity")
            context = old_context  # Preserve original close-time bounds and reporting context for retries.
            old_evidence = previous.get("evidence", {})
            old_qty = _decimal(old_evidence.get("filled_qty"))
            if old_qty is not None and (evidence.filled_qty is None or evidence.filled_qty < old_qty):
                _touch_preserved_evidence(order, previous, evidence.reason)
                return order
            if (old_evidence.get("is_full_close") and old_evidence.get("fee_known")
                    and not (evidence.is_full_close and evidence.fee_known)):
                _touch_preserved_evidence(order, previous, evidence.reason)
                return order
        data = {"version": VERSION, "namespace": context["namespace"], "position_key": key,
                "context": context, "evidence": _json(asdict(evidence)), "is_partial": context["is_partial"],
                "status": "captured" if context["is_partial"] and evidence.is_full_close and evidence.fee_known
                           and _leg_timestamp(context, evidence.filled_timestamp_ms) is not None
                           and not context.get("lifecycle_conflict") else "pending",
                "attempts": int(previous.get("attempts", 0)) + 1, "last_checked_at": timezone.now().isoformat()}
        payload = dict(order.raw_response or {}) if order is not None else {}
        if response is not None:
            payload["exchange_response"] = _json(response)
        payload["close_accounting"] = data
        if order is None:
            order = Order(instrument=inst, side="sell" if context["side"] == "buy" else "buy", type="market",
                          reduce_only=True, correlation_id=context["correlation_id"][:64],
                          parent_correlation_id=context["correlation_id"][:64], opened_at=timezone.now())
        order.exchange_order_id = evidence.order_id or order.exchange_order_id or ""
        order.qty = evidence.filled_qty if evidence.filled_qty is not None else Decimal(context["requested_qty"])
        order.price = evidence.average_price
        order.fee_usdt = evidence.fee if evidence.fee_known else Decimal("0")
        order.notional_usdt = (evidence.filled_qty * evidence.average_price * Decimal(context["contract_size"])
                               if evidence.filled_qty is not None and evidence.average_price is not None else Decimal("0"))
        order.leverage = _decimal(context["leverage"]) or Decimal("0")
        order.status = "filled" if evidence.is_full_close else "partially_filled" if evidence.filled_qty else "new"
        order.status_reason = f"close_accounting:{evidence.reason}"[:255]
        order.closed_at = (datetime.fromtimestamp(evidence.filled_timestamp_ms / 1000, datetime_timezone.utc)
                           if evidence.filled_timestamp_ms is not None else None)
        order.raw_response = payload
        order.save()
        return order


def _touch_preserved_evidence(order, metadata, reason):
    metadata["attempts"] = int(metadata.get("attempts", 0)) + 1
    metadata["last_checked_at"] = timezone.now().isoformat()
    metadata["last_incomplete_fetch_reason"] = reason
    order.raw_response["close_accounting"] = metadata
    order.save(update_fields=["raw_response", "updated_at"])


def capture_close(adapter, *, inst, side, qty, entry_price, reason, signal_id, correlation_id,
                  leverage, equity_before, opened_at, contract_size=1, order_response=None,
                  exchange_closed=False, is_partial=False):
    """Persist a close acknowledgement/fill once; return (Order, summary)."""
    runtime = get_runtime_exchange_context()
    requested, size = _decimal(qty), _decimal(contract_size)
    if requested is None or requested <= 0 or size is None or size <= 0:
        raise ValueError("Close capture requires positive requested quantity and contract size")
    opened = _date(opened_at)
    fee_asset = fee_asset_for_context(runtime)
    account_asset = str(runtime.get("primary_asset") or "").upper()
    context = {"namespace": str(runtime.get("risk_namespace") or ""), "runtime": _json(runtime),
               "mode": str(runtime.get("mode") or ""), "account_asset": account_asset,
               "exchange_fee_asset": fee_asset,
               "fee_asset_mapping": "bingx_demo_virtual_settlement" if fee_asset != account_asset else "exact",
               "side": str(side).lower(), "requested_qty": str(requested), "entry_price": str(entry_price),
               "reason": str(reason or ""), "signal_id": str(signal_id or ""),
               "correlation_id": str(correlation_id or ""), "leverage": str(leverage or 0),
               "equity_before": str(equity_before) if equity_before is not None else None,
               "opened_at": opened.isoformat() if opened else None, "closed_at": timezone.now().isoformat(),
               "contract_size": str(size), "exchange_closed": bool(exchange_closed), "is_partial": bool(is_partial)}
    evidence = _resolve(adapter, inst, context, order_response)
    if not context["opened_at"] and not context["correlation_id"]:
        context["intent_token"] = f"order:{evidence.order_id}" if evidence.order_id else uuid.uuid4().hex
    order = _save_evidence(inst, context, evidence, order_response)
    return order, summarize_position(order)


def summarize_position(order) -> PositionCloseSummary:
    """Aggregate cumulative close-order legs exactly once, never entry fees."""
    metadata = _metadata(order)
    key, context = metadata.get("position_key", ""), metadata.get("context", {})
    summary = PositionCloseSummary(position_key=key, context=context)
    if context.get("lifecycle_conflict"):
        return replace(summary, reason="correlation_lifecycle_conflict")
    if not key or not _context_valid(context):
        return replace(summary, reason="missing_position_context")
    legs = list(Order.objects.filter(instrument_id=order.instrument_id, reduce_only=True,
                raw_response__close_accounting__namespace=metadata["namespace"],
                raw_response__close_accounting__position_key=key).order_by("pk"))
    summary = replace(summary, leg_order_ids=tuple(leg.pk for leg in legs))
    finals = [leg for leg in legs if not _metadata(leg).get("is_partial")]
    if len(finals) > 1:
        return replace(summary, reason="multiple_final_close_legs")
    if finals:
        final = finals[0]
        # The reporting intent belongs to the unique final close even when a
        # partial leg still lacks evidence. Never inherit partial qty/reason.
        context = _metadata(final).get("context", {})
        summary = replace(summary, final_order_id=final.pk, context=context, closed_at=final.closed_at)
        if not _context_valid(context):
            return replace(summary, reason="missing_final_position_context")
    complete = []
    identities = set()
    for leg in legs:
        leg_metadata = _metadata(leg)
        evidence = leg_metadata.get("evidence", {})
        ident = leg.exchange_order_id
        if not ident or ident in identities:
            return replace(summary, reason="missing_or_duplicate_close_order_identity")
        identities.add(ident)
        qty, price, fee = (_decimal(evidence.get(name)) for name in ("filled_qty", "average_price", "fee"))
        if (not evidence.get("is_full_close") or not evidence.get("fee_known")
                or qty is None or qty <= 0 or price is None or price <= 0 or fee is None):
            return replace(summary, reason="close_leg_evidence_incomplete")
        stamp = evidence.get("filled_timestamp_ms")
        if stamp is None:
            return replace(summary, reason="close_leg_timestamp_missing")
        leg_time = _leg_timestamp(leg_metadata.get("context", {}), stamp)
        if leg_time is None:
            return replace(summary, reason="close_leg_timestamp_outside_lifecycle")
        if (finals and final.closed_at is not None
                and leg_time.replace(microsecond=0) > final.closed_at.replace(microsecond=0)):
            return replace(summary, reason="close_leg_timestamp_after_final")
        complete.append((leg, qty, price, fee))
    if not finals:
        return replace(summary, status="partial", reason="no_final_close_leg")
    if final.closed_at is None:
        return replace(summary, reason="missing_final_fill_timestamp")
    quantity = sum((qty for _, qty, _, _ in complete), Decimal("0"))
    opened = _date(context["opened_at"])
    entries = Order.objects.filter(instrument_id=order.instrument_id, reduce_only=False, side=context["side"],
                status=Order.OrderStatus.FILLED, opened_at__gte=opened - timedelta(seconds=120),
                opened_at__lte=opened + timedelta(seconds=120))
    if context["correlation_id"]:
        entries = entries.filter(correlation_id=context["correlation_id"])
    entries = list(entries[:2])
    if len(entries) > 1:
        return replace(summary, reason="ambiguous_entry_quantity")
    if not entries:
        return replace(summary, reason="missing_root_entry_quantity")
    if entries:
        root_entry = entries[0]
        children = []
        if context["correlation_id"]:
            children = list(Order.objects.filter(instrument_id=order.instrument_id, reduce_only=False,
                    side=context["side"], status=Order.OrderStatus.FILLED,
                    parent_correlation_id=context["correlation_id"],
                    opened_at__gte=root_entry.opened_at, opened_at__lte=final.closed_at
                    ).exclude(pk=root_entry.pk).order_by("pk")[:20])
            if len(children) >= 20:
                return replace(summary, reason="too_many_entry_legs_to_reconcile")
        explicit_ids = [entry.exchange_order_id for entry in [root_entry, *children] if entry.exchange_order_id]
        if len(set(explicit_ids)) != len(explicit_ids):
            return replace(summary, reason="duplicate_entry_order_identity")
        entry_quantity = sum((entry.qty for entry in [root_entry, *children]), Decimal("0"))
        if quantity > entry_quantity + TOLERANCE:
            return replace(summary, reason="closed_quantity_exceeds_entry_quantity")
        if quantity < entry_quantity - TOLERANCE:
            return replace(summary, reason="close_legs_do_not_cover_entry_quantity")
    vwap = sum((qty * price for _, qty, price, _ in complete), Decimal("0")) / quantity
    fees = sum((fee for _, _, _, fee in complete), Decimal("0"))
    return replace(summary, status="confirmed", reason="all_close_legs_verified", context=context,
                   filled_qty=quantity, average_price=vwap, exit_fee=fees, closed_at=final.closed_at,
                   final_order_id=final.pk)


def refresh_close(adapter, order):
    """Read pending order evidence from persisted context after position removal."""
    metadata = _metadata(order)
    context = metadata.get("context", {})
    runtime = get_runtime_exchange_context()
    if context.get("namespace") != runtime.get("risk_namespace"):
        return order, replace(summarize_position(order), status="pending", reason="runtime_namespace_mismatch")
    if not context:
        return order, PositionCloseSummary(reason="missing_position_context")
    response = None
    if not order.exchange_order_id:
        # The strict historical resolver may recover a missing acknowledgement
        # ID from the original window; retries never shift that window forward.
        response = (order.raw_response or {}).get("exchange_response")
    evidence = _resolve(adapter, order.instrument, context, response, order.exchange_order_id or None)
    refreshed = _save_evidence(order.instrument, context, evidence, response)
    return refreshed, summarize_position(refreshed)


def has_partial_close(inst, side, correlation_id, opened_at):
    """A durable attempted partial leg prevents duplicate submission after cache loss."""
    opened = _date(opened_at)
    context = {"namespace": get_runtime_exchange_context().get("risk_namespace", ""),
               "side": str(side).lower(), "correlation_id": str(correlation_id or ""),
               "opened_at": opened.isoformat() if opened else None}
    if not context["namespace"] or not opened:
        return False
    key = _position_key(inst, context)
    return Order.objects.filter(instrument=inst, reduce_only=True,
                raw_response__close_accounting__namespace=context["namespace"],
                raw_response__close_accounting__position_key=key,
                raw_response__close_accounting__is_partial=True).exists()


def get_pending_closes(limit=10, min_retry_seconds=60):
    """Bounded fair scan; only the current account namespace is eligible."""
    limit = max(0, min(int(limit), 50))
    if not limit:
        return []
    namespace = get_runtime_exchange_context().get("risk_namespace")
    if not namespace:
        return []
    rows = Order.objects.filter(reduce_only=True, raw_response__close_accounting__version=VERSION,
                raw_response__close_accounting__namespace=namespace,
                raw_response__close_accounting__status="pending",
                updated_at__lte=timezone.now() - timedelta(seconds=max(0, min(int(min_retry_seconds), 3600)))
                ).select_related("instrument").order_by("updated_at", "pk")
    has_accounting_key = any(field.name == "accounting_key" for field in OperationReport._meta.fields)
    if has_accounting_key:
        # Extract text, not JSONB, for a portable text-key subquery. Exclude
        # confirmed reports in SQL before LIMIT so history cannot starve work.
        confirmed_keys = OperationReport.objects.filter(accounting_status="confirmed").exclude(
            accounting_key="").values("accounting_key")
        rows = rows.annotate(accounting_position_key=KeyTextTransform(
            "position_key", KeyTransform("close_accounting", "raw_response"))).exclude(
                accounting_position_key__in=confirmed_keys)
    return list(rows[:limit])
