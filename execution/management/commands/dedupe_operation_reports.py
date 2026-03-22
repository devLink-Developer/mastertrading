from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Any

from django.core.management.base import BaseCommand
from django.db import transaction
from django.utils import timezone as dj_tz

from execution.models import OperationReport
from execution.tasks import _operation_reason_priority


def _identity_key(op: OperationReport) -> tuple[Any, ...] | None:
    correlation_id = str(getattr(op, "correlation_id", "") or "").strip()
    signal_id = str(getattr(op, "signal_id", "") or "").strip()
    token = correlation_id or (f"signal:{signal_id}" if signal_id else "")
    if not token:
        return None
    opened_at = getattr(op, "opened_at", None)
    opened_key = opened_at.isoformat() if opened_at else ""
    return (
        getattr(op, "instrument_id", None),
        str(getattr(op, "side", "") or ""),
        str(getattr(op, "mode", "") or ""),
        opened_key,
        token,
    )


def build_operation_report_dedupe_plan(*, days: int = 30, symbol: str = "", mode: str = "") -> dict[str, Any]:
    cutoff = dj_tz.now() - timedelta(days=max(1, int(days or 1)))
    qs = OperationReport.objects.filter(closed_at__gte=cutoff).select_related("instrument").order_by("closed_at", "id")
    if symbol:
        qs = qs.filter(instrument__symbol__iexact=symbol)
    if mode:
        qs = qs.filter(mode=mode)

    grouped: dict[tuple[Any, ...], list[OperationReport]] = defaultdict(list)
    for op in qs:
        key = _identity_key(op)
        if key is None:
            continue
        grouped[key].append(op)

    duplicate_groups: list[dict[str, Any]] = []
    delete_ids: list[int] = []
    for group_ops in grouped.values():
        if len(group_ops) <= 1:
            continue
        canonical = max(
            group_ops,
            key=lambda op: (
                _operation_reason_priority(getattr(op, "reason", ""), getattr(op, "close_sub_reason", "")),
                getattr(op, "closed_at", None) or dj_tz.make_aware(datetime.min),
                getattr(op, "id", 0),
            ),
        )
        losers = [op for op in group_ops if op.id != canonical.id]
        delete_ids.extend(op.id for op in losers)
        duplicate_groups.append(
            {
                "symbol": canonical.instrument.symbol,
                "side": canonical.side,
                "mode": canonical.mode,
                "correlation_id": canonical.correlation_id,
                "signal_id": canonical.signal_id,
                "opened_at": canonical.opened_at.isoformat() if canonical.opened_at else None,
                "canonical_id": canonical.id,
                "canonical_reason": canonical.reason,
                "canonical_close_sub_reason": canonical.close_sub_reason,
                "canonical_closed_at": canonical.closed_at.isoformat() if canonical.closed_at else None,
                "duplicate_ids": [op.id for op in losers],
                "reports": [
                    {
                        "id": op.id,
                        "reason": op.reason,
                        "close_sub_reason": op.close_sub_reason,
                        "closed_at": op.closed_at.isoformat() if op.closed_at else None,
                    }
                    for op in group_ops
                ],
            }
        )

    duplicate_groups.sort(key=lambda row: (row["symbol"], row["opened_at"] or "", row["canonical_id"]))
    return {
        "window_days": int(days),
        "symbol": symbol.upper() if symbol else "",
        "mode": mode,
        "duplicate_group_count": len(duplicate_groups),
        "delete_count": len(delete_ids),
        "delete_ids": sorted(delete_ids),
        "groups": duplicate_groups,
    }


class Command(BaseCommand):
    help = "Find and optionally delete duplicate OperationReport rows for the same position lifecycle."

    def add_arguments(self, parser):
        parser.add_argument("--days", type=int, default=30)
        parser.add_argument("--symbol", type=str, default="")
        parser.add_argument("--mode", type=str, default="")
        parser.add_argument("--apply", action="store_true")
        parser.add_argument("--json", type=str, default="")

    def handle(self, *args, **opts):
        plan = build_operation_report_dedupe_plan(
            days=int(opts.get("days") or 30),
            symbol=str(opts.get("symbol") or "").strip(),
            mode=str(opts.get("mode") or "").strip(),
        )

        self.stdout.write(
            self.style.NOTICE(
                f"Duplicate OperationReport groups={plan['duplicate_group_count']} rows_to_delete={plan['delete_count']}"
            )
        )
        for row in plan["groups"][:20]:
            self.stdout.write(
                f"  {row['symbol']} {row['side']} corr={row['correlation_id'] or row['signal_id']} canonical={row['canonical_id']} delete={row['duplicate_ids']}"
            )

        if opts.get("json"):
            path = Path(str(opts["json"]))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(plan, ensure_ascii=True, indent=2), encoding="utf-8")
            self.stdout.write(self.style.SUCCESS(f"Saved -> {path}"))

        if not opts.get("apply"):
            self.stdout.write(self.style.WARNING("Dry-run only. Re-run with --apply to delete duplicates."))
            return

        delete_ids = list(plan["delete_ids"])
        if not delete_ids:
            self.stdout.write(self.style.SUCCESS("No duplicate rows to delete."))
            return

        with transaction.atomic():
            deleted, _ = OperationReport.objects.filter(id__in=delete_ids).delete()
        self.stdout.write(self.style.SUCCESS(f"Deleted {deleted} duplicate OperationReport rows."))