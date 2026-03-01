from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand
from django.utils import timezone as dj_tz

from core.ai_feedback import _stream_path, feedback_event_to_compact_payload
from core.models import AiFeedbackEvent


class Command(BaseCommand):
    help = "Rebuild compact AI feedback JSONL stream from PostgreSQL events."

    def add_arguments(self, parser):
        parser.add_argument(
            "--hours",
            type=int,
            default=168,
            help="Include events from last N hours (default: 168).",
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=5000,
            help="Max events to export (default: 5000).",
        )
        parser.add_argument(
            "--write",
            type=str,
            default="",
            help="Output path (defaults to AI_FEEDBACK_JSONL_PATH).",
        )

    def handle(self, *args, **options):
        hours = max(1, int(options["hours"] or 168))
        limit = max(1, int(options["limit"] or 5000))
        write_path = str(options.get("write") or "").strip()
        out = _stream_path(path_override=write_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        cutoff = dj_tz.now() - timedelta(hours=hours)
        qs = (
            AiFeedbackEvent.objects.filter(created_at__gte=cutoff)
            .order_by("-created_at")
            .only(
                "created_at",
                "event_type",
                "level",
                "account_alias",
                "account_service",
                "symbol",
                "strategy",
                "allow",
                "risk_mult",
                "reason",
                "latency_ms",
                "fingerprint",
                "payload_json",
            )[:limit]
        )
        rows = list(reversed(list(qs)))
        with out.open("w", encoding="utf-8", newline="\n") as fh:
            meta = {
                "_meta": {
                    "generated_at": dj_tz.now().isoformat(),
                    "hours": hours,
                    "limit": limit,
                    "count": len(rows),
                    "path": out.relative_to(Path(settings.BASE_DIR).resolve()).as_posix()
                    if Path(settings.BASE_DIR).resolve() in out.parents
                    else str(out),
                }
            }
            fh.write(json.dumps(meta, ensure_ascii=True, separators=(",", ":")))
            fh.write("\n")
            for row in rows:
                payload = feedback_event_to_compact_payload(row)
                fh.write(json.dumps(payload, ensure_ascii=True, separators=(",", ":")))
                fh.write("\n")

        self.stdout.write(
            self.style.SUCCESS(
                f"feedback stream rebuilt: {len(rows)} events -> {out}"
            )
        )
