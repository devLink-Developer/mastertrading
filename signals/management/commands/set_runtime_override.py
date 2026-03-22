from __future__ import annotations

import json

from django.core.management.base import BaseCommand, CommandError

from signals.models import StrategyConfig
from signals.runtime_overrides import (
    RUNTIME_OVERRIDES_VERSION,
    invalidate_runtime_overrides_cache,
)


class Command(BaseCommand):
    help = "Create, update, show, or delete a DB-backed runtime override."

    def add_arguments(self, parser):
        parser.add_argument("name", nargs="?", default="")
        parser.add_argument("--value", type=str, default="")
        parser.add_argument("--show", action="store_true")
        parser.add_argument("--delete", action="store_true")

    def handle(self, *args, **opts):
        name = str(opts.get("name") or "").strip()
        show = bool(opts.get("show"))
        delete = bool(opts.get("delete"))
        raw_value = str(opts.get("value") or "")

        if show:
            qs = StrategyConfig.objects.filter(version=RUNTIME_OVERRIDES_VERSION).order_by("name")
            if name:
                qs = qs.filter(name=name)
            for row in qs:
                self.stdout.write(
                    json.dumps(
                        {
                            "name": row.name,
                            "enabled": bool(row.enabled),
                            "value": (row.params_json or {}).get("value"),
                        },
                        ensure_ascii=True,
                    )
                )
            return

        if not name:
            raise CommandError("name is required unless --show is used without arguments")

        if delete:
            deleted, _ = StrategyConfig.objects.filter(
                version=RUNTIME_OVERRIDES_VERSION,
                name=name,
            ).delete()
            invalidate_runtime_overrides_cache()
            self.stdout.write(self.style.SUCCESS(f"Deleted {deleted} override rows for {name}."))
            return

        if raw_value == "":
            raise CommandError("--value is required when creating or updating an override")

        try:
            parsed = json.loads(raw_value)
        except Exception:
            parsed = raw_value

        row, created = StrategyConfig.objects.update_or_create(
            version=RUNTIME_OVERRIDES_VERSION,
            name=name,
            defaults={"enabled": True, "params_json": {"value": parsed}},
        )
        invalidate_runtime_overrides_cache()
        action = "Created" if created else "Updated"
        self.stdout.write(self.style.SUCCESS(f"{action} runtime override {row.name}."))