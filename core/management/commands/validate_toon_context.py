from __future__ import annotations

from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from core.toon_validator import validate_toon_file


class Command(BaseCommand):
    help = "Validate TOON context files (structure markers + anti-narrative guardrails)."

    def add_arguments(self, parser):
        parser.add_argument(
            "--glob",
            type=str,
            default="docs/*.toon.md",
            help="Glob relative to BASE_DIR for TOON files.",
        )
        parser.add_argument(
            "--strict",
            action="store_true",
            help="Return non-zero when any file fails validation.",
        )

    def handle(self, **options):
        rel_glob = str(options.get("glob") or "docs/*.toon.md").strip()
        strict = bool(options.get("strict", False))

        base = Path(settings.BASE_DIR).resolve()
        paths = sorted(base.glob(rel_glob))
        if not paths:
            raise CommandError(f"no files matched glob: {rel_glob}")

        failures = 0
        for path in paths:
            result = validate_toon_file(path)
            rel = path.relative_to(base).as_posix() if path.is_absolute() else str(path)
            if result.valid:
                self.stdout.write(self.style.SUCCESS(f"[OK] {rel}"))
            else:
                failures += 1
                self.stdout.write(self.style.ERROR(f"[FAIL] {rel}"))
                for err in result.errors:
                    self.stdout.write(f"  - {err}")

        self.stdout.write(f"validated={len(paths)} failed={failures}")
        if strict and failures > 0:
            raise CommandError(f"toon validation failed: {failures} file(s)")
