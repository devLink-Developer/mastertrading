from __future__ import annotations

from django.core.management.base import BaseCommand, CommandError

from core.models import ExchangeCredential


class Command(BaseCommand):
    help = "Enable/disable sandbox mode for a service in DB credentials."

    def add_arguments(self, parser):
        parser.add_argument(
            "--service",
            type=str,
            required=True,
            help="Exchange service to update: kucoin | binance | bingx",
        )
        mode_group = parser.add_mutually_exclusive_group(required=True)
        mode_group.add_argument(
            "--on",
            action="store_true",
            help="Enable sandbox/test mode for the service.",
        )
        mode_group.add_argument(
            "--off",
            action="store_true",
            help="Disable sandbox/test mode for the service.",
        )

    def handle(self, *args, **options):
        service = (options["service"] or "").strip().lower()
        valid = {c[0] for c in ExchangeCredential.Service.choices}
        if service not in valid:
            raise CommandError(f"Invalid service '{service}'. Valid: {sorted(valid)}")

        row = ExchangeCredential.objects.filter(service=service).first()
        if not row:
            raise CommandError(f"Credential row for '{service}' not found.")

        sandbox_enabled = bool(options.get("on"))
        row.sandbox = sandbox_enabled
        row.save(update_fields=["sandbox", "updated_at"])

        mode = "sandbox ON" if sandbox_enabled else "sandbox OFF"
        self.stdout.write(
            self.style.SUCCESS(
                f"{service}: {mode}. active={'yes' if row.active else 'no'}"
            )
        )

