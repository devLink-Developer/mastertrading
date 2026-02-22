from __future__ import annotations

from django.core.management.base import BaseCommand, CommandError

from core.models import ExchangeCredential


class Command(BaseCommand):
    help = "Set the active exchange service in DB without restarting services."

    def add_arguments(self, parser):
        parser.add_argument(
            "--service",
            type=str,
            required=True,
            help="Exchange service to activate: kucoin | binance | bingx",
        )

    def handle(self, *args, **options):
        service = (options["service"] or "").strip().lower()
        valid = {c[0] for c in ExchangeCredential.Service.choices}
        if service not in valid:
            raise CommandError(f"Invalid service '{service}'. Valid: {sorted(valid)}")

        row = ExchangeCredential.objects.filter(service=service).first()
        if not row:
            raise CommandError(f"Credential row for '{service}' not found.")

        ExchangeCredential.objects.update(active=False)
        row.active = True
        row.save(update_fields=["active", "updated_at"])
        self.stdout.write(
            self.style.SUCCESS(
                f"Active exchange set to {service}. key_set={'yes' if bool(row.api_key) else 'no'}"
            )
        )
