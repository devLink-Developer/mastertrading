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
        parser.add_argument(
            "--name-alias",
            type=str,
            default="",
            help="Account alias to update (recommended when multiple rows share the same service).",
        )
        parser.add_argument(
            "--owner",
            type=str,
            default="",
            help="Optional owner username filter.",
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
        alias = (options.get("name_alias") or "").strip().lower()
        owner = (options.get("owner") or "").strip()
        valid = {c[0] for c in ExchangeCredential.Service.choices}
        if service not in valid:
            raise CommandError(f"Invalid service '{service}'. Valid: {sorted(valid)}")

        qs = ExchangeCredential.objects.filter(service=service)
        if alias:
            qs = qs.filter(name_alias__iexact=alias)
        if owner:
            qs = qs.filter(owner__username=owner)
        if qs.count() > 1 and not alias:
            raise CommandError(
                f"Multiple credentials found for '{service}'. Use --name-alias. "
                f"Candidates: {list(qs.values_list('name_alias', flat=True))}"
            )

        row = qs.order_by("-updated_at").first()
        if not row:
            raise CommandError(
                f"Credential row not found for service='{service}' alias='{alias or '*'}' owner='{owner or '*'}'."
            )

        sandbox_enabled = bool(options.get("on"))
        row.sandbox = sandbox_enabled
        row.save(update_fields=["sandbox", "updated_at"])

        mode = "sandbox ON" if sandbox_enabled else "sandbox OFF"
        self.stdout.write(
            self.style.SUCCESS(
                f"{service}/{row.name_alias}: {mode}. "
                f"owner={getattr(row.owner, 'username', 'unowned')} active={'yes' if row.active else 'no'}"
            )
        )
