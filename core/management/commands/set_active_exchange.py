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
        parser.add_argument(
            "--name-alias",
            type=str,
            default="",
            help="Account alias to activate (recommended when multiple rows share the same service).",
        )
        parser.add_argument(
            "--owner",
            type=str,
            default="",
            help="Optional owner username filter.",
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

        siblings = ExchangeCredential.objects.exclude(pk=row.pk).filter(
            service=row.service,
            sandbox=row.sandbox,
            active=True,
        )
        if row.owner_id:
            siblings = siblings.filter(owner_id=row.owner_id)
        else:
            siblings = siblings.filter(owner__isnull=True)
        siblings.update(active=False)
        row.active = True
        row.save(update_fields=["active", "updated_at"])
        self.stdout.write(
            self.style.SUCCESS(
                "Active exchange set to "
                f"{service} alias={row.name_alias} owner={getattr(row.owner, 'username', 'unowned')} "
                f"sandbox={'on' if row.sandbox else 'off'} key_set={'yes' if bool(row.api_key) else 'no'}"
            )
        )
