from __future__ import annotations

import os

from django.core.management.base import BaseCommand

from core.models import ExchangeCredential


class Command(BaseCommand):
    help = "Sync exchange credentials from environment variables into DB table."

    def handle(self, *args, **options):
        rows = [
            {
                "service": ExchangeCredential.Service.KUCOIN,
                "api_key": os.getenv("KUCOIN_API_KEY", ""),
                "api_secret": os.getenv("KUCOIN_API_SECRET", ""),
                "api_passphrase": os.getenv("KUCOIN_API_PASSPHRASE", ""),
                "sandbox": os.getenv("KUCOIN_SANDBOX", "true").lower() == "true",
                "margin_mode": os.getenv("KUCOIN_MARGIN_MODE", "cross"),
                "leverage": int(os.getenv("KUCOIN_LEVERAGE", "3")),
                "label": os.getenv("KUCOIN_API_LABEL", ""),
            },
            {
                "service": ExchangeCredential.Service.BINANCE,
                "api_key": os.getenv("BINANCE_API_KEY", ""),
                "api_secret": os.getenv("BINANCE_API_SECRET", ""),
                "api_passphrase": "",
                "sandbox": os.getenv("BINANCE_TESTNET", "true").lower() == "true",
                "margin_mode": os.getenv("BINANCE_MARGIN_MODE", "cross"),
                "leverage": int(os.getenv("BINANCE_LEVERAGE", "3")),
                "label": os.getenv("BINANCE_API_LABEL", ""),
            },
            {
                "service": ExchangeCredential.Service.BINGX,
                "api_key": os.getenv("BINGX_API_KEY", ""),
                "api_secret": os.getenv("BINGX_API_SECRET", ""),
                "api_passphrase": os.getenv("BINGX_API_PASSPHRASE", ""),
                "sandbox": os.getenv("BINGX_SANDBOX", "false").lower() == "true",
                "margin_mode": os.getenv("BINGX_MARGIN_MODE", "cross"),
                "leverage": int(os.getenv("BINGX_LEVERAGE", "3")),
                "label": os.getenv("BINGX_API_LABEL", ""),
            },
        ]

        default_active = os.getenv("EXCHANGE", "kucoin").strip().lower()
        created, updated = 0, 0

        for row in rows:
            service = row["service"]
            defaults = {
                "api_key": row["api_key"],
                "api_secret": row["api_secret"],
                "api_passphrase": row["api_passphrase"],
                "sandbox": row["sandbox"],
                "margin_mode": row["margin_mode"],
                "leverage": row["leverage"],
                "label": row["label"],
                "active": service == default_active,
            }
            obj, was_created = ExchangeCredential.objects.update_or_create(
                service=service,
                defaults=defaults,
            )
            created += int(was_created)
            updated += int(not was_created)
            self.stdout.write(
                f"{service}: active={obj.active} key_set={'yes' if bool(obj.api_key) else 'no'}"
            )

        self.stdout.write(
            self.style.SUCCESS(f"Done. created={created} updated={updated} active={default_active}")
        )
