from __future__ import annotations

import os

from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from django.db import OperationalError, ProgrammingError

from core.models import ApiProviderConfig, ExchangeCredential


class Command(BaseCommand):
    help = "Sync exchange credentials from environment variables into DB table."

    def _default_alias(self, service: str, sandbox: bool) -> str:
        env = "demo" if sandbox else "live"
        return f"{service}-{env}"

    def _resolve_owner_id(self, username: str) -> int | None:
        username = (username or "").strip()
        if not username:
            return None
        user = get_user_model().objects.filter(username=username).only("id").first()
        if not user:
            self.stderr.write(
                self.style.WARNING(
                    f"Owner username '{username}' not found. Credential will be stored without owner."
                )
            )
            return None
        return int(user.id)

    def _resolve_ai_provider_config_id(self, alias: str) -> int | None:
        alias = (alias or "").strip()
        if not alias:
            return None
        try:
            row = ApiProviderConfig.objects.filter(name_alias=alias).only("id").first()
        except (OperationalError, ProgrammingError):
            return None
        if not row:
            self.stderr.write(
                self.style.WARNING(
                    f"AI provider config alias '{alias}' not found. Credential will keep ai_provider_config empty."
                )
            )
            return None
        return int(row.id)

    def handle(self, *args, **options):
        kucoin_sandbox = os.getenv("KUCOIN_SANDBOX", "true").lower() == "true"
        binance_sandbox = os.getenv("BINANCE_TESTNET", "true").lower() == "true"
        bingx_sandbox = os.getenv("BINGX_SANDBOX", "false").lower() == "true"

        kucoin_alias = (os.getenv("KUCOIN_ACCOUNT_ALIAS", "") or "").strip().lower() or self._default_alias("kucoin", kucoin_sandbox)
        binance_alias = (os.getenv("BINANCE_ACCOUNT_ALIAS", "") or "").strip().lower() or self._default_alias("binance", binance_sandbox)
        bingx_alias = (os.getenv("BINGX_ACCOUNT_ALIAS", "") or "").strip().lower() or self._default_alias("bingx", bingx_sandbox)

        rows = [
            {
                "service": ExchangeCredential.Service.KUCOIN,
                "name_alias": kucoin_alias,
                "owner_username": os.getenv("KUCOIN_ACCOUNT_OWNER", ""),
                "api_key": os.getenv("KUCOIN_API_KEY", ""),
                "api_secret": os.getenv("KUCOIN_API_SECRET", ""),
                "api_passphrase": os.getenv("KUCOIN_API_PASSPHRASE", ""),
                "sandbox": kucoin_sandbox,
                "margin_mode": os.getenv("KUCOIN_MARGIN_MODE", "cross"),
                "leverage": int(os.getenv("KUCOIN_LEVERAGE", "3")),
                "label": os.getenv("KUCOIN_API_LABEL", ""),
                "ai_enabled": os.getenv("KUCOIN_AI_ENABLED", "false").lower() == "true",
                "ai_provider_config_alias": os.getenv("KUCOIN_AI_PROVIDER_CONFIG_ALIAS", ""),
            },
            {
                "service": ExchangeCredential.Service.BINANCE,
                "name_alias": binance_alias,
                "owner_username": os.getenv("BINANCE_ACCOUNT_OWNER", ""),
                "api_key": os.getenv("BINANCE_API_KEY", ""),
                "api_secret": os.getenv("BINANCE_API_SECRET", ""),
                "api_passphrase": "",
                "sandbox": binance_sandbox,
                "margin_mode": os.getenv("BINANCE_MARGIN_MODE", "cross"),
                "leverage": int(os.getenv("BINANCE_LEVERAGE", "3")),
                "label": os.getenv("BINANCE_API_LABEL", ""),
                "ai_enabled": os.getenv("BINANCE_AI_ENABLED", "false").lower() == "true",
                "ai_provider_config_alias": os.getenv("BINANCE_AI_PROVIDER_CONFIG_ALIAS", ""),
            },
            {
                "service": ExchangeCredential.Service.BINGX,
                "name_alias": bingx_alias,
                "owner_username": os.getenv("BINGX_ACCOUNT_OWNER", ""),
                "api_key": os.getenv("BINGX_API_KEY", ""),
                "api_secret": os.getenv("BINGX_API_SECRET", ""),
                "api_passphrase": os.getenv("BINGX_API_PASSPHRASE", ""),
                "sandbox": bingx_sandbox,
                "margin_mode": os.getenv("BINGX_MARGIN_MODE", "cross"),
                "leverage": int(os.getenv("BINGX_LEVERAGE", "3")),
                "label": os.getenv("BINGX_API_LABEL", ""),
                "ai_enabled": os.getenv("BINGX_AI_ENABLED", "false").lower() == "true",
                "ai_provider_config_alias": os.getenv("BINGX_AI_PROVIDER_CONFIG_ALIAS", ""),
            },
        ]

        default_active_service = os.getenv("EXCHANGE", "kucoin").strip().lower()
        default_active_alias = (os.getenv("EXCHANGE_ACCOUNT_ALIAS", "") or "").strip().lower()
        created, updated = 0, 0

        for row in rows:
            service = row["service"]
            alias = row["name_alias"]
            owner_id = self._resolve_owner_id(row.get("owner_username", ""))
            ai_provider_config_id = self._resolve_ai_provider_config_id(
                row.get("ai_provider_config_alias", "")
            )
            is_default_alias = not default_active_alias or alias == default_active_alias
            defaults = {
                "service": service,
                "owner_id": owner_id,
                "name_alias": alias,
                "api_key": row["api_key"],
                "api_secret": row["api_secret"],
                "api_passphrase": row["api_passphrase"],
                "sandbox": row["sandbox"],
                "margin_mode": row["margin_mode"],
                "leverage": row["leverage"],
                "label": row["label"],
                "ai_enabled": row.get("ai_enabled", False),
                "ai_provider_config_id": ai_provider_config_id,
                "active": service == default_active_service and is_default_alias,
            }
            obj, was_created = ExchangeCredential.objects.update_or_create(
                name_alias=alias,
                defaults=defaults,
            )
            if obj.active:
                siblings = ExchangeCredential.objects.exclude(pk=obj.pk).filter(
                    service=obj.service,
                    sandbox=obj.sandbox,
                    active=True,
                )
                if obj.owner_id:
                    siblings = siblings.filter(owner_id=obj.owner_id)
                else:
                    siblings = siblings.filter(owner__isnull=True)
                siblings.update(active=False)
            created += int(was_created)
            updated += int(not was_created)
            self.stdout.write(
                f"{service}/{obj.name_alias}: active={obj.active} "
                f"owner={getattr(obj.owner, 'username', 'unowned')} "
                f"key_set={'yes' if bool(obj.api_key) else 'no'} "
                f"ai={'on' if bool(getattr(obj, 'ai_enabled', False)) else 'off'}"
            )

        self.stdout.write(
            self.style.SUCCESS(
                "Done. "
                f"created={created} updated={updated} "
                f"active_service={default_active_service} active_alias={default_active_alias or '*'}"
            )
        )
