import os
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase

from adapters.credentials import (
    get_active_service,
    get_default_adapter_signature,
    get_exchange_credentials,
)
from core.models import ExchangeCredential


class CredentialSelectionTests(TestCase):
    def setUp(self):
        user_model = get_user_model()
        self.alice = user_model.objects.create_user(
            username="alice",
            email="alice@example.com",
            password="x",
        )
        self.bob = user_model.objects.create_user(
            username="bob",
            email="bob@example.com",
            password="x",
        )
        self.alice_demo = ExchangeCredential.objects.create(
            owner=self.alice,
            name_alias="alice-demo",
            service=ExchangeCredential.Service.KUCOIN,
            api_key="k_alice_demo",
            api_secret="s_alice_demo",
            api_passphrase="p_alice_demo",
            sandbox=True,
            active=True,
        )
        self.alice_live = ExchangeCredential.objects.create(
            owner=self.alice,
            name_alias="alice-live",
            service=ExchangeCredential.Service.KUCOIN,
            api_key="k_alice_live",
            api_secret="s_alice_live",
            api_passphrase="p_alice_live",
            sandbox=False,
            active=True,
        )
        self.bob_live = ExchangeCredential.objects.create(
            owner=self.bob,
            name_alias="bob-live",
            service=ExchangeCredential.Service.BINGX,
            api_key="k_bob_live",
            api_secret="s_bob_live",
            api_passphrase="",
            sandbox=False,
            active=True,
        )

    def test_alias_selector_can_switch_service(self):
        with patch.dict(
            os.environ,
            {
                "EXCHANGE": "kucoin",
                "EXCHANGE_ACCOUNT_ALIAS": "bob-live",
            },
            clear=False,
        ):
            service = get_active_service()
            cfg = get_exchange_credentials(service)

        self.assertEqual(service, "bingx")
        self.assertEqual(cfg["name_alias"], "bob-live")
        self.assertEqual(cfg["owner_username"], "bob")
        self.assertEqual(cfg["credential_id"], self.bob_live.id)

    def test_owner_and_sandbox_select_expected_account(self):
        with patch.dict(
            os.environ,
            {
                "EXCHANGE": "kucoin",
                "EXCHANGE_ACCOUNT_OWNER": "alice",
                "EXCHANGE_ACCOUNT_SANDBOX": "true",
            },
            clear=False,
        ):
            service = get_active_service()
            cfg = get_exchange_credentials(service)

        self.assertEqual(service, "kucoin")
        self.assertEqual(cfg["name_alias"], "alice-demo")
        self.assertEqual(cfg["credential_id"], self.alice_demo.id)
        self.assertTrue(cfg["sandbox"])

    def test_signature_includes_account_context(self):
        with patch.dict(
            os.environ,
            {
                "EXCHANGE": "kucoin",
                "EXCHANGE_ACCOUNT_ALIAS": "alice-live",
            },
            clear=False,
        ):
            signature = get_default_adapter_signature()

        self.assertIn("|alice-live|", signature)
        self.assertIn("|alice|", signature)
        self.assertIn(f"|{self.alice_live.id}", signature)
