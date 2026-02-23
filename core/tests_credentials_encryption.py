from django.db import connection
from django.test import TestCase

from core.crypto import decrypt_secret, encrypt_secret, is_encrypted_secret
from core.models import ExchangeCredential


class CredentialEncryptionTest(TestCase):
    def test_crypto_helpers_roundtrip(self):
        plain = "my-secret-key"
        encrypted = encrypt_secret(plain)
        self.assertTrue(is_encrypted_secret(encrypted))
        self.assertNotEqual(encrypted, plain)
        self.assertEqual(decrypt_secret(encrypted), plain)

    def test_exchange_credentials_are_encrypted_at_rest(self):
        cred = ExchangeCredential.objects.create(
            service=ExchangeCredential.Service.BINGX,
            api_key="k_plain",
            api_secret="s_plain",
            api_passphrase="p_plain",
        )
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT api_key, api_secret, api_passphrase "
                "FROM core_exchangecredential WHERE id = %s",
                [cred.id],
            )
            raw_key, raw_secret, raw_passphrase = cursor.fetchone()

        self.assertTrue(str(raw_key).startswith("enc::"))
        self.assertTrue(str(raw_secret).startswith("enc::"))
        self.assertTrue(str(raw_passphrase).startswith("enc::"))
        self.assertNotEqual(raw_key, "k_plain")
        self.assertNotEqual(raw_secret, "s_plain")
        self.assertNotEqual(raw_passphrase, "p_plain")

        cred.refresh_from_db()
        self.assertEqual(cred.api_key, "k_plain")
        self.assertEqual(cred.api_secret, "s_plain")
        self.assertEqual(cred.api_passphrase, "p_plain")

    def test_legacy_plaintext_row_is_readable_and_reencrypted_on_save(self):
        cred = ExchangeCredential.objects.create(
            service=ExchangeCredential.Service.KUCOIN,
            api_key="legacy_k",
            api_secret="legacy_s",
            api_passphrase="legacy_p",
        )

        # Simulate pre-encryption legacy row persisted in plaintext.
        with connection.cursor() as cursor:
            cursor.execute(
                "UPDATE core_exchangecredential "
                "SET api_key = %s, api_secret = %s, api_passphrase = %s "
                "WHERE id = %s",
                ["legacy_k", "legacy_s", "legacy_p", cred.id],
            )

        legacy = ExchangeCredential.objects.get(id=cred.id)
        self.assertEqual(legacy.api_key, "legacy_k")
        self.assertEqual(legacy.api_secret, "legacy_s")
        self.assertEqual(legacy.api_passphrase, "legacy_p")

        legacy.save(update_fields=["api_key", "api_secret", "api_passphrase"])

        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT api_key, api_secret, api_passphrase "
                "FROM core_exchangecredential WHERE id = %s",
                [cred.id],
            )
            raw_key, raw_secret, raw_passphrase = cursor.fetchone()

        self.assertTrue(str(raw_key).startswith("enc::"))
        self.assertTrue(str(raw_secret).startswith("enc::"))
        self.assertTrue(str(raw_passphrase).startswith("enc::"))
