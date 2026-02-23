from __future__ import annotations

from django.db import models

from core.crypto import decrypt_secret, encrypt_secret


class EncryptedCredentialField(models.TextField):
    """
    Transparently encrypt/decrypt credential values at rest.

    Reads return plaintext; DB storage contains encrypted tokens prefixed with `enc::`.
    Legacy plaintext rows are still readable and get encrypted on next save.
    """

    description = "Encrypted credential text field"

    def from_db_value(self, value, expression, connection):  # type: ignore[override]
        return decrypt_secret(value)

    def to_python(self, value):  # type: ignore[override]
        if value is None:
            return ""
        if isinstance(value, str):
            return decrypt_secret(value)
        return value

    def get_prep_value(self, value):  # type: ignore[override]
        if value is None:
            return ""
        if isinstance(value, str):
            return encrypt_secret(value)
        return encrypt_secret(str(value))
