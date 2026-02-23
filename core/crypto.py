from __future__ import annotations

import base64
import hashlib
import logging
import os

from cryptography.fernet import Fernet, InvalidToken

logger = logging.getLogger(__name__)

_ENC_PREFIX = "enc::"
_DEFAULT_SECRET_FALLBACK = "changeme-in-prod"


def _derive_fernet_key(secret: str) -> str:
    digest = hashlib.sha256(secret.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest).decode("utf-8")


def _normalize_fernet_key(raw: str) -> str:
    candidate = (raw or "").strip()
    if not candidate:
        return _derive_fernet_key(_DEFAULT_SECRET_FALLBACK)
    # Allow explicit 44-char Fernet key, otherwise derive one from the provided secret.
    if len(candidate) == 44:
        return candidate
    return _derive_fernet_key(candidate)


def _encryption_key() -> str:
    # Prefer a dedicated key; fallback to SECRET_KEY for zero-config deployments.
    raw = os.getenv("CREDENTIALS_ENCRYPTION_KEY", "").strip()
    if raw:
        return _normalize_fernet_key(raw)
    return _normalize_fernet_key(os.getenv("SECRET_KEY", _DEFAULT_SECRET_FALLBACK))


def is_encrypted_secret(value: str | None) -> bool:
    return bool(value) and str(value).startswith(_ENC_PREFIX)


def encrypt_secret(value: str | None) -> str:
    if value is None:
        return ""
    plain = str(value)
    if plain == "":
        return ""
    if is_encrypted_secret(plain):
        return plain
    token = Fernet(_encryption_key()).encrypt(plain.encode("utf-8")).decode("utf-8")
    return f"{_ENC_PREFIX}{token}"


def decrypt_secret(value: str | None) -> str:
    if value is None:
        return ""
    raw = str(value)
    if raw == "":
        return ""
    if not is_encrypted_secret(raw):
        # Backward compatibility for legacy plaintext rows.
        return raw
    token = raw[len(_ENC_PREFIX):]
    try:
        return Fernet(_encryption_key()).decrypt(token.encode("utf-8")).decode("utf-8")
    except InvalidToken:
        logger.warning("Credential decryption failed: invalid token/key")
        return ""
    except Exception as exc:
        logger.warning("Credential decryption failed: %s", exc)
        return ""
