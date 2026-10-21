import hashlib
import hmac

import pytest

from src.domain.exceptions import WebhookValidationError
from src.infrastructure.github.webhook_validator import WebhookValidator


def test_webhook_validator_valid_signature() -> None:
    secret = "super_secret_webhook_key"
    validator = WebhookValidator(secret=secret)

    payload_bytes = b'{"action": "closed", "number": 42}'
    signature = "sha256=" + hmac.new(secret.encode(), payload_bytes, hashlib.sha256).hexdigest()

    assert validator.verify_signature(payload_bytes, signature) is True
    parsed = validator.validate_and_parse(payload_bytes, signature)
    assert parsed["action"] == "closed"
    assert parsed["number"] == 42


def test_webhook_validator_invalid_signature() -> None:
    secret = "super_secret_webhook_key"
    validator = WebhookValidator(secret=secret)

    payload_bytes = b'{"action": "closed"}'
    invalid_signature = "sha256=invalidhex0000000000000000000000000000000000000000000000000000"

    assert validator.verify_signature(payload_bytes, invalid_signature) is False
    with pytest.raises(WebhookValidationError):
        validator.validate_and_parse(payload_bytes, invalid_signature)


def test_webhook_validator_missing_signature_prefix() -> None:
    secret = "secret"
    validator = WebhookValidator(secret=secret)

    assert validator.verify_signature(b"{}", None) is False
    assert validator.verify_signature(b"{}", "nosha256prefix") is False


def test_webhook_validator_no_secret_configured() -> None:
    validator = WebhookValidator(secret="")
    assert validator.verify_signature(b"{}", None) is True


def test_webhook_validator_invalid_json_payloads() -> None:
    secret = "test_secret"
    validator = WebhookValidator(secret=secret)

    # 1. Payload que no es dict (es una lista)
    list_payload = b"[1, 2, 3]"
    sig_list = "sha256=" + hmac.new(secret.encode(), list_payload, hashlib.sha256).hexdigest()
    with pytest.raises(WebhookValidationError) as exc1:
        validator.validate_and_parse(list_payload, sig_list)
    assert "El payload JSON debe ser un objeto" in str(exc1.value)

    # 2. JSON malformado
    bad_json = b"not a json"
    sig_bad = "sha256=" + hmac.new(secret.encode(), bad_json, hashlib.sha256).hexdigest()
    with pytest.raises(WebhookValidationError) as exc2:
        validator.validate_and_parse(bad_json, sig_bad)
    assert "Payload JSON malformado" in str(exc2.value)

    # 3. UnicodeDecodeError
    invalid_utf8 = b"\x80abc"
    sig_utf8 = "sha256=" + hmac.new(secret.encode(), invalid_utf8, hashlib.sha256).hexdigest()
    with pytest.raises(WebhookValidationError) as exc3:
        validator.validate_and_parse(invalid_utf8, sig_utf8)
    assert "Error de decodificación UTF-8" in str(exc3.value)
