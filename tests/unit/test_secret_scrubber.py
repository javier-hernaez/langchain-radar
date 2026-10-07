from src.infrastructure.security.secret_scrubber import SecretScrubber


def test_secret_scrubber_detects_openai_and_github_tokens() -> None:
    scrubber = SecretScrubber()
    dirty_text = (
        'OPENAI_API_KEY = "sk-proj-1234567890abcdef1234567890abcdef12345678"\n'
        'GITHUB_TOKEN = "ghp_1234567890abcdef1234567890abcdef12"\n'
        "print('safe string')\n"
    )

    res = scrubber.scrub(dirty_text)
    assert res.has_secrets is True
    assert "openai_api_key" in res.secrets_found
    assert "github_pat" in res.secrets_found
    assert "sk-proj-" not in res.sanitized_text
    assert "ghp_" not in res.sanitized_text
    assert SecretScrubber.REDACTION_TOKEN in res.sanitized_text
    assert "print('safe string')" in res.sanitized_text


def test_secret_scrubber_detects_aws_and_pem_keys() -> None:
    scrubber = SecretScrubber()
    dirty_text = (
        "AWS_KEY = 'AKIA1234567890ABCDEF'\n"
        "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA0...\n-----END RSA PRIVATE KEY-----\n"
    )

    res = scrubber.scrub(dirty_text)
    assert res.has_secrets is True
    assert "aws_access_key" in res.secrets_found
    assert "pem_private_key" in res.secrets_found
    assert "AKIA" not in res.sanitized_text
    assert "-----BEGIN" not in res.sanitized_text


def test_secret_scrubber_clean_text() -> None:
    scrubber = SecretScrubber()
    clean_text = "def hello_world(): return 'clean code without keys'"

    res = scrubber.scrub(clean_text)
    assert res.has_secrets is False
    assert res.secrets_found == ()
    assert res.sanitized_text == clean_text
