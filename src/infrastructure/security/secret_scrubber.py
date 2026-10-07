import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Expresiones regulares basadas en especificaciones oficiales de Gitleaks
SECRET_PATTERNS: dict[str, re.Pattern[str]] = {
    "openai_api_key": re.compile(r"\b(sk-[a-zA-Z0-9]{20,48}|sk-proj-[a-zA-Z0-9_-]{20,80})\b"),
    "github_pat": re.compile(r"\b(ghp_[a-zA-Z0-9]{30,40}|github_pat_[a-zA-Z0-9_]{82})\b"),
    "aws_access_key": re.compile(r"\b(AKIA[0-9A-Z]{16})\b"),
    "jwt_token": re.compile(
        r"\b(eyJ[a-zA-Z0-9_-]{10,}\.eyJ[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,})\b"
    ),
    "pem_private_key": re.compile(
        r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----[\s\S]+?-----END (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"
    ),
    "slack_token": re.compile(r"\b(xox[baprs]-[0-9a-zA-Z]{10,48})\b"),
    "anthropic_api_key": re.compile(r"\b(sk-ant-[a-zA-Z0-9_-]{20,80})\b"),
}


@dataclass(frozen=True)
class ScrubResult:
    """Resultado del proceso de sanitización de secretos."""

    sanitized_text: str
    secrets_found: tuple[str, ...]
    has_secrets: bool


class SecretScrubber:
    """Interceptor de seguridad pre-ingesta para detectar y enmascarar credenciales sensibles."""

    REDACTION_TOKEN = "[REDACTED_SECRET]"

    def __init__(self, patterns: dict[str, re.Pattern[str]] | None = None) -> None:
        self._patterns = patterns or SECRET_PATTERNS

    def scrub(self, text: str) -> ScrubResult:
        """Sanitiza el texto reemplazando cualquier secreto detectado por [REDACTED_SECRET]."""
        if not text:
            return ScrubResult(sanitized_text="", secrets_found=(), has_secrets=False)

        detected_types: list[str] = []
        clean_text = text

        for secret_name, pattern in self._patterns.items():
            matches = pattern.findall(clean_text)
            if matches:
                detected_types.append(secret_name)
                logger.warning(
                    "Secreto detectado y sanitizado pre-ingesta: tipo=%s, coincidencias=%d",
                    secret_name,
                    len(matches),
                )
                clean_text = pattern.sub(self.REDACTION_TOKEN, clean_text)

        return ScrubResult(
            sanitized_text=clean_text,
            secrets_found=tuple(detected_types),
            has_secrets=len(detected_types) > 0,
        )
