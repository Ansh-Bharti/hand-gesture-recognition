"""URL validation for the user-configurable webhook endpoint."""

from __future__ import annotations

from typing import Optional
from urllib.parse import urlparse

_ALLOWED_SCHEMES = {"http", "https"}


def validate_webhook_url(url: str) -> Optional[str]:
    """Return a human-readable validation error, or None if the URL is usable.

    Deliberately conservative and cheap: checks that the URL is well-formed
    and uses http/https, not that the endpoint is actually reachable (that
    is only knowable by attempting the request, handled separately in
    services/webhook.py). This does not attempt SSRF-style protections
    (blocking private/loopback addresses) — this is a single-user local
    tool where pointing the webhook at, say, a local test receiver on
    localhost is a legitimate and expected use case, not an attack surface;
    see DECISIONS.md and the README Limitations section.
    """
    if url is None or not url.strip():
        return "Webhook URL cannot be empty."

    candidate = url.strip()

    try:
        parsed = urlparse(candidate)
    except ValueError:
        return "Webhook URL is malformed."

    if parsed.scheme not in _ALLOWED_SCHEMES:
        shown = parsed.scheme or "none"
        return f"Webhook URL must start with http:// or https:// (got scheme: {shown})."

    if not parsed.netloc or " " in parsed.netloc or parsed.hostname is None:
        return "Webhook URL must include a valid host."

    return None


def is_valid_webhook_url(url: str) -> bool:
    return validate_webhook_url(url) is None
