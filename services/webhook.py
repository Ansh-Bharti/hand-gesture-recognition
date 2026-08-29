"""Webhook dispatch for confirmed gesture events.

Sends a JSON POST to a user-configured URL when a gesture event is
confirmed. Validation, timeouts, and error handling all happen here so the
rest of the app can treat "notify the webhook" as a fire-and-forget call
that can never itself crash the gesture-detection pipeline.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Optional
from urllib.parse import urlparse

import requests

from utils.validation import validate_webhook_url

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = float(os.getenv("WEBHOOK_TIMEOUT_SECONDS", "5"))


@dataclass(frozen=True)
class WebhookResult:
    """Outcome of one webhook delivery attempt."""

    success: bool
    status_code: Optional[int]
    error: Optional[str]
    sent_at: datetime


def _safe_host_for_logging(url: str) -> str:
    """Return only scheme+host for logging, never the full URL.

    A webhook URL may embed a token or signature in its path or query
    string (many services use a URL-as-secret pattern); logging the full
    URL would leak that. The host alone is enough to diagnose delivery
    issues without exposing anything sensitive.
    """
    try:
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}" if parsed.netloc else "<unparseable-url>"
    except ValueError:
        return "<unparseable-url>"


def send_webhook(
    url: str,
    gesture: str,
    timestamp: datetime,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> WebhookResult:
    """Send one gesture-event webhook synchronously. Never raises.

    Every failure mode (invalid URL, connection failure, timeout, non-2xx
    response) is caught and reported as a WebhookResult rather than an
    exception, so a broken or slow webhook endpoint can never take down
    the caller.
    """
    validation_error = validate_webhook_url(url)
    if validation_error:
        logger.warning("Webhook not sent: %s", validation_error)
        return WebhookResult(
            success=False,
            status_code=None,
            error=validation_error,
            sent_at=datetime.now(timezone.utc),
        )

    payload = {
        "event": "gesture_detected",
        "gesture": gesture,
        "timestamp": timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    host = _safe_host_for_logging(url)

    try:
        response = requests.post(url, json=payload, timeout=timeout)
    except requests.exceptions.RequestException as exc:
        logger.warning("Webhook request to %s failed: %s", host, type(exc).__name__)
        return WebhookResult(
            success=False,
            status_code=None,
            error=f"Could not reach webhook endpoint ({type(exc).__name__}).",
            sent_at=datetime.now(timezone.utc),
        )

    if 200 <= response.status_code < 300:
        logger.info("Webhook delivered to %s (status %s)", host, response.status_code)
        return WebhookResult(
            success=True,
            status_code=response.status_code,
            error=None,
            sent_at=datetime.now(timezone.utc),
        )

    logger.warning("Webhook to %s returned non-2xx status %s", host, response.status_code)
    return WebhookResult(
        success=False,
        status_code=response.status_code,
        error=f"Webhook endpoint responded with HTTP {response.status_code}.",
        sent_at=datetime.now(timezone.utc),
    )


def send_webhook_async(
    url: str,
    gesture: str,
    timestamp: datetime,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    on_result: Optional[Callable[[WebhookResult], None]] = None,
) -> None:
    """Dispatch send_webhook() on a background daemon thread.

    The gesture-detection video callback must stay fast; blocking it on a
    network round trip (a slow or unresponsive webhook endpoint, up to the
    full timeout) would stall the live video feed. Gesture events are
    already rare by the time they reach here (one per confirmed gesture,
    not one per frame — see gesture/state.py), so a plain thread per event
    is simple and has no meaningful resource cost. A full async event loop
    or task queue was considered and rejected as unwarranted complexity
    for this event rate; see DECISIONS.md.
    """

    def _worker() -> None:
        result = send_webhook(url, gesture, timestamp, timeout=timeout)
        if on_result is not None:
            try:
                on_result(result)
            except Exception:
                logger.exception("Webhook on_result callback raised an exception")

    threading.Thread(target=_worker, daemon=True, name="webhook-dispatch").start()
