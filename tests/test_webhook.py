"""Deterministic tests for utils/validation.py and services/webhook.py.

Network calls are mocked throughout (via unittest.mock.patch on
requests.post) — these tests must never depend on real network access.
"""

import threading
from datetime import datetime, timezone
from unittest.mock import patch

import pytest
import requests

from services.webhook import send_webhook, send_webhook_async
from utils.validation import is_valid_webhook_url, validate_webhook_url


# --- URL validation -----------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://example.com/webhook",
        "https://example.com/webhook?token=abc",
        "https://localhost:9000/hook",
        "http://127.0.0.1:5000/hook",
    ],
)
def test_valid_urls_pass(url):
    assert validate_webhook_url(url) is None
    assert is_valid_webhook_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "",
        "   ",
        "not-a-url",
        "javascript:alert(1)",
        "ftp://example.com/file",
        "http://",
        "example.com/webhook",  # missing scheme
    ],
)
def test_invalid_urls_are_rejected_with_a_message(url):
    error = validate_webhook_url(url)
    assert error is not None
    assert isinstance(error, str) and error
    assert not is_valid_webhook_url(url)


def test_validate_webhook_url_handles_none():
    assert validate_webhook_url(None) is not None


# --- send_webhook --------------------------------------------------------


def _fixed_timestamp():
    return datetime(2026, 8, 27, 15, 30, 45, tzinfo=timezone.utc)


def test_send_webhook_rejects_invalid_url_without_a_network_call():
    with patch("services.webhook.requests.post") as mock_post:
        result = send_webhook("not-a-url", "thumbs_up", _fixed_timestamp())

    mock_post.assert_not_called()
    assert result.success is False
    assert result.status_code is None
    assert "http" in result.error.lower() or "url" in result.error.lower()


def test_send_webhook_success_sends_expected_payload():
    fake_response = type("FakeResponse", (), {"status_code": 200})()
    with patch("services.webhook.requests.post", return_value=fake_response) as mock_post:
        result = send_webhook("https://example.com/hook", "thumbs_up", _fixed_timestamp())

    assert result.success is True
    assert result.status_code == 200
    assert result.error is None

    _, kwargs = mock_post.call_args
    assert kwargs["json"] == {
        "event": "gesture_detected",
        "gesture": "thumbs_up",
        "timestamp": "2026-08-27T15:30:45Z",
    }
    assert kwargs["timeout"] > 0


def test_send_webhook_handles_non_2xx_response():
    fake_response = type("FakeResponse", (), {"status_code": 500})()
    with patch("services.webhook.requests.post", return_value=fake_response):
        result = send_webhook("https://example.com/hook", "fist", _fixed_timestamp())

    assert result.success is False
    assert result.status_code == 500
    assert "500" in result.error


def test_send_webhook_handles_connection_error_without_raising():
    with patch(
        "services.webhook.requests.post",
        side_effect=requests.exceptions.ConnectionError("refused"),
    ):
        result = send_webhook("https://example.com/hook", "fist", _fixed_timestamp())

    assert result.success is False
    assert result.status_code is None
    assert result.error is not None


def test_send_webhook_handles_timeout_without_raising():
    with patch(
        "services.webhook.requests.post",
        side_effect=requests.exceptions.Timeout("too slow"),
    ):
        result = send_webhook("https://example.com/hook", "fist", _fixed_timestamp())

    assert result.success is False
    assert result.error is not None


def test_send_webhook_never_logs_full_url(caplog):
    with patch(
        "services.webhook.requests.post",
        side_effect=requests.exceptions.ConnectionError("refused"),
    ):
        with caplog.at_level("WARNING"):
            send_webhook("https://example.com/hook?token=super-secret", "fist", _fixed_timestamp())

    assert "super-secret" not in caplog.text


# --- send_webhook_async ---------------------------------------------------


def test_send_webhook_async_invokes_callback_without_blocking():
    done = threading.Event()
    captured = {}

    def on_result(result):
        captured["result"] = result
        done.set()

    fake_response = type("FakeResponse", (), {"status_code": 200})()
    with patch("services.webhook.requests.post", return_value=fake_response):
        send_webhook_async(
            "https://example.com/hook", "peace", _fixed_timestamp(), on_result=on_result
        )
        # send_webhook_async must return immediately (fire-and-forget).
        assert done.wait(timeout=2), "callback was not invoked in time"

    assert captured["result"].success is True


def test_send_webhook_async_swallows_callback_exceptions():
    done = threading.Event()

    def bad_callback(result):
        done.set()
        raise RuntimeError("boom")

    fake_response = type("FakeResponse", (), {"status_code": 200})()
    with patch("services.webhook.requests.post", return_value=fake_response):
        send_webhook_async(
            "https://example.com/hook", "peace", _fixed_timestamp(), on_result=bad_callback
        )
        assert done.wait(timeout=2)
    # No exception should propagate out of the background thread to the test.
