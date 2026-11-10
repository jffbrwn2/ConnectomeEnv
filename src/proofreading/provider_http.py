"""Bounded retries for rate limits and temporary server errors."""
import json
import math
import random
import re
import time
from email.utils import parsedate_to_datetime
from urllib.error import HTTPError
from urllib.request import urlopen


def billing_error(error):
    code = error.get("code")
    if code in {"credit_balance_exhausted", "insufficient_quota",
                "organization_spend_limit_exceeded", "project_spend_limit_exceeded",
                "organization_usage_limit_exceeded"}:
        return True
    # Anthropic reports exhausted credits as invalid_request_error with no code.
    message = error.get("message", "")
    return (error.get("type") == "invalid_request_error"
            and isinstance(message, str) and "credit balance" in message.lower()
            and "too low" in message.lower())


def retry_after(value):
    try:
        delay = float(value)
    except (TypeError, ValueError):
        try:
            delay = parsedate_to_datetime(value).timestamp() - time.time()
        except (TypeError, ValueError, OverflowError):
            return None
    return max(0., delay) if math.isfinite(delay) else None


def request_json(request):
    """At most six attempts within a 300s retry scheduling window.

    Retry only explicit HTTP responses, never ambiguous disconnects/timeouts.
    The caller receives one successful response, so local tools are not replayed.
    In-flight requests retain their separate 1200s timeout. No payload logging."""
    started = time.monotonic()
    for attempt in range(6):
        try:
            with urlopen(request, timeout=1200) as response:
                return json.load(response)
        except HTTPError as exc:
            try:
                body = json.loads(exc.read(65536))
                error = body.get("error", {}) if isinstance(body, dict) else {}
                if not isinstance(error, dict):
                    error = {}
            except (ValueError, UnicodeError):
                error = {}
            # Keep only bounded identifiers; never emit provider message text,
            # authorization, endpoint query strings, or request content.
            def identifier(value):
                return value if isinstance(value, str) and re.fullmatch(r"[a-zA-Z0-9_.:-]{1,160}", value) else None
            code = identifier(error.get("code"))
            kind = identifier(error.get("type"))
            detail = {"http_status": exc.code, "code": code, "type": kind,
                      "request_id": identifier(exc.headers.get("x-request-id")),
                      "attempt": attempt + 1, "billing_action_required": billing_error(error)}
            # Classify known limits without logging arbitrary provider text.
            message = error.get("message")
            if isinstance(message, str):
                limit = re.fullmatch(
                    r"Exceeded maximum number of images \((\d{1,6})\) allowed in the request\.",
                    message.strip(),
                ) or re.fullmatch(
                    r"Too many images in request: \d{1,6}, maximum allowed: (\d{1,6})\.",
                    message.strip(),
                )
                if limit:
                    detail.update(reason="image_count_limit", max_images=int(limit.group(1)))
            exc.provider_detail = detail
            # Only recognized rate limits and temporary server statuses qualify.
            # Billing errors take precedence even under a server-error status.
            rate_limited = exc.code == 429 and (
                code in {"rate_limit_exceeded", "slow_down", "429", "TooManyRequests"}
                or (not error.get("code") and kind == "rate_limit_error"))
            transient = not detail["billing_action_required"] and (
                exc.code in {500, 502, 503, 504, 529} or rate_limited)
            detail["retryable"] = transient
            delay = retry_after(exc.headers.get("Retry-After"))
            if delay is None:
                delay = min(60., 5. * 2 ** attempt)
            delay += random.uniform(0., 1.)
            remaining = 300. - (time.monotonic() - started)
            if not transient or attempt == 5 or delay > remaining:
                print("PROVIDER_HTTP_STOP " + json.dumps(detail), flush=True)
                raise
            print("PROVIDER_HTTP_RETRY " + json.dumps({**detail, "delay_seconds": delay}), flush=True)
            exc.close()
            time.sleep(delay)
