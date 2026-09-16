"""A failed connection names what failed.

The vendor SDKs report every transport failure as "Connection error."; the reason is the exception
theirs was raised from. On 2026-09-16 every key probe from a phone read "anthropic: unreachable --
anthropic: Connection error." and nothing a person could act on."""
import httpx
import pytest

from agent.router.llm_provider.base import ProviderUnavailable, connection_detail


def test_an_error_with_nothing_underneath_is_shown_as_it_is():
    assert connection_detail(RuntimeError("Connection error.")) == "Connection error."
    assert connection_detail(RuntimeError()) == "RuntimeError"


def test_the_failure_underneath_is_named():
    try:
        try:
            raise OSError("[SSL: CERTIFICATE_VERIFY_FAILED] unable to get local issuer certificate")
        except OSError as inner:
            raise RuntimeError("Connection error.") from inner
    except RuntimeError as outer:
        shown = connection_detail(outer)
    assert shown.startswith("Connection error. (OSError: [SSL: CERTIFICATE_VERIFY_FAILED]")


def test_the_deepest_cause_is_the_one_shown_and_a_repeat_is_not():
    try:
        try:
            try:
                raise OSError("[Errno 7] No address associated with hostname")
            except OSError as dns:
                raise httpx.ConnectError(str(dns)) from dns
        except httpx.ConnectError as connect:
            raise RuntimeError("Connection error.") from connect
    except RuntimeError as outer:
        assert connection_detail(outer) == "Connection error. (OSError: [Errno 7] No address associated with hostname)"
    same = RuntimeError("timed out")
    same.__cause__ = OSError("timed out")
    assert connection_detail(same) == "timed out"


def test_the_anthropic_probe_says_what_failed():
    anthropic = pytest.importorskip("anthropic")
    from agent.router.llm_provider.anthropic_provider import _translate

    request = httpx.Request("GET", "https://api.anthropic.com/v1/models")
    try:
        try:
            raise httpx.ConnectError("[Errno -3] Temporary failure in name resolution", request=request)
        except httpx.ConnectError as cause:
            raise anthropic.APIConnectionError(request=request) from cause
    except anthropic.APIConnectionError as exc:
        translated = _translate(exc)
    assert isinstance(translated, ProviderUnavailable)
    assert str(translated) == ("anthropic: Connection error. "
                               "(ConnectError: [Errno -3] Temporary failure in name resolution)")
