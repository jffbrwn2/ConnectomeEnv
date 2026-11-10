import io
import json
from email.message import Message
from urllib.error import HTTPError, URLError
from unittest.mock import patch

import pytest
from proofreading import provider_http as http


def rejected(code='rate_limit_exceeded', kind='rate_limit_error', retry=None):
    headers = Message()
    if retry is not None:
        headers['Retry-After'] = retry
    return HTTPError('https://example.invalid', 429, 'Rejected', headers,
                     io.BytesIO(json.dumps({'error': {'code': code, 'type': kind}}).encode()))


def test_rate_limit_retries_same_request_after_minimum():
    request = object()
    with patch.object(http, 'urlopen', side_effect=[rejected(retry='12'), io.BytesIO(b'{"ok":true}')]) as send, patch.object(http.time, 'sleep') as sleep, patch.object(http.random, 'uniform', return_value=.5):
        assert http.request_json(request) == {'ok': True}
        assert send.call_count == 2
        assert all(call.args[0] is request for call in send.call_args_list)
        sleep.assert_called_once_with(12.5)


@pytest.mark.parametrize('error', [rejected('insufficient_quota'), rejected('project_spend_limit_exceeded'), rejected('unrecognized'), rejected(None, None), TimeoutError(), URLError('connection closed')])
def test_no_retry_for_quota_unknown_or_ambiguous_delivery(error):
    with patch.object(http, 'urlopen', side_effect=error) as send, patch.object(http.time, 'sleep') as sleep:
        with pytest.raises(type(error)):
            http.request_json(object())
        assert send.call_count == 1
        sleep.assert_not_called()


def test_server_wait_exceeding_budget_is_not_shortened():
    with patch.object(http, 'urlopen', side_effect=rejected(retry='600')) as send, patch.object(http.time, 'sleep') as sleep:
        with pytest.raises(HTTPError):
            http.request_json(object())
        assert send.call_count == 1
        sleep.assert_not_called()


def test_exhaustion_saves_identifiers_and_limits_attempts():
    with patch.object(http, 'urlopen', side_effect=[rejected() for _ in range(6)]) as send, patch.object(http.time, 'sleep'):
        with pytest.raises(HTTPError) as caught:
            http.request_json(object())
        assert send.call_count == 6
        assert caught.value.provider_detail['code'] == 'rate_limit_exceeded'
        assert caught.value.provider_detail['attempt'] == 6


def test_total_window_stops_retries():
    with patch.object(http, 'urlopen', side_effect=[rejected(), rejected()]) as send, patch.object(http.time, 'sleep'), patch.object(http.time, 'monotonic', side_effect=[0, 0, 299]):
        with pytest.raises(HTTPError):
            http.request_json(object())
        assert send.call_count == 2


@pytest.mark.parametrize('value', ['nan', 'inf', 'bad', None])
def test_invalid_retry_after(value):
    assert http.retry_after(value) is None

@pytest.mark.parametrize('error, expected', [
 ({'code':'credit_balance_exhausted'}, True),
 ({'code':'project_spend_limit_exceeded'}, True),
 ({'type':'invalid_request_error','message':'Your credit balance is too low to access the API.'}, True),
 ({'code':'rate_limit_exceeded'}, False),
 ({'type':'invalid_request_error','message':'image dimensions invalid'}, False),
 ({'type':'invalid_request_error','message':None}, False),
])
def test_billing_classification(error, expected):
    assert http.billing_error(error) is expected


@pytest.mark.parametrize("status", [500, 502, 503, 504, 529])
def test_temporary_server_error_recovers_same_request(status):
    error = HTTPError('https://example.invalid', status, 'Server error', Message(), io.BytesIO(b'<html>unavailable</html>'))
    request = object()
    with patch.object(http, 'urlopen', side_effect=[error, io.BytesIO(b'{"ok":true}')]) as send, patch.object(http.time, 'sleep'), patch.object(http.random, 'uniform', return_value=0):
        assert http.request_json(request) == {'ok': True}
        assert send.call_count == 2
        assert all(call.args[0] is request for call in send.call_args_list)


def test_server_error_exhaustion_is_bounded():
    errors = [HTTPError('https://example.invalid', 500, 'Server error', Message(), io.BytesIO(b'{}')) for _ in range(6)]
    with patch.object(http, 'urlopen', side_effect=errors) as send, patch.object(http.time, 'sleep') as sleep:
        with pytest.raises(HTTPError) as caught:
            http.request_json(object())
        assert send.call_count == 6
        assert sleep.call_count == 5
        assert caught.value.provider_detail['retryable'] is True


@pytest.mark.parametrize("status", [400, 401, 403, 404, 500])
def test_billing_or_permanent_error_is_not_retried(status):
    body = {'error': {'code': 'insufficient_quota'}} if status == 500 else {}
    error = HTTPError('https://example.invalid', status, 'Rejected', Message(), io.BytesIO(json.dumps(body).encode()))
    with patch.object(http, 'urlopen', side_effect=error) as send, patch.object(http.time, 'sleep') as sleep:
        with pytest.raises(HTTPError):
            http.request_json(object())
        assert send.call_count == 1
        sleep.assert_not_called()


@pytest.mark.parametrize('message', [
    'Exceeded maximum number of images (50) allowed in the request.',
    'Too many images in request: 51, maximum allowed: 50.',
])
def test_image_limit_is_identified_without_retry(message, capsys):
    error = HTTPError('https://example.invalid', 400, 'Rejected', Message(),
                      io.BytesIO(json.dumps({'error': {'message': message,
                                 'type': 'invalid_request_error'}}).encode()))
    with patch.object(http, 'urlopen', side_effect=error) as send, patch.object(http.time, 'sleep') as sleep:
        with pytest.raises(HTTPError) as caught:
            http.request_json(object())
        assert caught.value.provider_detail['reason'] == 'image_count_limit'
        assert caught.value.provider_detail['max_images'] == 50
        assert send.call_count == 1
        sleep.assert_not_called()
    assert message not in capsys.readouterr().out


@pytest.mark.parametrize('code',['429','TooManyRequests'])
def test_azure_rate_limit_codes_retry(code):
    with patch.object(http,'urlopen',side_effect=[rejected(code=code),io.BytesIO(b'{"ok":true}')]) as send,patch.object(http.time,'sleep'):
        assert http.request_json(object())=={'ok':True}
        assert send.call_count==2
