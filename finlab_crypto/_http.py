"""Shared HTTP session with retries for the public data sources."""
from functools import cache

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

TIMEOUT_SECONDS = 30
RETRIES = 5
BACKOFF_SECONDS = 1
RETRY_STATUS = (429, 500, 502, 503, 504)


@cache
def _session():
    retry = Retry(total=RETRIES, backoff_factor=BACKOFF_SECONDS,
                  status_forcelist=RETRY_STATUS, respect_retry_after_header=True)
    session = requests.Session()
    session.mount('https://', HTTPAdapter(max_retries=retry))
    return session


def get_json(url, params=None):
    """GET ``url`` and decode JSON, raising ``requests.HTTPError`` on failure."""
    response = _session().get(url, params=params, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()
