"""Thin, polite client for the public MLB Stats API (no key required)."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterable, Sequence

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import API_BASE

log = logging.getLogger(__name__)

_RETRY = Retry(
    total=5,
    backoff_factor=0.5,
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=("GET",),
)


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = "mlb-predictor/0.1 (personal research project)"
    adapter = HTTPAdapter(max_retries=_RETRY, pool_connections=16, pool_maxsize=16)
    s.mount("https://", adapter)
    return s


_SESSION = make_session()


def get_json(path: str, params: dict[str, Any] | None = None, *, sleep: float = 0.0) -> dict:
    """GET {API_BASE}/{path} and return parsed JSON."""
    url = f"{API_BASE}/{path.lstrip('/')}"
    resp = _SESSION.get(url, params=params, timeout=60)
    resp.raise_for_status()
    if sleep:
        time.sleep(sleep)
    return resp.json()


def map_parallel(fn: Callable[[Any], Any], items: Sequence[Any], workers: int = 4) -> list:
    """Run `fn` over `items` with a small thread pool, preserving order.

    Kept deliberately small: this is a public, free API and hammering it is both
    rude and a good way to get rate limited.
    """
    if not items:
        return []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, items))


def chunked(items: Iterable[Any], size: int) -> Iterable[list]:
    buf: list = []
    for it in items:
        buf.append(it)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf
