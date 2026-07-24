# yargitay_client.py
"""
API Client for Yargıtay (Court of Cassation) decision search and document retrieval.
Includes token-bucket rate-limiting, non-blocking HTML-to-Markdown parsing,
TTL caching, and secure HTTPS requests.
"""

import asyncio
import httpx
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from typing import Dict, Any, Optional, NamedTuple
import logging
import html
import time
import os

from yargitay_models import (
    YargitayDetailedSearchRequest,
    YargitayApiSearchResponse,
    YargitayApiDecisionEntry,
    YargitayDocumentMarkdown,
    CompactYargitaySearchResult
)

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class YargitayRateLimited(Exception):
    """Raised when the local rate-limit bucket would block longer than allowed."""
    def __init__(self, retry_after: float) -> None:
        self.retry_after = retry_after
        super().__init__(f"Local rate-limit bucket would block for {retry_after:.1f}s")


class _TokenBucket:
    """Asyncio token bucket with explicit back-pressure.

    Default limit: 1 request per 3.5s (capacity 1).
    Override via env vars:
      YARGITAY_RATE_CAPACITY (default 1)
      YARGITAY_RATE_REFILL_S (default 3.5)
      YARGITAY_RATE_MAX_WAIT_S (default 8.0)
    """

    def __init__(self, capacity: int = 1, refill_per_s: float = 1.0 / 3.5) -> None:
        self.capacity = float(capacity)
        self.refill_per_s = float(refill_per_s)
        self._tokens = float(capacity)
        self._last = time.monotonic()
        self._not_before = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self, max_wait: Optional[float] = 8.0) -> None:
        """Acquire one token. If max_wait is exceeded, raise YargitayRateLimited."""
        deadline = (time.monotonic() + max_wait) if max_wait is not None else None
        while True:
            async with self._lock:
                now = time.monotonic()
                if now < self._not_before:
                    wait_s = self._not_before - now
                else:
                    self._tokens = min(
                        self.capacity,
                        self._tokens + (now - self._last) * self.refill_per_s,
                    )
                    self._last = now
                    if self._tokens >= 1.0:
                        self._tokens -= 1.0
                        return
                    wait_s = (1.0 - self._tokens) / self.refill_per_s

            if deadline is not None:
                remaining = deadline - time.monotonic()
                if wait_s > remaining:
                    raise YargitayRateLimited(retry_after=wait_s)
            await asyncio.sleep(wait_s)

    def penalize_until(self, monotonic_deadline: float) -> None:
        """Pause the bucket until monotonic_deadline."""
        self._not_before = max(self._not_before, monotonic_deadline)
        self._tokens = 0.0
        self._last = time.monotonic()


class CacheEntry(NamedTuple):
    content: Any
    expires_at: float


class YargitayCache:
    """Simple in-memory TTL cache for Yargitay queries and documents."""

    def __init__(self, default_ttl: int = 3600):
        self._cache: Dict[str, CacheEntry] = {}
        self._default_ttl = default_ttl

    def get(self, key: str) -> Optional[Any]:
        if key not in self._cache:
            return None
        entry = self._cache[key]
        if time.time() > entry.expires_at:
            del self._cache[key]
            return None
        return entry.content

    def put(self, key: str, content: Any, ttl: Optional[int] = None) -> None:
        expires_at = time.time() + (ttl if ttl is not None else self._default_ttl)
        self._cache[key] = CacheEntry(content=content, expires_at=expires_at)


class YargitayOfficialApiClient:
    """
    API Client for Yargıtay's official decision search system (karararama.yargitay.gov.tr).
    Includes rate-limiting, non-blocking HTML-to-Markdown parsing, and TTL caching.
    """
    BASE_URL = "https://karararama.yargitay.gov.tr"
    DETAILED_SEARCH_ENDPOINT = "/aramadetaylist"
    DOCUMENT_ENDPOINT = "/getDokuman"

    def __init__(self, request_timeout: float = 60.0, cache_ttl: int = 3600, enable_cache: bool = True):
        self.http_client = httpx.AsyncClient(
            base_url=self.BASE_URL,
            headers={
                "Content-Type": "application/json; charset=UTF-8",
                "Accept": "application/json, text/plain, */*",
                "X-Requested-With": "XMLHttpRequest",
                "Referer": f"{self.BASE_URL}/",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            },
            timeout=request_timeout,
            verify=True
        )

        capacity = int(os.getenv("YARGITAY_RATE_CAPACITY", "1"))
        refill_s = float(os.getenv("YARGITAY_RATE_REFILL_S", "3.5"))
        self._bucket = _TokenBucket(capacity=capacity, refill_per_s=1.0 / refill_s)
        
        self.enable_cache = enable_cache
        self.cache = YargitayCache(default_ttl=cache_ttl)

    async def search_detailed_decisions(
        self, 
        search_params: YargitayDetailedSearchRequest
    ) -> YargitayApiSearchResponse:
        """
        Performs a detailed search for decisions in Yargıtay using search_params.
        """
        # Convert "ALL" to empty string for API compatibility
        if search_params.birimYrgKurulDaire == "ALL":
            search_params.birimYrgKurulDaire = ""

        cache_key = f"search:{search_params.model_dump_json()}"
        if self.enable_cache:
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                logger.info("YargitayOfficialApiClient: Returning cached search result.")
                return cached_result

        # Acquire token from rate-limiter bucket
        await self._bucket.acquire()

        request_payload = {"data": search_params.model_dump(exclude_none=True, by_alias=True)}
        logger.info(f"YargitayOfficialApiClient: Detailed search payload: {request_payload}")

        try:
            response = await self.http_client.post(self.DETAILED_SEARCH_ENDPOINT, json=request_payload)

            if response.status_code == 429:
                self._bucket.penalize_until(time.monotonic() + 30.0)
                raise YargitayRateLimited(retry_after=30.0)

            response.raise_for_status()
            response_json_data = response.json()

            if not isinstance(response_json_data, dict) or response_json_data.get("data") is None:
                # Check if the API returned an error in its metadata field before
                # silently treating this as "0 results" — the Yargıtay API often
                # returns {"data": null, "metadata": {"FMTY": "ERROR", ...}}.
                error_detail = None
                if isinstance(response_json_data, dict):
                    meta = response_json_data.get("metadata")
                    if isinstance(meta, dict) and meta.get("FMTY") == "ERROR":
                        error_detail = meta.get("FMU") or meta.get("FMTE") or "Unknown API error"
                        logger.warning(f"YargitayOfficialApiClient: API returned error metadata: {error_detail}")

                if error_detail:
                    # Surface the real error instead of hiding it behind empty results
                    api_response = YargitayApiSearchResponse(**response_json_data)
                    return api_response

                logger.warning("YargitayOfficialApiClient: API returned empty or non-dict data field.")
                response_json_data = {"data": {"data": [], "recordsTotal": 0, "recordsFiltered": 0}}

            api_response = YargitayApiSearchResponse(**response_json_data)

            if api_response.data and api_response.data.data:
                for decision_item in api_response.data.data:
                    decision_item.document_url = f"{self.BASE_URL}{self.DOCUMENT_ENDPOINT}?id={decision_item.id}"

            if self.enable_cache and api_response.data and api_response.data.data:
                self.cache.put(cache_key, api_response)

            return api_response

        except httpx.RequestError as e:
            logger.error(f"YargitayOfficialApiClient: HTTP request error during detailed search: {e}")
            raise
        except Exception as e:
            logger.error(f"YargitayOfficialApiClient: Error processing detailed search response: {e}")
            raise

    def _convert_html_to_markdown_sync(self, html_content: str) -> Optional[str]:
        """
        Synchronous helper for HTML to Markdown conversion using BeautifulSoup and markdownify.
        Runs inside asyncio.to_thread to prevent blocking the event loop.
        """
        if not html_content:
            return None

        # Pre-process HTML entities & escape codes
        processed_html = html.unescape(html_content)
        processed_html = processed_html.replace('\\"', '"').replace('\\r\\n', '\n').replace('\\n', '\n').replace('\\t', '\t')

        soup = BeautifulSoup(processed_html, "html.parser")

        # Strip script and style tags
        for element in soup(["script", "style"]):
            element.decompose()

        markdown_output = md(str(soup), heading_style="ATX", strip=['script', 'style'])
        return markdown_output.strip() if markdown_output else None

    async def get_decision_document_as_markdown(self, id: str) -> YargitayDocumentMarkdown:
        """
        Retrieves a specific Yargıtay decision by ID and converts its content to Markdown.
        """
        if not id or not id.strip():
            raise ValueError("Document ID must be a non-empty string.")

        cache_key = f"doc:{id}"
        if self.enable_cache:
            cached_doc = self.cache.get(cache_key)
            if cached_doc is not None:
                logger.info(f"YargitayOfficialApiClient: Returning cached document for ID: {id}")
                return cached_doc

        await self._bucket.acquire()

        document_api_url = f"{self.DOCUMENT_ENDPOINT}?id={id}"
        source_url = f"{self.BASE_URL}{document_api_url}"
        logger.info(f"YargitayOfficialApiClient: Fetching document ID: {id}")

        try:
            response = await self.http_client.get(document_api_url)

            if response.status_code == 429:
                self._bucket.penalize_until(time.monotonic() + 30.0)
                raise YargitayRateLimited(retry_after=30.0)

            response.raise_for_status()
            response_json = response.json()
            html_content_from_api = response_json.get("data")

            if not isinstance(html_content_from_api, str):
                logger.error(f"YargitayOfficialApiClient: 'data' field missing or invalid for ID: {id}")
                raise ValueError("Expected HTML content not found in API response's 'data' field.")

            # Non-blocking HTML -> Markdown conversion
            markdown_content = await asyncio.to_thread(self._convert_html_to_markdown_sync, html_content_from_api)

            doc_result = YargitayDocumentMarkdown(
                id=id,
                markdown_content=markdown_content,
                source_url=source_url
            )

            if self.enable_cache and markdown_content:
                self.cache.put(cache_key, doc_result)

            return doc_result

        except httpx.RequestError as e:
            logger.error(f"YargitayOfficialApiClient: HTTP error fetching document (ID: {id}): {e}")
            raise
        except Exception as e:
            logger.error(f"YargitayOfficialApiClient: Error fetching/processing document (ID: {id}): {e}")
            raise

    async def close_client_session(self):
        """Closes the HTTPX client session."""
        await self.http_client.aclose()
        logger.info("YargitayOfficialApiClient: HTTP client session closed.")