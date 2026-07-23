"""
mcp_client/tool_cache.py

Feature 3: "the client provides a gracefully aged cache entry, keeping the
LLM responsive" when a live tool call times out or errors.

Not MCP's "Resources" primitive (the current server only exposes Tools) -
this caches tool-call results by (tool_name, arguments) so a slow/failing
call (e.g. get_teblig_content on a scraping timeout) can fall back to the
last known-good response instead of a hard error.
"""
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    value: Any
    stored_at: float


class AgedToolCache:
    def __init__(self, fresh_ttl_s: float, max_stale_age_s: float):
        self.fresh_ttl_s = fresh_ttl_s
        self.max_stale_age_s = max_stale_age_s
        self._store: Dict[str, CacheEntry] = {}

    @staticmethod
    def _key(tool_name: str, arguments: Dict[str, Any]) -> str:
        blob = json.dumps({"tool": tool_name, "args": arguments}, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def put(self, tool_name: str, arguments: Dict[str, Any], value: Any) -> None:
        key = self._key(tool_name, arguments)
        self._store[key] = CacheEntry(value=value, stored_at=time.time())

    def get_fresh(self, tool_name: str, arguments: Dict[str, Any]) -> Optional[Any]:
        """Return a value only if it's within the normal TTL (used to skip a live call entirely, if desired)."""
        entry = self._store.get(self._key(tool_name, arguments))
        if entry is None:
            return None
        if time.time() - entry.stored_at > self.fresh_ttl_s:
            return None
        return entry.value

    def get_stale_fallback(self, tool_name: str, arguments: Dict[str, Any]) -> Optional[tuple]:
        """
        Return (value, age_seconds) for use ONLY after a live call has already
        failed, regardless of normal TTL, as long as it's not older than
        max_stale_age_s. Returns None if nothing usable is cached.
        """
        entry = self._store.get(self._key(tool_name, arguments))
        if entry is None:
            return None
        age = time.time() - entry.stored_at
        if age > self.max_stale_age_s:
            return None
        return entry.value, age

    def size(self) -> int:
        return len(self._store)
