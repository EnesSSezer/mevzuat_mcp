from __future__ import annotations
import urllib.parse  
import asyncio
import base64
import io
import json
import logging
from typing import Any

import httpx
from bs4 import BeautifulSoup

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - optional dependency
    PdfReader = None

# DOMAIN STANDARDIZATION: Prevent HTTP redirect loops (301/302) during lookup routines
HOME_URL = "https://www.mevzuat.gov.tr/"
SEARCH_URL = "https://www.mevzuat.gov.tr/Anasayfa/MevzuatDatatable"
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36"
)
MAX_DOCUMENT_TEXT_LENGTH = 6000
TRUNCATION_SUFFIX = "\n...[CONTENT TRUNCATED FOR LENGTH]..."

# Pagination: DataTables 'length' request size. Bumped from the old hardcoded
# 10 so a relevant document ranked 11-25 isn't invisible to the agent; capped
# so a single page can't blow the tool-response token budget.
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 30

# NOTE: "Baslik" and "Tumu" are confirmed against a real captured request.
# "Icerik" (content-only search) is inferred from the site's title/content/both
# terminology described by the user, but has NOT been confirmed against a real
# captured request/response pair. If it turns out to be wrong, this is the one
# value to correct.
_VALID_SEARCH_PLACES = {"Tumu", "Baslik", "Icerik"}

_client: httpx.AsyncClient | None = None
_client_loop: asyncio.AbstractEventLoop | None = None
_client_lock = asyncio.Lock()
logger = logging.getLogger(__name__)

_BLOCK_MARKERS = (
    "attention required",
    "cloudflare",
    "cf-ray",
    "verify you are human",
    "checking your browser",
    "access denied",
    "service unavailable",
)


def _build_headers(*, referer: str | None = None) -> dict[str, str]:
    headers = {
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
        "Connection": "keep-alive",
        "Content-Type": "application/json; charset=UTF-8",
        "Origin": "https://www.mevzuat.gov.tr",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
        "User-Agent": DEFAULT_USER_AGENT,
        "X-Requested-With": "XMLHttpRequest",
    }
    if referer:
        headers["Referer"] = referer
    return headers


async def _get_client() -> httpx.AsyncClient:
    """
    Thread-safe lazy initialization block. Prepares connection pool configurations
    and warms up session cookies EXACTLY ONCE during the app lifecycle.

    Loop-safety: an httpx.AsyncClient is bound to the event loop it was created
    on. If the caller is now running on a *different* loop than the one that
    built the cached client (e.g. this module got reused across separate
    asyncio.run() cycles), reusing it raises confusing "Event loop is closed"
    errors. We detect that and transparently rebuild instead. The CLI itself
    is architected to run entirely inside one loop, so this branch should
    normally never trigger — it's a safety net for other future callers.
    """
    global _client, _client_loop

    current_loop = asyncio.get_running_loop()

    if _client is not None and _client_loop is not current_loop:
        logger.warning("Detected event-loop change; discarding stale HTTP client instead of reusing it.")
        stale_client = _client
        _client = None
        _client_loop = None
        try:
            await stale_client.aclose()
        except Exception as exc:
            logger.debug(f"Ignoring error while closing stale client: {exc}")

    if _client is None:
        async with _client_lock:
            if _client is None:
                client_instance = httpx.AsyncClient(
                    timeout=httpx.Timeout(30.0),
                    follow_redirects=True,
                    trust_env=False,
                    headers={"User-Agent": DEFAULT_USER_AGENT},
                )
                # LAZY WARMUP: Safely hit the entrypoint once to grab critical cookies
                try:
                    await client_instance.get(HOME_URL, headers={"User-Agent": DEFAULT_USER_AGENT})
                except Exception as exc:
                    logger.warning(f"Initial session warm-up failed: {exc}")

                _client = client_instance
                _client_loop = current_loop
    return _client


def _looks_like_block_page(response: httpx.Response) -> bool:
    server_header = (response.headers.get("server") or "").lower()
    if "cloudflare" in server_header:
        return True
    text = response.text.lower()
    return any(marker in text for marker in _BLOCK_MARKERS)


def _normalize_record(item: Any) -> dict[str, Any]:
    if isinstance(item, dict):
        raw_title = (
            item.get("mevAdi") or item.get("MevzuatAdi") or 
            item.get("title") or item.get("Baslik") or item.get("Adi") or ""
        )
        
        if raw_title and "<" in raw_title:
            try:
                title = BeautifulSoup(raw_title, "html.parser").text
            except Exception:
                title = raw_title
        else:
            title = raw_title

        law_number = item.get("mevzuatNo") or item.get("MevzuatNo") or item.get("law_number") or item.get("No") or ""
        gazette_date = item.get("resmiGazeteTarihi") or item.get("YayimTarihi") or item.get("ResmiGazeteTarihi") or item.get("gazette_date") or ""
        document_url = item.get("url") or item.get("DocumentUrl") or item.get("document_url") or item.get("DetayUrl") or ""
        document_id = item.get("mevzuatNo") or item.get("MevzuatId") or item.get("Id") or ""
        
        # ALIGNED HOSTNAME COMPLETION: Prepend standardized www prefix
        if document_url and not str(document_url).startswith(("http://", "https://")):
            if str(document_url).startswith("/"):
                document_url = f"https://www.mevzuat.gov.tr{document_url}"
            else:
                document_url = f"https://www.mevzuat.gov.tr/{document_url}"

        law_type = item.get("mevzuatTurEnumString") or ""

        return {
            "source_header": f"[SOURCE: {title or document_id or 'Unknown'} | Type: {law_type} | Law No: {law_number or 'N/A'} | Gazette Date: {gazette_date or 'N/A'}]",
            "title": title,
            "law_type": law_type,
            "law_number": law_number,
            "gazette_date": gazette_date,
            "document_url": document_url,
            "document_id": document_id,
        }
    return {"source_header": "[SOURCE: Unknown]", "title": str(item)}


def _parse_search_payload(data: Any) -> list[Any]:
    if isinstance(data, dict):
        for key in ("data", "Data", "result", "Result", "rows", "Rows"):
            value = data.get(key)
            if isinstance(value, list):
                return value
        return []
    return data if isinstance(data, list) else []


def _extract_total_count(data: Any) -> int:
    """
    DataTables responses carry recordsFiltered (matches for the current
    search) and recordsTotal (rows in the whole table, unfiltered).
    recordsFiltered is the one that actually tells the agent "how many results
    exist for my query" — previously this was parsed out of the response and
    then thrown away entirely.
    """
    if isinstance(data, dict):
        for key in ("recordsFiltered", "RecordsFiltered", "recordsTotal", "RecordsTotal"):
            value = data.get(key)
            if isinstance(value, int):
                return value
    return 0


def _format_document_text(source_label: str, text: str) -> str:
    clean_text = text[:MAX_DOCUMENT_TEXT_LENGTH]
    if len(text) > MAX_DOCUMENT_TEXT_LENGTH:
        clean_text += TRUNCATION_SUFFIX
    return f"[SOURCE: {source_label}]\n---\n{clean_text}"


async def search_mevzuat(
    query: str = "",
    search_place: str = "Tumu",
    page: int = 1,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> dict[str, Any]:
    empty_result = {
        "results": [],
        "total_matches": 0,
        "returned": 0,
        "page": page,
        "page_size": page_size,
        "has_more": False,
    }

    if not query:
        return empty_result

    if search_place not in _VALID_SEARCH_PLACES:
        search_place = "Tumu"

    page = max(1, page)
    page_size = max(1, min(page_size, MAX_PAGE_SIZE))
    start = (page - 1) * page_size

    try:
        client = await _get_client()

        query_bytes = query.encode("utf-8")
        encoded_query = base64.b64encode(query_bytes).decode("utf-8")

        payload = {
            "draw": 1,
            "columns": [
                {"data": None, "name": "", "searchable": True, "orderable": False, "search": {"value": "", "regex": False}},
                {"data": None, "name": "", "searchable": True, "orderable": False, "search": {"value": "", "regex": False}},
                {"data": None, "name": "", "searchable": True, "orderable": False, "search": {"value": "", "regex": False}}
            ],
            "order": [],
            "start": start,
            "length": page_size,
            "search": {"value": "", "regex": False},
            "parameters": {
                "AranacakIfade": encoded_query,
                "AranacakYer": search_place,
                "TamCumle": False,
                "MevzuatTur": 0,
                "GenelArama": True
            },
        }

        post_response = await client.post(
            SEARCH_URL,
            headers={
                "User-Agent": DEFAULT_USER_AGENT,
                "Content-Type": "application/json; charset=UTF-8",
                "Referer": HOME_URL,
                "X-Requested-With": "XMLHttpRequest"
            },
            json=payload,
        )

        if post_response.status_code != 200 or _looks_like_block_page(post_response):
            return {**empty_result, "error": f"Arama isteği sunucu tarafından reddedildi. HTTP {post_response.status_code}"}

        # [REMOVED]: Unreachable post_response.raise_for_status()

        try:
            response_data = post_response.json()
        except json.JSONDecodeError as exc:
            return {**empty_result, "error": f"JSON Çözümleme Hatası: {exc}"}

        records = _parse_search_payload(response_data)
        normalized = [_normalize_record(item) for item in records]
        total_matches = _extract_total_count(response_data)
        returned = len(normalized)

        return {
            "results": normalized,
            "total_matches": total_matches,
            "returned": returned,
            "page": page,
            "page_size": page_size,
            "has_more": (start + returned) < total_matches,
        }

    except Exception as exc:
        return {**empty_result, "error": f"Beklenmeyen Hata: {exc}"}


def _looks_like_pdf(content_type: str, url: str, raw_bytes: bytes) -> bool:
    if "application/pdf" in (content_type or "").lower():
        return True
    if url.lower().split("?", 1)[0].endswith(".pdf"):
        return True
    # Magic-bytes sniff: covers misconfigured servers that mislabel the
    # Content-Type (common enough on older gov sites) or omit it entirely.
    return raw_bytes[:5] == b"%PDF-"


def _extract_pdf_text(source_label: str, pdf_bytes: bytes) -> str:
    if PdfReader is None:
        return (
            f"[SOURCE: {source_label}]\n---\n"
            "Error: This document is only available as a PDF, and PDF text extraction is not "
            "available in this environment (the 'pypdf' package is not installed). "
            "Install it with `pip install pypdf` to enable PDF extraction."
        )

    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        page_texts: list[str] = []
        for page in reader.pages:
            try:
                page_texts.append(page.extract_text() or "")
            except Exception:
                continue  # skip a single unparseable page rather than failing the whole document

        full_text = "\n".join(t for t in page_texts if t).strip()
        if not full_text:
            return (
                f"[SOURCE: {source_label}]\n---\n"
                "Error: The PDF was fetched successfully but no extractable text was found — "
                "it is likely a scanned/image-only PDF, which would require OCR."
            )
        return _format_document_text(source_label, full_text)

    except Exception as exc:
        return f"[SOURCE: {source_label}]\n---\nError: Failed to parse PDF content: {exc}"



async def fetch_document_text(url: str) -> str:
    """Belirtilen doğrudan doküman URL'sinden tam metni çeker ve temizler."""
    if not url:
        return "Geçersiz doküman URL'si."

    try:
        client = await _get_client()

        # =====================================================================
        # [X-RAY V2] REVERSE-ENGINEER THE STATIC BACKEND URL
        # The web UI is an empty SPA shell loaded by Javascript.
        # We bypass it by constructing the direct PDF link from the query parameters.
        # =====================================================================
        parsed_url = urllib.parse.urlparse(url)
        query_params = urllib.parse.parse_qs(parsed_url.query)

        if "MevzuatNo" in query_params and "MevzuatTur" in query_params and "MevzuatTertip" in query_params:
            tur = query_params["MevzuatTur"][0]
            tertip = query_params["MevzuatTertip"][0]
            no = query_params["MevzuatNo"][0]
            
            # Predict the absolute PDF path on the government's static server
            direct_pdf_url = f"https://www.mevzuat.gov.tr/MevzuatMetin/{tur}.{tertip}.{no}.pdf"
            logger.debug(f"[X-RAY V2] Bypassing HTML shell. Requesting direct PDF: {direct_pdf_url}")
            
            try:
                # Try to fetch the PDF directly first
                pdf_response = await client.get(direct_pdf_url, headers=_build_headers(referer=url), timeout=15.0)
                if pdf_response.status_code == 200 and _looks_like_pdf(pdf_response.headers.get("Content-Type", ""), direct_pdf_url, pdf_response.content):
                    return _extract_pdf_text(direct_pdf_url, pdf_response.content)
            except Exception as e:
                logger.debug(f"[X-RAY V2] Direct PDF fetch failed: {e}")

        # =====================================================================
        # FALLBACK: STANDARD HTML/SHELL FETCH
        # =====================================================================
        response = await client.get(url, headers=_build_headers(referer=HOME_URL))

        if _looks_like_block_page(response):
            return "Doküman içeriği Cloudflare/Bot korumasına takıldı."

        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "")
        raw_bytes = response.content

        if _looks_like_pdf(content_type, url, raw_bytes):
            return _extract_pdf_text(url, raw_bytes)

        soup = BeautifulSoup(response.text, "html.parser")

        # Clean all heavy boilerplate (JS, CSS, Headers)
        for element in soup(["script", "style", "noscript", "meta", "link", "header", "footer"]):
            element.extract()

        text = soup.get_text(separator="\n", strip=True)

        if len(text) < 150 and "Mevzuat Bilgi Sistemi" in text:
             return f"[SOURCE: {url}]\n---\n[HATA: Metin bulunamadı. Doküman Javascript ile yükleniyor veya salt resim (scanned) formatında olabilir.]"

        return _format_document_text(url, text)

    except Exception as exc:
        return f"Doküman metni çekilirken bir hata oluştu: {exc}"

    
async def close_client() -> None:
    """Açık olan HTTPX istemcisini güvenli bir şekilde kapatır."""
    global _client, _client_loop
    if _client is not None:
        async with _client_lock:
            if _client is not None:
                await _client.aclose()
                _client = None
                _client_loop = None