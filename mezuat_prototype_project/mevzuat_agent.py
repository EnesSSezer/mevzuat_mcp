from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

import httpx
from openai import APIConnectionError, AsyncOpenAI, OpenAIError
from dotenv import load_dotenv

from mevzuat_tools import fetch_document_text, search_mevzuat

logger = logging.getLogger(__name__)

LEGAL_PERSONA_PROMPT = (
    "You are an expert, precise Turkish Legal Assistant. Your job is to answer user queries using ONLY "
    "information retrieved from the provided search tools.\n\n"
    "CRITICAL SEARCH REALITY: The target search engine is extremely rigid and only performs exact literal string matching. "
    "Official Turkish legislation NEVER uses abbreviations or acronyms (e.g., it will never say 'TFF', 'SGK', 'BDDK'). "
    "Instead, it strictly uses the full official names ('Türkiye Futbol Federasyonu', 'Sosyal Güvenlik Kurumu').\n"
    "1. ABSOLUTELY FORBIDDEN to use acronyms or abbreviations in search queries. You MUST expand them to their full official legal names.\n"
    "2. Strip all Turkish suffixes and search for uninflected root words or phrases.\n"
    "3. Keep queries focused on the core legal subject. If a tool call returns an empty list, immediately use your next iteration to try an alternative keyword or related phrasing.\n\n"
    "For every claim or legal citation you make, you MUST explicitly cite the source metadata (e.g. Law Name, Law Number, or Article). Do not guess or extrapolate. "
    "If the context does not contain the answer, explicitly state that it could not be found in the current legislation.\n\n"
    "ERROR SENTENCE — READ CAREFULLY: 'Sistem hatası nedeniyle bilgiye ulaşılamadı' is reserved EXCLUSIVELY for the "
    "case where a tool call's JSON result literally contains an \"error\" field. When that happens, your ENTIRE "
    "reply must be that exact sentence and nothing else — no partial answer, no explanation before or after it. "
    "If you already have real information to answer with (search results, document text), give that answer "
    "and do NOT mention this sentence at all, even as a caveat or disclaimer at the end."
)

BOUNCER_SYSTEM_PROMPT = (
    "You are a binary classification filter for a Turkish legal database. Given the user's "
    "message below, decide: does this relate to Turkish laws, regulations, decrees, or legal "
    "procedures? Reply strictly with TRUE or FALSE, nothing else."
)

MAX_ITERATIONS = 3
DEFAULT_BOUNCER_TEMPERATURE = 0.0
DEFAULT_ANSWER_TEMPERATURE = 0.2

# ─── Conversation-memory tuning (all overridable via env) ──────────────────
# Coarse cap: how many finished user/assistant turn-pairs we keep at most.
DEFAULT_MAX_HISTORY_TURNS = 6
# Fine cap: rough token budget for the persisted history alone.
DEFAULT_MAX_CONTEXT_TOKENS = 8000
# Fraction of the total context budget reserved for persisted history; the
# rest is left as headroom for this turn's live tool-call/tool-result
# scaffolding (search results + fetched document text) plus the response.
DEFAULT_HISTORY_TOKEN_SHARE = 0.6
# Hard per-message character ceiling, independent of the rolling truncation,
# so a single oversized paste can't blow the budget math by itself.
DEFAULT_MAX_MESSAGE_CHARS = 4000
# Very rough, model-agnostic token estimate (works acceptably across
# different tokenizers without depending on any one model's exact vocab).
CHARS_PER_TOKEN_ESTIMATE = 4

load_dotenv()


def _configure_logging() -> None:
    logging.basicConfig(
        level=os.getenv("MEVZUAT_LOG_LEVEL", "INFO").upper(),
        format="%(levelname)s:%(name)s:%(message)s",
    )


def _get_temperature(env_name: str, default: float) -> float:
    raw_value = os.getenv(env_name)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        return float(raw_value)
    except ValueError:
        return default


def _get_float_env(env_name: str, default: float) -> float:
    return _get_temperature(env_name, default)


def _get_int_env(env_name: str, default: int) -> int:
    raw_value = os.getenv(env_name)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default


def _max_history_turns() -> int:
    return _get_int_env("MEVZUAT_MAX_HISTORY_TURNS", DEFAULT_MAX_HISTORY_TURNS)


def _max_context_tokens() -> int:
    return _get_int_env("MEVZUAT_MAX_CONTEXT_TOKENS", DEFAULT_MAX_CONTEXT_TOKENS)


def _history_token_share() -> float:
    return _get_float_env("MEVZUAT_HISTORY_TOKEN_SHARE", DEFAULT_HISTORY_TOKEN_SHARE)


def _max_message_chars() -> int:
    return _get_int_env("MEVZUAT_MAX_MESSAGE_CHARS", DEFAULT_MAX_MESSAGE_CHARS)


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // CHARS_PER_TOKEN_ESTIMATE)


def _message_tokens(message: dict[str, Any]) -> int:
    content = message.get("content") or ""
    # Small flat overhead per message for role/formatting tokens.
    return _estimate_tokens(content) + 4


def _total_tokens(messages: list[dict[str, Any]]) -> int:
    return sum(_message_tokens(m) for m in messages)


class ConversationSession:
    """
    Holds only *finished* user/assistant turns across CLI turns — never the
    intermediate tool-call/tool-result scaffolding a single turn's ReAct loop
    generates (search results, fetched document text). That scaffolding can
    be several KB per turn and must never leak into next turn's context.

    Bounded by both a turn-count cap and a rough token budget, so a long
    conversation can't silently blow the model's context window when
    combined with the current turn's live tool payloads.
    """

    def __init__(self) -> None:
        self.history: list[dict[str, Any]] = []

    def add_turn(self, user_text: str, assistant_text: str) -> None:
        bounded_user_text = (user_text or "")[: _max_message_chars()]
        bounded_assistant_text = (assistant_text or "")[: _max_message_chars()]
        self.history.append({"role": "user", "content": bounded_user_text})
        self.history.append({"role": "assistant", "content": bounded_assistant_text})
        self._trim()

    def _trim(self) -> None:
        # 1) Coarse cap: keep at most N user/assistant turn-pairs.
        max_messages = _max_history_turns() * 2
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]

        # 2) Fine cap: drop the oldest whole turn-pairs until under the
        #    token budget. Never truncate mid-turn, never touch the system
        #    prompt or the newest question (those aren't part of `history`).
        budget = int(_max_context_tokens() * _history_token_share())
        while self.history and _total_tokens(self.history) > budget:
            drop_count = 2 if len(self.history) >= 2 else 1
            self.history = self.history[drop_count:]

    def context_messages(self) -> list[dict[str, Any]]:
        return list(self.history)


def _build_client() -> AsyncOpenAI:
    base_url = os.getenv("MEVZUAT_LLM_BASE_URL", "http://localhost:11434/v1")
    api_key = os.getenv("MEVZUAT_LLM_API_KEY", "empty")
    return AsyncOpenAI(base_url=base_url, api_key=api_key)


def _llm_model_name() -> str:
    return os.getenv("MEVZUAT_LLM_MODEL", os.getenv("MEVZUAT_LLM_MODEL_NAME", "llama3.1"))


def _tool_specs() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "search_mevzuat",
                "description": (
                    "Mevzuat.gov.tr ana arama çubuğunu kullanarak arama yapar."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string", 
                            "description": (
                                "ÇOK KRİTİK TALİMAT: Mevzuat veritabanı harfi harfine tam eşleşme arar ve resmi metinlerde KISALTMALAR ASLA KULLANILMAZ. "
                                "Arama sorgusunda 'TFF', 'SGK', 'BDDK', 'MHK' gibi kısaltmaları KESİNLİKLE kullanmayın. Bunları her zaman resmi ve tam "
                                "açılımlarıyla yazın (Örn: 'Türkiye Futbol Federasyonu'). Ayrıca kelimelerdeki ekleri (-ı, -si, -nin) temizleyerek yalın kök haliyle aratın.\n"
                                "Eğer ilk aramanız boş dönerse, bir sonraki ajan iterasyonunda mutlaka alternatif bir resmi kelime veya tam açılım deneyin."
                            )
                        },
                        "search_place": {
                            "type": "string",
                            "description": (
                                "Aramanın yapılacağı yer. 'Tumu' hem başlıkta hem doküman içeriğinde arar (tavsiye "
                                "edilen, ilk denemede bunu kullanın). Sadece mevzuat başlıklarında aramak için "
                                "'Baslik', sadece doküman metninin içeriğinde aramak için 'Icerik' gönderin. "
                                "'Baslik' ile sonuç bulunamazsa ve terimin metin içinde geçebileceğini "
                                "düşünüyorsanız 'Icerik' ile tekrar deneyin."
                            ),
                            "enum": ["Tumu", "Baslik", "Icerik"],
                            "default": "Tumu"
                        },
                        "page": {
                            "type": "integer",
                            "description": (
                                "Sayfa numarası (1'den başlar). İlk aramayı her zaman page=1 ile yapın. Sonuç "
                                "metasında 'more_results_available: true' görürseniz ve ilk sayfadaki hiçbir sonuç "
                                "uygun değilse, aynı sorguyla page=2 gönderip devamını getirin."
                            ),
                            "default": 1
                        },
                        "page_size": {
                            "type": "integer",
                            "description": "Sayfa başına getirilecek sonuç sayısı (varsayılan 20, en fazla 30).",
                            "default": 20
                        }
                    },
                    "required": ["query"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "fetch_document_text",
                "description": "Bir mevzuat dokümanının doğrudan URL'sinden tam metni çeker. PDF veya HTML olsun fark etmez.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "Dokümanın doğrudan metin URL'si."},
                    },
                    "required": ["url"],
                    "additionalProperties": False,
                },
            },
        },
    ]

def _format_search_results(result: dict[str, Any]) -> str:
    records = result.get("results") or []
    total_matches = result.get("total_matches", 0)
    returned = result.get("returned", len(records))
    page = result.get("page", 1)
    has_more = result.get("has_more", False)

    if not records:
        return (
            f"[SEARCH META: total_matches=0 | page={page}]\n"
            "No search results were returned. Try a different keyword, a different search_place, "
            "or check for abbreviations that need to be expanded to their full official name."
        )

    blocks: list[str] = []
    for record in records:
        header = record.get("source_header", "[SOURCE: Unknown | Law No: N/A | Gazette Date: N/A]")
        body = {
            "title": record.get("title", ""),
            "law_number": record.get("law_number", ""),
            "gazette_date": record.get("gazette_date", ""),
            "document_url": record.get("document_url", ""),
            "document_id": record.get("document_id", ""),
        }
        blocks.append(f"{header}\n---\n{json.dumps(body, ensure_ascii=False)}")

    meta_line = (
        f"[SEARCH META: showing {returned} of {total_matches} total matches | page={page} | "
        f"more_results_available={has_more}]"
    )
    if has_more:
        meta_line += (
            "\nIf none of these results look right, call search_mevzuat again with the same query "
            "and a higher 'page' number to see more matches, or narrow the query."
        )

    return meta_line + "\n\n" + "\n\n".join(blocks)


def _normalize_bool_label(text: str) -> bool:
    return text.strip().upper().startswith("TRUE")


async def _route_query(client: AsyncOpenAI, query: str) -> bool:
    """
    Sorgunun Türk mevzuatıyla ilgili olup olmadığını kontrol eden filtre.
    Hatalar yakalanmadan doğrudan üst katmandaki merkezi hata yakalama bloğuna fırlatılır.

    Deliberately kept as a single-turn classification (system + one user
    message, no extra context). This is only ever called for the FIRST
    message of a session (see answer_mevzuat_query) — once a legal
    conversation is underway, follow-ups are no longer re-classified, so
    this never has to judge a short message sitting after a long block of
    prior context. That combination (max_tokens=4 + a lot of prior context)
    is what let a tiny classifier drift off "TRUE"/"FALSE" and default-reject
    perfectly good follow-up questions.
    """
    response = await client.chat.completions.create(
        model=_llm_model_name(),
        messages=[
            {"role": "system", "content": BOUNCER_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        temperature=_get_temperature("MEVZUAT_LLM_BOUNCER_TEMPERATURE", DEFAULT_BOUNCER_TEMPERATURE),
        max_tokens=4,
    )
    content = response.choices[0].message.content or ""
    return _normalize_bool_label(content)


async def answer_mevzuat_query(query: str, session: ConversationSession) -> str:
    """
    Public entry point. Handles the domain gate, delegates to the tool-calling
    ReAct loop for the actual answer, and — only on a genuine successful
    exchange — persists the (query, answer) pair into the session's bounded
    history. Hard failures (connection errors, unexpected exceptions) never
    get written into history, so a flaky turn doesn't pollute later context.

    The domain gate only runs on the FIRST message of a session. Natural
    follow-ups ("bu kararnamenin içeriğini özetleyebilir misin") often don't
    restate any legal keywords on their own and can't be reliably judged in
    isolation by a tiny max_tokens=4 classifier call — and there's no need
    to re-judge them: once a legal conversation is underway, LEGAL_PERSONA_PROMPT
    itself will decline anything it can't actually answer from the tools,
    which is a safe enough backstop for the rare off-topic pivot mid-session.
    """
    try:
        client = _build_client()

        if not session.history and not await _route_query(client, query):
            return "I am a specialized legal assistant. I can only answer questions related to Turkish legislation."

        answer_text = await _run_agent_loop(client, session, query)
        session.add_turn(query, answer_text)
        return answer_text

    except (APIConnectionError, OpenAIError, httpx.ConnectError, httpx.TimeoutException):
        return (
            "[ERROR: Could not connect to the configured LLM endpoint. Check MEVZUAT_LLM_BASE_URL, "
            "MEVZUAT_LLM_API_KEY, and that the model server is running.]"
        )
    except Exception as general_exc:
        # GLOBAL INSULATION SAFEGUARD: Fully protects CLI workflows against raw trace dumps
        return f"[ERROR: An unexpected pipeline error occurred: {general_exc}]"


async def _run_agent_loop(client: AsyncOpenAI, session: ConversationSession, query: str) -> str:
    """The tool-calling ReAct loop itself. Raises on hard failures; the caller
    (answer_mevzuat_query) is responsible for catching and for session bookkeeping."""
    bounded_query = query[: _max_message_chars()]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": LEGAL_PERSONA_PROMPT},
        *session.context_messages(),
        {"role": "user", "content": bounded_query},
    ]
    tool_call_count = 0
    breaker_message_added = False
    tools = _tool_specs()

    while True:
        # COLLAPSED STATE ROUTING: Single execution loop surface
        use_tools = tool_call_count < MAX_ITERATIONS

        if not use_tools and not breaker_message_added:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "SYSTEM: You have reached the maximum allowed search attempts (3). You must immediately stop "
                        "searching and formulate the best possible answer using the context you have gathered so far. "
                        "If you do not have the answer, state clearly that the specific information could not be found."
                    ),
                }
            )
            breaker_message_added = True

        completion = await client.chat.completions.create(
            model=_llm_model_name(),
            messages=messages,
            tools=tools if use_tools else None,
            tool_choice="auto" if use_tools else "none",
            temperature=_get_temperature("MEVZUAT_LLM_ANSWER_TEMPERATURE", DEFAULT_ANSWER_TEMPERATURE),
        )

        # DEFENSIVE GUARD: Avoid downstream IndexError crash on empty response contexts
        if not completion.choices:
            return "The agent pipeline was interrupted due to an empty completion response from the model."

        message = completion.choices[0].message

        # ================== UNIFIED LOGICAL MONITORING ==================
        # Routed through the logger (not print) so it respects MEVZUAT_LOG_LEVEL
        # instead of always firing regardless of configuration.
        if logger.isEnabledFor(logging.DEBUG):
            tool_call_summary = (
                [f"{tc.function.name}({tc.function.arguments})" for tc in message.tool_calls]
                if message.tool_calls
                else "none"
            )
            logger.debug(
                "[STEP %d] content=%r tool_calls=%s",
                tool_call_count + 1,
                message.content,
                tool_call_summary,
            )
        # ================================================================

        # SPEC-FAITHFUL MAPPING: Replaces raw manual maps with full spec compliance (preserves content: null)
        messages.append(message.model_dump(exclude_unset=True))

        tool_calls = message.tool_calls or []
        content = message.content or ""

        # Standard exit checking or safety fallback if the model forces tools beyond bounds
        if not tool_calls or not use_tools:
            return content.strip() or "The answer could not be found in the current legislation."

        tool_call_count += 1

        for tool_call in tool_calls:
            # DEFENSIVE STEP RUNNER: Keeps loops fully operational during syntax errors
            try:
                arguments = json.loads(tool_call.function.arguments or "{}")

                if tool_call.function.name == "search_mevzuat":
                    result = await search_mevzuat(**arguments)
                    logger.debug("[TOOL OUTPUT] search_mevzuat -> %r", result)

                    if isinstance(result, dict) and result.get("error"):
                        tool_content = json.dumps(result, ensure_ascii=False)
                    else:
                        tool_content = _format_search_results(result)

                elif tool_call.function.name == "fetch_document_text":
                    fetched_text = await fetch_document_text(**arguments)
                    logger.debug("[TOOL OUTPUT] fetch_document_text -> %r", fetched_text)
                    tool_content = fetched_text
                else:
                    tool_content = json.dumps({"error": f"Unknown tool: {tool_call.function.name}"}, ensure_ascii=False)

            except json.JSONDecodeError as json_exc:
                # SELF-CORRECTION LOOP FEEDBACK: Feed back error metrics so local models fix syntax mistakes
                tool_content = json.dumps({
                    "error": "Malformed JSON arguments provided for tool invocation.",
                    "details": str(json_exc),
                    "suggestion": "Rewrite the arguments in strict JSON format matching the schema parameters exactly."
                }, ensure_ascii=False)

            except Exception as tool_exc:
                tool_content = json.dumps({"error": f"Tool execution failed: {tool_exc}"}, ensure_ascii=False)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": tool_content,
                }
            )


async def _run_cli(argv_query: str) -> None:
    """
    Everything the CLI does lives inside ONE event loop for the entire process
    lifetime. Previously each turn called asyncio.run(...) separately, which
    spins up (and fully tears down) a fresh loop every time — but the shared
    httpx.AsyncClient singleton in mevzuat_tools is bound to whichever loop
    created it, so reusing it from a *new* loop on the next turn raised
    "Event loop is closed" style errors. Running the whole session inside a
    single asyncio.run() call (see main(), below) fixes this at the root;
    close_client() is called in `finally` so teardown always happens on the
    same loop that built the client.
    """
    from mevzuat_tools import close_client

    session = ConversationSession()

    try:
        if argv_query:
            print(await answer_mevzuat_query(argv_query, session))
            return

        print("Enter your query. Type 'exit' or press Enter on an empty line to quit.")
        while True:
            try:
                # input() is blocking; run it in a worker thread so it never
                # blocks the event loop the rest of the app depends on.
                query = (await asyncio.to_thread(input, "mevzuat> ")).strip()
            except EOFError:
                break

            if not query or query.lower() in {"exit", "quit"}:
                break

            print(await answer_mevzuat_query(query, session))
            print()
    finally:
        await close_client()


def main() -> None:
    import sys

    _configure_logging()
    argv_query = " ".join(sys.argv[1:]).strip()
    asyncio.run(_run_cli(argv_query))


if __name__ == "__main__":
    main()