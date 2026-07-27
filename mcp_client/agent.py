"""
mcp_client/agent.py

The agent loop:

  user message
      -> checkpoint state
      -> loop:
           prune old tool outputs / sliding-window evict if needed
           call LLM with current messages + tool specs
           if no tool_calls: done, return final answer
           for each tool_call:
             - validate arguments against the tool's JSON Schema (feature 1)
             - check loop guard (feature 7) -> may pause for user confirmation
             - execute via MCP transport with timeout/retry, falling back to
               an aged cache entry on failure (feature 3)
             - append the (possibly synthetic) tool result message
      -> on success: commit state
      -> on unrecoverable error: rollback state (feature 2), report cleanly
"""
import asyncio
import json
import logging
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional

from .config import ClientConfig
from .context_manager import (
    enforce_hard_token_budget,
    prune_old_tool_outputs,
    sliding_window_evict,
    truncate_oversized_tool_result,
)
from .events import AgentEvent, EventType
from .llm import LLMClient
from .loop_guard import LoopGuard
from .mcp_transport import MCPTransport, ToolCallError
from .schema_validator import ToolSchemaValidator
from .state import ConversationState
from .tool_cache import AgedToolCache

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are a Turkish legislation search assistant using MCP tools against "
    "mevzuat.gov.tr. CRITICAL: ALWAYS respond to the user in TURKISH (Türkçe). "
    "CRITICAL: keyword search is SUBSTRING MATCHING, not "
    "semantic search - an unquoted multi-word query matches as ONE LITERAL "
    "PHRASE, not 'these words somewhere.' Use AND to combine words instead.\n\n"
    "CRITICAL MANDATE: NEVER answer any Turkish legal, legislation, law, or article "
    "question using your internal training memory. ALL answers MUST be derived "
    "exclusively from information retrieved via MCP tool calls. On the initial "
    "question, you MUST issue a search tool call immediately.\n\n"    "SEARCH STRATEGY:\n"
    "1. Convert the question to 1-3 core Turkish legal keywords, not natural language.\n"
    "2. Prefer the shortest distinctive keyword first (e.g. 'vergi' before 'vergi usul kanunu').\n"
    "3. If a search returns 0 results or you're clearly not converging, do NOT retry "
    "trivial variations (page_number, tam_cumle, capitalization) - change the actual "
    "search term, or switch semantic=True, or move on with what you have.\n"
    "4. Tool result messages will tell you explicitly when pagination/retrying won't help "
       "and suggest specific next steps - follow them rather than guessing.\n"
    "5. WATCH FOR SEMANTIC FIXATION: if you've made 4+ searches built around the same "
    "specific term/name/institution and NONE found a clearly relevant document, that term "
    "itself may be wrong, unfamiliar, or misspelled - rephrasing it slightly and trying "
    "again will not help, because the problem isn't the phrasing. Instead:\n"
    "   a) Drop that specific term entirely and search on the broader regulatory concept "
    "instead (e.g. general teacher-appointment qualification requirements, not a specific "
    "institution name you're unsure about), or\n"
    "   b) Tell the user directly that the term might be unfamiliar or misspelled and ask "
    "them to confirm/clarify it, rather than continuing to guess variations of the same word.\n\n"
    
    "COMMON ABBREVIATIONS: KDV=katma değer vergisi, GV=gelir vergisi, KV=kurumlar vergisi, "
    "VUK=vergi usul, CMK=ceza muhakemesi, TCK=türk ceza kanunu, TTK=türk ticaret kanunu, "
    "TMK=türk medeni kanunu, SPK=sermaye piyasası, HMK=hukuk muhakemeleri, İİK=icra iflas, "
    "SGK=sosyal güvenlik (the server also auto-expands these when they appear alone or "
    "as the first word of a query - you don't have to expand them yourself, but it helps "
    "to know the target law name for follow-up search_within_* calls)."
)

# Async callback: (reason: str) -> bool (True = keep going, False = stop this turn)
ConfirmCallback = Callable[[str], Awaitable[bool]]
# Async callback: (event: AgentEvent) -> None. Fire-and-forget - the agent
# doesn't wait on a UI to render anything, it just notifies.
EventCallback = Callable[[AgentEvent], Awaitable[None]]


async def _noop_event_callback(event: AgentEvent) -> None:
    pass


class Agent:
    def __init__(
        self,
        config: ClientConfig,
        confirm_callback: ConfirmCallback,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        on_event: Optional[EventCallback] = None,
    ):
        self.config = config
        self.confirm_callback = confirm_callback
        self.on_event = on_event or _noop_event_callback
        self.state = ConversationState(system_prompt)
        self.llm = LLMClient(config.llm)
        self.transport = MCPTransport(config.server)
        self.validator = ToolSchemaValidator()
        self.tool_cache = AgedToolCache(
            fresh_ttl_s=config.resilience.tool_cache_ttl_s,
            max_stale_age_s=config.resilience.tool_cache_max_age_s,
        )
        self.loop_guard = LoopGuard(
            repeat_limit=config.resilience.loop_repeat_limit,
            hard_cap=config.resilience.loop_hard_cap_per_turn,
        )
        self._turn_index = 0
        self._round_index = 0

    async def _emit(self, event: AgentEvent) -> None:
        try:
            await self.on_event(event)
        except Exception:
            logger.exception("on_event callback raised - ignoring so it can't break the agent loop.")

    async def start(self) -> None:
        await self.transport.connect()
        for tool in self.transport.tools:
            self.validator.register(tool.name, tool.inputSchema)

    async def shutdown(self) -> None:
        await self.transport.close()

    async def run_turn(self, user_input: str) -> str:
        """Processes one user message end-to-end and returns the final assistant text."""
        self._turn_index += 1
        self.loop_guard.reset()
        self.state.checkpoint()

        try:
            self.state.add_user_message(user_input)
            answer = await self._agent_loop()
            self.state.commit()
            return answer
        except Exception as e:
            logger.exception("Unrecoverable error during turn - rolling back conversation state.")
            self.state.rollback()
            return (
                "Sorry, something went wrong while processing that and I've rolled back "
                f"to before your message so the conversation stays consistent. ({e})"
            )

    async def _agent_loop(self) -> str:
        tool_specs = self.transport.openai_tool_specs()
        tool_calls_executed_in_turn = 0

        while True:
            self._round_index += 1
            prune_old_tool_outputs(
                self.state.messages,
                current_round_index=self._round_index,
                max_age_rounds=self.config.resilience.prune_after_n_rounds,
                size_threshold_chars=self.config.resilience.prune_size_threshold_chars,
            )
            sliding_window_evict(
                self.state.messages,
                max_context_tokens=self.config.resilience.max_context_tokens,
                trigger_fraction=self.config.resilience.sliding_window_trigger_fraction,
                keep_recent_pairs=self.config.resilience.sliding_window_keep_recent_pairs,
            )
            # Last-resort safety net, independent of turn/round age - this is
            # what actually prevents the "96001 input tokens" crash: a single
            # still-in-progress turn that ballooned past the real model limit
            # before anything else had a chance to age it out.
            enforce_hard_token_budget(
                self.state.messages,
                hard_max_tokens=self.config.resilience.hard_max_context_tokens,
            )

            tool_choice = "auto"
            if tool_calls_executed_in_turn == 0:
                tool_choice = self.config.llm.force_tool_choice

            await self._emit(AgentEvent(type=EventType.LLM_THINKING))
            assistant_msg = await self.llm.chat(self.state.messages, tool_specs, tool_choice=tool_choice)
            self.state.add_message(assistant_msg)

            tool_calls = assistant_msg.get("tool_calls")
            if not tool_calls:
                if tool_calls_executed_in_turn == 0 and self.config.llm.require_tool_before_answer:
                    logger.warning("Agent Catch Guard: LLM returned text without tools on round 1. Reprompting.")
                    guard_msg = {
                        "role": "user",
                        "content": "CRITICAL SYSTEM ERROR: You attempted to answer directly without using any MCP legislation tools. You are FORCED to use an MCP search tool first to retrieve official legal texts before providing an answer. Call a tool now."
                    }
                    self.state.add_message(guard_msg)
                    continue

                answer = assistant_msg.get("content") or ""
                await self._emit(AgentEvent(type=EventType.FINAL_ANSWER, detail=answer))
                return answer

            tool_calls_executed_in_turn += len(tool_calls)

            # Independent tool calls in one turn are executed concurrently.
            results = await asyncio.gather(*[
                self._handle_one_tool_call(tc) for tc in tool_calls
            ])

            stop_answer = None
            for tool_call, (content, extra) in zip(tool_calls, results):
                stop_requested = content is _STOP_TURN
                # Every tool_call_id from this assistant message MUST get a
                # matching tool response, even the one that triggered a stop -
                # otherwise the conversation is left with an orphaned
                # tool_call_id and the *next* LLM call will be rejected by
                # the API. Real-world side effects from OTHER calls in this
                # same batch (e.g. a write that already succeeded) must also
                # never be silently dropped just because a sibling call hit
                # the loop guard.
                msg = {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": "Cancelled: stopped after loop-guard trigger, see message to user." if stop_requested else content,
                    "_round_index": self._round_index,
                    "_tool_name": tool_call["function"]["name"],
                }
                self.state.add_message(msg)
                if stop_requested:
                    stop_answer = extra

            if stop_answer is not None:
                return stop_answer

    async def _handle_one_tool_call(self, tool_call: Dict[str, Any]):
        name = tool_call["function"]["name"]
        raw_args = tool_call["function"].get("arguments") or "{}"
        try:
            arguments = json.loads(raw_args)
        except json.JSONDecodeError as e:
            await self._emit(AgentEvent(type=EventType.TOOL_CALL_REJECTED, tool_name=name, detail=f"bad JSON: {e}"))
            return (f"Error: could not parse arguments as JSON ({e}). Raw: {raw_args[:200]}", None)

        # Fire as soon as we know a call is happening, before validation -
        # this is the "search_mevzuat tool called" moment you want to see live.
        await self._emit(AgentEvent(type=EventType.TOOL_CALL_STARTED, tool_name=name, arguments=arguments))

        # --- Feature 1: client-side schema validation, before hitting the transport ---
        validation_error = self.validator.validate(name, arguments)
        if validation_error:
            logger.info(f"Rejected malformed call to '{name}' before transport: {validation_error}")
            await self._emit(AgentEvent(type=EventType.TOOL_CALL_REJECTED, tool_name=name, detail=validation_error))
            return (f"Error: invalid arguments for '{name}'. {validation_error}", None)

        # --- Feature 7: loop protection (soft auto-nudge, then hard human-ask) ---
        loop_result = self.loop_guard.record_and_check(name, arguments)
        if loop_result.severity == "soft":
            await self._emit(AgentEvent(
                type=EventType.LOOP_GUARD_TRIGGERED, tool_name=name,
                detail=loop_result.reason, extra={"severity": "soft"},
            ))
            # Don't actually execute the redundant call - feed the model a
            # nudge as if it were the tool's response, and give it a chance
            # to self-correct on its own next turn before ever asking a human.
            nudge = (
                f"[System note: {loop_result.reason} This call was not executed. "
                "Consider a different tool, different arguments, or explain to the "
                "user what's blocking progress instead of repeating this exact call.]"
            )
            return (nudge, None)

        if loop_result.severity == "hard":
            await self._emit(AgentEvent(
                type=EventType.LOOP_GUARD_TRIGGERED, tool_name=name,
                detail=loop_result.reason, extra={"severity": "hard"},
            ))
            keep_going = await self.confirm_callback(loop_result.reason)
            if not keep_going:
                return (
                    _STOP_TURN,
                    "I stopped because it looked like I might be stuck in a loop "
                    f"({loop_result.reason}). Let me know if you'd like me to try a different approach.",
                )
            self.loop_guard.allow_continue(loop_result.fingerprint)

        # --- Execute, with aged-cache fallback on failure (feature 3) ---
        started_at = time.monotonic()
        try:
            content = await self.transport.call_tool(
                name,
                arguments,
                timeout_s=self.config.resilience.tool_call_timeout_s,
                max_retries=self.config.resilience.tool_call_max_retries,
                retry_backoff_s=self.config.resilience.tool_call_retry_backoff_s,
            )
            self.tool_cache.put(name, arguments, content)  # cache the FULL result, truncate only what goes to the LLM
            content = truncate_oversized_tool_result(content, name, self.config.resilience.max_tool_result_chars)
            duration = time.monotonic() - started_at
            await self._emit(AgentEvent(
                type=EventType.TOOL_CALL_SUCCEEDED, tool_name=name, duration_s=duration,
                detail=content[:200],
            ))
            return (content, None)
        except ToolCallError as e:
            stale = self.tool_cache.get_stale_fallback(name, arguments)
            if stale is not None:
                value, age = stale
                value = truncate_oversized_tool_result(value, name, self.config.resilience.max_tool_result_chars)
                await self._emit(AgentEvent(
                    type=EventType.TOOL_CALL_STALE, tool_name=name,
                    duration_s=time.monotonic() - started_at, detail=f"{int(age)}s old, live call failed: {e}",
                ))
                minutes = int(age // 60)
                logger.warning(f"Live call to '{name}' failed, serving {minutes}min-old cached result instead.")
                return (f"[STALE CACHED RESULT, {minutes} min old - live call failed: {e}]\n{value}", None)
            await self._emit(AgentEvent(
                type=EventType.TOOL_CALL_FAILED, tool_name=name,
                duration_s=time.monotonic() - started_at, detail=str(e),
            ))
            return (f"Error: '{name}' failed and no cached fallback is available. {e}", None)


_STOP_TURN = object()