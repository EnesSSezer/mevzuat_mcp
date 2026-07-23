"""
mcp_client/config.py

All tunables in one place, env-var overridable. Mirrors the pattern already
used in embedder.py (TUBITAK_EMBEDDINGS_*) for the chat endpoint.
"""
import os
import shlex
from dataclasses import dataclass, field
from typing import List


@dataclass
class LLMConfig:
    base_url: str = os.getenv(
        "TUBITAK_CHAT_BASE_URL", "https://ai-api.tubitak.gov.tr/vllm/gptoss-120b/v1"
    )
    model: str = os.getenv("TUBITAK_CHAT_MODEL", "GPTOSS-120B")
    api_key: str = os.getenv("TUBITAK_API_KEY", "EMPTY")
    temperature: float = float(os.getenv("TUBITAK_CHAT_TEMPERATURE", "0.2"))
    request_timeout_s: float = float(os.getenv("TUBITAK_CHAT_TIMEOUT_S", "60"))


@dataclass
class ServerConfig:
    # e.g. MCP_SERVER_COMMAND="python mevzuat_mcp_server.py"
    command: str = os.getenv("MCP_SERVER_COMMAND", "python mevzuat_mcp_server.py")

    @property
    def argv(self) -> List[str]:
        return shlex.split(self.command)


@dataclass
class ResilienceConfig:
    # --- tool call execution ---
    tool_call_timeout_s: float = float(os.getenv("MCP_TOOL_TIMEOUT_S", "120"))
    tool_call_max_retries: int = int(os.getenv("MCP_TOOL_MAX_RETRIES", "2"))
    tool_call_retry_backoff_s: float = float(os.getenv("MCP_TOOL_RETRY_BACKOFF_S", "1.5"))

    # --- aged-cache fallback (feature 3) ---
    tool_cache_ttl_s: float = float(os.getenv("MCP_TOOL_CACHE_TTL_S", "900"))       # "fresh" window
    tool_cache_max_age_s: float = float(os.getenv("MCP_TOOL_CACHE_MAX_AGE_S", "86400"))  # hard cutoff for even a stale serve

    # --- output pruning (feature 4) ---
    # NOTE: age is measured in "rounds" (one per LLM roundtrip), not user
    # turns - see context_manager.prune_old_tool_outputs for why. A single
    # long agentic turn (many tool calls before the model answers) can
    # prune its own early results this way, not just older completed turns.
    prune_after_n_rounds: int = int(os.getenv("MCP_PRUNE_AFTER_ROUNDS", "2"))
    prune_size_threshold_chars: int = int(os.getenv("MCP_PRUNE_SIZE_CHARS", "1500"))

    # NEW: immediate per-call truncation cap. Applied the instant a tool
    # result comes back, before it's even appended to state - the direct
    # fix for a single call (e.g. get_mevzuat_content dumping a full raw
    # law) being large enough to blow the budget by itself in one shot.
    max_tool_result_chars: int = int(os.getenv("MCP_MAX_TOOL_RESULT_CHARS", "8000"))

    # --- sliding window (feature 5) ---
    # Soft/efficiency target: keep the "normal" context lean for latency and
    # cost, evicting old completed turns once we cross this.
    max_context_tokens: int = int(os.getenv("MCP_MAX_CONTEXT_TOKENS", "24000"))
    sliding_window_trigger_fraction: float = float(os.getenv("MCP_SLIDING_TRIGGER_FRACTION", "0.75"))
    sliding_window_keep_recent_pairs: int = int(os.getenv("MCP_SLIDING_KEEP_RECENT", "3"))

    # NEW: hard safety-net ceiling, checked right before every LLM call,
    # regardless of turn/round age. Set this comfortably below your actual
    # model's real context limit (e.g. the TÜBİTAK endpoint reported a
    # 96000-token hard limit in testing) - our token counting is an
    # approximation (see context_manager.py), so leave real margin.
    hard_max_context_tokens: int = int(os.getenv("MCP_HARD_MAX_CONTEXT_TOKENS", "80000"))

    # --- loop protection (feature 7) ---
    # For weaker/open-weight models that don't self-correct well (observed with
    # some models via live testing), operators can tighten these via env vars:
    #   MCP_LOOP_REPEAT_LIMIT=2 MCP_LOOP_HARD_CAP=8
    # Do NOT set MCP_LOOP_REPEAT_LIMIT=1 - LoopGuard requires soft_repeat_limit <
    # repeat_limit for the two-tier soft-then-hard escalation to work at all; at
    # 1, hard fires on the very first repeat and the soft nudge tier never gets
    # a chance to let the model self-correct first.
    loop_repeat_limit: int = int(os.getenv("MCP_LOOP_REPEAT_LIMIT", "2"))       # identical (tool,args) calls
    loop_hard_cap_per_turn: int = int(os.getenv("MCP_LOOP_HARD_CAP", "12"))     # total tool calls in one user turn


@dataclass
class ClientConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    resilience: ResilienceConfig = field(default_factory=ResilienceConfig)