"""
mcp_client/events.py

The agent emits these as it works, independent of any particular UI. cli.py
renders them with ANSI colors; a web UI would render the same events as
chat bubbles/badges instead. Keeping this decoupled from cli.py means a
future UI doesn't need to touch agent.py at all - it just supplies its own
`on_event` callback.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class EventType(Enum):
    LLM_THINKING = "llm_thinking"                # about to call the LLM
    TOOL_CALL_STARTED = "tool_call_started"       # LLM asked for a tool call
    TOOL_CALL_REJECTED = "tool_call_rejected"     # failed client-side schema validation
    TOOL_CALL_SUCCEEDED = "tool_call_succeeded"   # live call returned normally
    TOOL_CALL_STALE = "tool_call_stale"           # live call failed, served aged cache instead
    TOOL_CALL_FAILED = "tool_call_failed"         # live call failed, no cache fallback available
    LOOP_GUARD_TRIGGERED = "loop_guard_triggered"  # repeated/runaway calls detected
    FINAL_ANSWER = "final_answer"                 # turn complete


@dataclass
class AgentEvent:
    type: EventType
    tool_name: Optional[str] = None
    arguments: Optional[Dict[str, Any]] = None
    detail: str = ""
    duration_s: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)
