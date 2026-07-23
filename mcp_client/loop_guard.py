"""
mcp_client/loop_guard.py

Feature 7: protect against the agent getting stuck calling tools forever,
with two escalating tiers rather than jumping straight to a human prompt:

  1. SOFT: the same (tool, arguments) pair is repeated, or the turn's total
     call count crosses a "getting close" fraction of the hard cap. Instead
     of executing the call (or asking a human), the client auto-injects a
     nudge as that tool's response - "you've already tried this, consider a
     different approach" - and lets the model try to self-correct on its own
     next turn. No human interruption, and the redundant call is never
     actually sent to the transport.
  2. HARD: repeats/total calls exceed the outer limit even after a soft
     nudge already fired for that fingerprint. Pause and ask via
     `confirm_callback`.

IMPORTANT: approving past a HARD trigger grants exactly ONE more try for
THAT SPECIFIC (tool, arguments) pair - it does not wipe its repeat count
back to zero. Earlier versions fully reset all counters on approval, which
meant every "yes" silently bought the model two more free identical
retries before asking again (soft, then hard) - a real bug, confirmed
against a live transcript where 3 separate approvals only bought 2 extra
identical calls each time instead of escalating. The *total calls this
turn* budget still gets a full reset on approval (that one's a deliberate
"ok, give it more overall budget for this turn" signal, not a per-call
loophole).

NEAR-DUPLICATE DETECTION: exact-fingerprint matching alone misses a common
real-world pattern - a model thrashing pagination/formatting knobs
(page_number, tam_cumle, page_size...) on a query whose actual target
(aranacak_ifade, keyword, mevzuat_no...) never changes. Confirmed against a
live transcript where a model tried the same search 6+ times with only
page_number/tam_cumle differing, never tripping the exact-repeat counter
even once, and only getting caught ~9 calls in by the blunt total-calls
cap. `cosmetic_arg_keys` lists argument names treated as "refinement, not
retargeting" - when the REST of the arguments repeat while only these
differ, a soft nudge fires much earlier, with a message naming exactly
which knobs were being varied pointlessly.

Both trigger counts are turn-scoped and reset at the start of every new
user turn via reset().
"""
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Optional, Tuple

logger = logging.getLogger(__name__)

Fingerprint = Tuple[str, str]

# Argument names that typically refine/paginate a result set rather than
# change what's actually being searched for. Tool-specific, but these cover
# the mevzuat.gov.tr and bedesten tool schemas; override per-deployment if
# your tools use different parameter names for the same concept.
DEFAULT_COSMETIC_ARG_KEYS: FrozenSet[str] = frozenset({
    "page_number", "page_size", "tam_cumle", "case_sensitive",
    "max_results", "aranacak_yer", "basliktaAra",
})


@dataclass
class LoopCheckResult:
    severity: Optional[str]  # None, "soft", or "hard"
    reason: str = ""
    fingerprint: Optional[Fingerprint] = None  # so the caller can pass it straight to allow_continue()


@dataclass
class LoopGuard:
    repeat_limit: int = 2         # hard: identical (tool,args) calls beyond this -> ask human
    soft_repeat_limit: int = 1    # soft: identical (tool,args) calls beyond this -> auto-nudge
    hard_cap: int = 12            # hard: total calls this turn beyond this -> ask human
    soft_cap_fraction: float = 0.7  # soft: nudge once total calls cross this fraction of hard_cap
    near_duplicate_soft_limit: int = 2  # soft: core-query repeats (ignoring cosmetic args) beyond this -> auto-nudge
    cosmetic_arg_keys: FrozenSet[str] = field(default_factory=lambda: DEFAULT_COSMETIC_ARG_KEYS)

    _call_counts: Dict[Fingerprint, int] = field(default_factory=dict)
    _extra_allowance: Dict[Fingerprint, int] = field(default_factory=dict)
    _core_call_counts: Dict[Fingerprint, int] = field(default_factory=dict)
    _total_calls: int = 0

    def __post_init__(self):
        if self.soft_repeat_limit >= self.repeat_limit:
            raise ValueError("soft_repeat_limit must be < repeat_limit for the soft tier to fire first")

    def reset(self) -> None:
        """Call at the start of every new user turn."""
        self._call_counts.clear()
        self._extra_allowance.clear()
        self._core_call_counts.clear()
        self._total_calls = 0

    @staticmethod
    def _fingerprint(tool_name: str, arguments: Dict[str, Any]) -> Fingerprint:
        return (tool_name, json.dumps(arguments or {}, sort_keys=True, default=str))

    def _core_fingerprint(self, tool_name: str, arguments: Dict[str, Any]) -> Fingerprint:
        core_args = {k: v for k, v in (arguments or {}).items() if k not in self.cosmetic_arg_keys}
        return (tool_name, json.dumps(core_args, sort_keys=True, default=str))

    def record_and_check(self, tool_name: str, arguments: Dict[str, Any]) -> LoopCheckResult:
        """
        Records one intended call and returns the trigger tier (if any).
        Call this BEFORE executing the tool call - on "soft", the caller
        should skip the real execution and use the nudge as the response.
        """
        self._total_calls += 1
        fp = self._fingerprint(tool_name, arguments)
        self._call_counts[fp] = self._call_counts.get(fp, 0) + 1
        count = self._call_counts[fp]
        effective_repeat_limit = self.repeat_limit + self._extra_allowance.get(fp, 0)

        if count > effective_repeat_limit:
            return LoopCheckResult(
                "hard",
                f"The agent has called '{tool_name}' with the same arguments {count} times in this turn.",
                fingerprint=fp,
            )
        if count > self.soft_repeat_limit:
            return LoopCheckResult(
                "soft",
                f"'{tool_name}' has already been called with these exact arguments {count - 1} time(s) "
                f"in this turn without a new approach.",
                fingerprint=fp,
            )

        # Near-duplicate check: same core query, only cosmetic/pagination args differ.
        core_fp = self._core_fingerprint(tool_name, arguments)
        self._core_call_counts[core_fp] = self._core_call_counts.get(core_fp, 0) + 1
        core_count = self._core_call_counts[core_fp]
        if core_count > self.near_duplicate_soft_limit:
            varied_keys = sorted(k for k in (arguments or {}) if k in self.cosmetic_arg_keys)
            return LoopCheckResult(
                "soft",
                f"'{tool_name}' has been called {core_count} times with the same core query, only "
                f"varying {', '.join(varied_keys) if varied_keys else 'formatting options'}. This is "
                f"unlikely to surface new results - try a substantively different search term, a "
                f"different tool, or move on with what you already have.",
            )

        if self._total_calls > self.hard_cap:
            return LoopCheckResult(
                "hard",
                f"The agent has made {self._total_calls} tool calls in this single turn (limit: {self.hard_cap}).",
            )
        if self._total_calls > self.hard_cap * self.soft_cap_fraction:
            return LoopCheckResult(
                "soft",
                f"{self._total_calls} tool calls made this turn, approaching the {self.hard_cap} limit.",
            )

        return LoopCheckResult(None)

    def allow_continue(self, fingerprint: Optional[Fingerprint] = None) -> None:
        """
        Call when the user approves continuing past a HARD trigger.

        - The total-calls budget for this turn gets a full reset (the human
          is deliberately giving more overall runway).
        - If the trigger was a per-fingerprint repeat, that SPECIFIC
          fingerprint gets exactly +1 to its effective limit - one more
          try, not a clean slate. If it repeats past that too, hard fires
          again on the very next identical call instead of silently
          granting a whole new soft-then-hard cycle.
        """
        self._total_calls = 0
        if fingerprint is not None:
            self._extra_allowance[fingerprint] = self._extra_allowance.get(fingerprint, 0) + 1