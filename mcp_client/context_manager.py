"""
mcp_client/context_manager.py

Feature 4: prune large tool outputs after N turns -> short placeholder.
Feature 5: sliding-window FIFO eviction once token usage crosses a threshold.
Feature 6: instead of just dropping evicted turns, fold them into a running
           rolling-summary system message so old context isn't lost outright.

Token counting: tries tiktoken's cl100k_base encoding as a reasonable proxy.
This is NOT the real Qwen3-Coder tokenizer (nothing publicly guarantees an
exact match), so treat `max_context_tokens` as a conservative budget, not a
precise limit - it's deliberately approximate, cheap, and dependency-light.
"""
import logging
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text: str) -> int:
        return len(_ENC.encode(text))
except Exception:
    logger.warning("tiktoken unavailable - falling back to a chars/4 token estimate.")

    def count_tokens(text: str) -> int:
        return max(1, len(text) // 4)


def _message_text(message: Dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # OpenAI-style multi-part content blocks
        return " ".join(str(part.get("text", part)) if isinstance(part, dict) else str(part) for part in content)
    return str(content or "")


def count_message_tokens(message: Dict[str, Any]) -> int:
    return count_tokens(_message_text(message)) + count_tokens(str(message.get("tool_calls", "")))


def total_tokens(messages: List[Dict[str, Any]]) -> int:
    return sum(count_message_tokens(m) for m in messages)


# ---------------------------------------------------------------------------
# NEW: immediate per-call truncation - the actual proximate fix for the
# 96k-token crash. A single get_mevzuat_content call can dump an entire raw
# law's text in one response; nothing about "age" or "sliding windows"
# helps if ONE tool result is already most of the budget by itself. This
# runs the instant a result comes back, before it's even appended to state.
# ---------------------------------------------------------------------------

def truncate_oversized_tool_result(content: str, tool_name: str, max_chars: int) -> str:
    if not content or len(content) <= max_chars:
        return content
    first_line = next((ln.strip() for ln in content.splitlines() if ln.strip()), "")
    return (
        f"{content[:max_chars]}\n\n"
        f"[Truncated: '{tool_name}' returned {len(content)} chars, showing the first {max_chars}. "
        f"If you need a specific part, use a search_within_* tool with a keyword instead of "
        f"fetching the full document.]"
    )


# ---------------------------------------------------------------------------
# NEW: hard safety-net budget enforcer. prune_old_tool_outputs and
# sliding_window_evict are both AGE-based - they only ever touch messages
# from turns/rounds older than some threshold. Neither one can do anything
# about a single still-in-progress turn that's already ballooned past the
# real model limit (this is exactly what caused the 400 "96001 input
# tokens" crash - one turn, 7 tool calls, never aged out because it was
# still turn 1 of 2). This runs LAST, right before every LLM call, as an
# unconditional last resort: if we're still over hard_max_tokens after the
# normal age-based passes, forcibly shrink the largest tool messages
# (oldest-largest first), regardless of which turn/round they belong to,
# until back under budget. The system prompt, the rolling summary, and the
# most recent user message are never touched.
# ---------------------------------------------------------------------------

def enforce_hard_token_budget(messages: List[Dict[str, Any]], hard_max_tokens: int) -> None:
    if total_tokens(messages) <= hard_max_tokens:
        return

    protected_ids = {id(messages[0])}  # system prompt
    for m in messages:
        if m.get(_SUMMARY_TAG):
            protected_ids.add(id(m))
    last_user = next((m for m in reversed(messages) if m.get("role") == "user"), None)
    if last_user is not None:
        protected_ids.add(id(last_user))

    # Candidates: tool messages not already protected, largest first so we
    # free the most space per truncation.
    candidates = [m for m in messages if m.get("role") == "tool" and id(m) not in protected_ids]
    candidates.sort(key=lambda m: len(m.get("content", "") or ""), reverse=True)

    for msg in candidates:
        if total_tokens(messages) <= hard_max_tokens:
            break
        if msg.get("_hard_truncated"):
            continue
        content = msg.get("content", "") or ""
        if len(content) < 500:
            continue  # not worth truncating further, too small to matter
        msg["content"] = (
            f"[Dropped: this tool result was removed to stay under the hard context limit "
            f"({hard_max_tokens} tokens). Original was {len(content)} chars from "
            f"'{msg.get('_tool_name', 'unknown')}'. Re-call the tool if you still need this.]"
        )
        msg["_hard_truncated"] = True

    remaining = total_tokens(messages)
    if remaining > hard_max_tokens:
        logger.error(
            f"Still over hard token budget after truncating all eligible tool messages "
            f"({remaining} > {hard_max_tokens}) - likely the remaining protected messages "
            f"(system prompt / summary / latest user message) are themselves too large."
        )
    else:
        logger.warning(f"Hard budget enforcer truncated tool messages down to {remaining} tokens.")


# ---------------------------------------------------------------------------
# Feature 4: prune large tool outputs after N turns
# ---------------------------------------------------------------------------

def prune_old_tool_outputs(
    messages: List[Dict[str, Any]],
    current_round_index: int,
    max_age_rounds: int,
    size_threshold_chars: int,
) -> None:
    """
    Mutates `messages` in place. Any tool-result message tagged with
    `_round_index` older than `max_age_rounds`, and whose content is larger
    than `size_threshold_chars`, gets replaced with a short placeholder.
    Already-pruned messages are skipped (idempotent).

    IMPORTANT: `_round_index` increments on every LLM roundtrip, not just on
    every new user message. A "round" is one iteration of the tool-calling
    loop. This is what lets a single long-running turn (many tool calls
    before the model finally answers) prune its own early tool results
    once it's several rounds deep - the earlier turn-based version could
    never touch anything inside the CURRENT turn, no matter how large it
    got, since eviction was only ever keyed on completed user turns.
    """
    for msg in messages:
        if msg.get("role") != "tool":
            continue
        if msg.get("_pruned"):
            continue
        round_index = msg.get("_round_index")
        if round_index is None:
            continue
        if current_round_index - round_index < max_age_rounds:
            continue

        content = msg.get("content", "") or ""
        if len(content) <= size_threshold_chars:
            continue

        line_count = content.count("\n") + 1
        first_line = next((ln.strip() for ln in content.splitlines() if ln.strip()), "")
        looks_like_error = "error" in content[:200].lower()
        takeaway = f"looked like an error: {first_line[:120]}" if looks_like_error else f"first line: \"{first_line[:120]}\""

        msg["content"] = (
            f"[Truncated: {len(content)} chars / {line_count} lines from tool "
            f"'{msg.get('_tool_name', 'unknown')}' dropped after {max_age_rounds} rounds. "
            f"Summary: {takeaway}]"
        )
        msg["_pruned"] = True
        logger.debug(f"Pruned tool output from round {round_index} (now round {current_round_index}).")


# ---------------------------------------------------------------------------
# Features 5 & 6: sliding window eviction + rolling summary of what left
# ---------------------------------------------------------------------------

_SUMMARY_TAG = "_rolling_summary"


def _heuristic_summarize_pair(user_msg: Dict[str, Any], assistant_msgs: List[Dict[str, Any]]) -> str:
    """Cheap, no-extra-API-call summary of one evicted user turn."""
    user_text = _message_text(user_msg).strip().replace("\n", " ")
    tool_names = []
    final_answer = ""
    for m in assistant_msgs:
        for tc in (m.get("tool_calls") or []):
            name = tc.get("function", {}).get("name")
            if name:
                tool_names.append(name)
        if m.get("role") == "assistant" and m.get("content"):
            final_answer = _message_text(m).strip().replace("\n", " ")

    parts = [f'User asked: "{user_text[:150]}"']
    if tool_names:
        parts.append(f"tools used: {', '.join(dict.fromkeys(tool_names))}")
    if final_answer:
        parts.append(f'answered: "{final_answer[:150]}"')
    return " | ".join(parts)


def sliding_window_evict(
    messages: List[Dict[str, Any]],
    max_context_tokens: int,
    trigger_fraction: float,
    keep_recent_pairs: int,
) -> None:
    """
    Mutates `messages` in place. messages[0] is always the system prompt and
    is never evicted. A rolling-summary system message (tagged) is kept
    right after it and updated (never duplicated) as older turns are folded in.
    """
    if total_tokens(messages) < max_context_tokens * trigger_fraction:
        return

    # Group messages into user-anchored "turns": each turn starts at a user
    # message and runs until (but not including) the next user message.
    turns: List[List[Dict[str, Any]]] = []
    for msg in messages[1:]:
        if msg.get(_SUMMARY_TAG):
            continue  # the rolling-summary message isn't part of any turn
        if msg.get("role") == "user":
            turns.append([msg])
        elif turns:
            turns[-1].append(msg)
        # else: stray message before any user turn - ignore, shouldn't happen

    if len(turns) <= keep_recent_pairs:
        return  # nothing safe to evict yet

    n_to_evict = len(turns) - keep_recent_pairs
    evicted_turns = turns[:n_to_evict]

    new_summaries = []
    for turn in evicted_turns:
        user_msg = turn[0]
        rest = turn[1:]
        new_summaries.append(_heuristic_summarize_pair(user_msg, rest))

    # Find or create the rolling-summary message (kept right after system prompt).
    summary_msg = next((m for m in messages if m.get(_SUMMARY_TAG)), None)
    if summary_msg is None:
        summary_msg = {
            "role": "system",
            "content": "Summary of earlier conversation (older turns were dropped to save context):\n",
            _SUMMARY_TAG: True,
        }
        messages.insert(1, summary_msg)

    summary_msg["content"] = summary_msg["content"].rstrip() + "\n" + "\n".join(f"- {s}" for s in new_summaries)

    # Drop the actual evicted messages from the live list.
    evicted_ids = {id(m) for turn in evicted_turns for m in turn}
    messages[:] = [m for m in messages if id(m) not in evicted_ids]

    logger.info(
        f"Sliding window: evicted {n_to_evict} old turn(s) into the rolling summary "
        f"(context was at {total_tokens(messages)} tokens after eviction)."
    )