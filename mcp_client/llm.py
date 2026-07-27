"""
mcp_client/llm.py

Wraps chat completions endpoints (OpenAI-compatible) with automatic fallback
from native tool calling to prompt-based tool calling for models/endpoints
(such as vLLM deployments of GPT-OSS-120B) whose native function-calling
parser is broken or unsupported for that specific model.
"""
import json
import logging
import os
import re
import uuid
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

from .config import LLMConfig

logger = logging.getLogger(__name__)


def parse_prompt_tool_calls(text: str) -> List[Dict[str, Any]]:
    """
    Extract tool calls from LLM text content if the LLM outputted tool calls
    as JSON blocks or XML-like tags (e.g. ```json {"tool": "...", "arguments": {...}} ```
    or {"name": "...", "arguments": {...}} or {"tool": ...}).
    """
    if not text:
        return []
    tool_calls = []

    json_blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    for block in json_blocks:
        try:
            data = json.loads(block)
            name = data.get("tool") or data.get("name") or data.get("function")
            args = data.get("arguments") or data.get("args") or {}
            if name:
                args_str = json.dumps(args) if isinstance(args, dict) else str(args)
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {"name": name, "arguments": args_str}
                })
        except Exception:
            pass

    if tool_calls:
        return tool_calls

    raw_matches = re.findall(r"(\{\s*\"(?:tool|name|function)\"\s*:\s*\"[^\"]+\"\s*,\s*\"(?:arguments|args)\"\s*:\s*\{.*?\}\s*\})", text, re.DOTALL)
    for raw in raw_matches:
        try:
            data = json.loads(raw)
            name = data.get("tool") or data.get("name") or data.get("function")
            args = data.get("arguments") or data.get("args") or {}
            if name:
                args_str = json.dumps(args) if isinstance(args, dict) else str(args)
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {"name": name, "arguments": args_str}
                })
        except Exception:
            pass

    return tool_calls


def format_tools_system_prompt(tools: List[Dict[str, Any]], must_use_tool: bool = False) -> str:
    """
    Format tools into a text prompt for models without working native tool
    calling. Emits the FULL JSON Schema per tool (not just param names/types)
    so fallback mode doesn't lose the hint text we've built into the schemas
    (parameter descriptions, defaults, guidance on what to do on 0 results,
    etc.) - those live in `description` fields inside the schema, and a
    thinner text format would silently discard all of it.

    Phrasing is intentionally conditional ("if a tool would help") rather
    than "you MUST respond with JSON" - once fallback mode triggers it stays
    on for the rest of the session (see LLMClient._use_fallback), including
    for turns that don't need any tool at all (a plain "merhaba" shouldn't
    be forced into pretending it needs a tool call).
    """
    lines = [
        "\nAVAILABLE MCP TOOLS:",
        "If calling a tool would help answer the user, respond with ONLY a JSON block "
        "in this exact format (no other text before or after it):",
        "```json",
        '{"tool": "tool_name", "arguments": {"param1": "value1"}}',
        "```",
    ]
    if must_use_tool:
        lines.append("CRITICAL: You MUST call one of the available MCP tools. Do NOT answer directly from memory. Output ONLY a JSON block with your tool call.")
    else:
        lines.append("If no tool is needed, just answer normally in plain text - do not force a tool call.")
        
    lines.extend([
        "",
        "Tool schemas (JSON Schema format - required fields, types, and guidance are in here):",
    ])
    for t in tools:
        fn = t.get("function", {})
        name = fn.get("name")
        desc = fn.get("description", "")
        params = fn.get("parameters", {"type": "object", "properties": {}})
        lines.append(f"\n### {name}")
        if desc:
            lines.append(desc.strip())
        lines.append("Parameters (JSON Schema):")
        lines.append("```json")
        lines.append(json.dumps(params, ensure_ascii=False, indent=2))
        lines.append("```")
    return "\n".join(lines)


class LLMClient:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = AsyncOpenAI(base_url=config.base_url, api_key=config.api_key)
        # Skip the wasted first native attempt entirely for a model already
        # known (from prior testing) to need prompt-based tools. Doesn't
        # require any config.py changes - just an env var, since this is a
        # per-deployment operational fact, not a design-level setting.
        self._use_fallback = os.getenv("LLM_FORCE_PROMPT_BASED_TOOLS", "false").strip().lower() in ("1", "true", "yes")
        if self._use_fallback:
            logger.info(f"LLM_FORCE_PROMPT_BASED_TOOLS set - skipping native tool-calling attempts for '{config.model}'.")

    async def chat(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]], tool_choice: Optional[Any] = None) -> Dict[str, Any]:
        """
        Sends one chat completion request. Automatically falls back to prompt-based
        tool calling if the endpoint fails or returns empty response for native tools.
        """
        if not tools:
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=self.config.temperature,
                timeout=self.config.request_timeout_s,
            )
            msg = response.choices[0].message
            return {"role": "assistant", "content": msg.content}

        if self._use_fallback:
            return await self._chat_prompt_based(messages, tools, must_use_tool=(tool_choice == "required"))

        try:
            kwargs = {
                "model": self.config.model,
                "messages": messages,
                "tools": tools,
                "temperature": self.config.temperature,
                "timeout": self.config.request_timeout_s,
            }
            if tool_choice is not None:
                kwargs["tool_choice"] = tool_choice
                
            response = await self.client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            message = choice.message

            result: Dict[str, Any] = {"role": "assistant", "content": message.content}
            if message.tool_calls:
                result["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in message.tool_calls
                ]
                return result

            if message.content and message.content.strip():
                parsed = parse_prompt_tool_calls(message.content)
                if parsed:
                    result["tool_calls"] = parsed
                return result

            # Both tool_calls and content are empty/None - native tool calling
            # isn't working for this model/endpoint. Log finish_reason too,
            # so a future "why did this trigger" question has real evidence
            # instead of having to guess after the fact.
            logger.warning(
                f"Endpoint '{self.config.model}' returned empty content and tool_calls with native "
                f"tools payload (finish_reason={choice.finish_reason!r}). "
                "Switching to prompt-based tool calling fallback for the rest of this session."
            )
            self._use_fallback = True
            return await self._chat_prompt_based(messages, tools, must_use_tool=(tool_choice == "required"))

        except Exception as e:
            if tool_choice == "required":
                logger.warning(f"Native tool call failed with tool_choice='required' ({e}). Retrying without tool_choice...")
                try:
                    kwargs.pop("tool_choice", None)
                    response = await self.client.chat.completions.create(**kwargs)
                    choice = response.choices[0]
                    message = choice.message
                    
                    result: Dict[str, Any] = {"role": "assistant", "content": message.content}
                    if message.tool_calls:
                        result["tool_calls"] = [{"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}} for tc in message.tool_calls]
                        return result
                    
                    if message.content and message.content.strip():
                        parsed = parse_prompt_tool_calls(message.content)
                        if parsed:
                            result["tool_calls"] = parsed
                        return result
                        
                except Exception as retry_e:
                    logger.warning(f"Retry without tool_choice also failed ({retry_e}).")

            logger.warning(f"Native tool call request failed ({e}). Switching to prompt-based tool calling fallback.")
            self._use_fallback = True
            return await self._chat_prompt_based(messages, tools, must_use_tool=(tool_choice == "required"))

    async def _chat_prompt_based(self, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]], must_use_tool: bool = False) -> Dict[str, Any]:
        """Executes tool calling by embedding full tool schemas in the system prompt."""
        fallback_messages = [dict(m) for m in messages]
        tools_prompt = format_tools_system_prompt(tools, must_use_tool=must_use_tool)

        if fallback_messages and fallback_messages[0]["role"] == "system":
            fallback_messages[0]["content"] += "\n" + tools_prompt
        else:
            fallback_messages.insert(0, {"role": "system", "content": tools_prompt})

        response = await self.client.chat.completions.create(
            model=self.config.model,
            messages=fallback_messages,
            temperature=self.config.temperature,
            timeout=self.config.request_timeout_s,
        )
        msg = response.choices[0].message
        content = msg.content or ""

        parsed_tools = parse_prompt_tool_calls(content)
        result: Dict[str, Any] = {"role": "assistant", "content": content}
        if parsed_tools:
            result["tool_calls"] = parsed_tools
        return result