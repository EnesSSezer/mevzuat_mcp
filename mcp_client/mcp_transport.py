"""
mcp_client/mcp_transport.py

Owns the stdio connection to the MCP server subprocess. Responsible for:
- initial handshake + tool discovery
- converting MCP Tool schemas into OpenAI-style `tools` specs
- executing tool calls with a timeout, bounded retries, and reconnect if the
  subprocess/pipe dies mid-session
"""
import asyncio
import logging
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import CallToolResult, Tool

from .config import ServerConfig

logger = logging.getLogger(__name__)


class ToolCallError(Exception):
    """Raised when a tool call ultimately fails after retries (no cache fallback available)."""


class MCPTransport:
    def __init__(self, server_config: ServerConfig):
        self.server_config = server_config
        self._stack: Optional[AsyncExitStack] = None
        self.session: Optional[ClientSession] = None
        self.tools: List[Tool] = []

    async def connect(self) -> None:
        if self._stack is not None:
            await self.close()

        self._stack = AsyncExitStack()
        argv = self.server_config.argv
        params = StdioServerParameters(command=argv[0], args=argv[1:])

        read_stream, write_stream = await self._stack.enter_async_context(stdio_client(params))
        self.session = await self._stack.enter_async_context(ClientSession(read_stream, write_stream))
        await self.session.initialize()

        listed = await self.session.list_tools()
        self.tools = listed.tools
        logger.info(f"Connected to MCP server. Discovered {len(self.tools)} tools.")

    async def close(self) -> None:
        if self._stack is not None:
            await self._stack.aclose()
        self._stack = None
        self.session = None

    async def reconnect(self) -> None:
        logger.warning("Reconnecting to MCP server subprocess...")
        await self.connect()

    def openai_tool_specs(self) -> List[Dict[str, Any]]:
        """Convert discovered MCP tools into the `tools=[...]` shape OpenAI-style APIs expect."""
        specs = []
        for tool in self.tools:
            specs.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": tool.inputSchema or {"type": "object", "properties": {}},
                },
            })
        return specs

    @staticmethod
    def _result_to_text(result: CallToolResult) -> str:
        parts = []
        for block in result.content:
            text = getattr(block, "text", None)
            if text is not None:
                parts.append(text)
            else:
                parts.append(str(block))
        joined = "\n".join(parts)
        if result.isError:
            return f"Error: {joined}"
        return joined

    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        timeout_s: float,
        max_retries: int,
        retry_backoff_s: float,
    ) -> str:
        """
        Executes one tool call with a timeout and bounded retries. Raises
        ToolCallError if every attempt fails - callers decide whether to fall
        back to a cached result or surface the error to the model.
        """
        last_exc: Optional[Exception] = None
        for attempt in range(max_retries + 1):
            try:
                assert self.session is not None, "Not connected to MCP server"
                result = await asyncio.wait_for(
                    self.session.call_tool(tool_name, arguments),
                    timeout=timeout_s,
                )
                return self._result_to_text(result)
            except (asyncio.TimeoutError, Exception) as e:
                last_exc = e
                exc_text = str(e)
                if exc_text:
                    detail = f"{type(e).__name__}: {exc_text}"
                else:
                    detail = type(e).__name__
                logger.warning(f"Tool call '{tool_name}' attempt {attempt + 1} failed: {detail}")
                if _looks_like_broken_transport(e):
                    try:
                        await self.reconnect()
                    except Exception as reconnect_err:
                        logger.error(f"Reconnect failed: {reconnect_err}")
                if attempt < max_retries:
                    await asyncio.sleep(retry_backoff_s * (attempt + 1))

        if last_exc is None:
            detail = "unknown error"
        else:
            exc_text = str(last_exc)
            detail = f"{type(last_exc).__name__}: {exc_text}" if exc_text else type(last_exc).__name__

        raise ToolCallError(f"Tool '{tool_name}' failed after {max_retries + 1} attempts: {detail}")


def _looks_like_broken_transport(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(s in text for s in ("broken pipe", "connection", "closed", "eof", "reset"))
