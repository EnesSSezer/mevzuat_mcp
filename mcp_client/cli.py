"""
mcp_client/cli.py

Interactive terminal chatbox for testing the MCP server end-to-end against
the TÜBİTAK Qwen3-Coder-80B endpoint over stdio - now with a live, colored
stream of what the agent is doing (which tool it called, with what args,
whether it succeeded/failed/served a stale cache, etc.) instead of just a
final answer appearing all at once.

Usage:
    python -m mcp_client.cli
    MCP_SERVER_COMMAND="python mevzuat_mcp_server.py" python -m mcp_client.cli
"""
import asyncio
import json
import logging

from .agent import Agent
from .config import ClientConfig
from .events import AgentEvent, EventType

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("mcp_client.cli")

# --- minimal ANSI color helpers (no extra dependency) ---
RESET = "\033[0m"
DIM = "\033[2m"
GREEN = "\033[32m"
BOLD_GREEN = "\033[1;32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"


def _short_args(arguments) -> str:
    if not arguments:
        return ""
    try:
        s = json.dumps(arguments, ensure_ascii=False)
    except Exception:
        s = str(arguments)
    return s if len(s) <= 100 else s[:97] + "..."


async def render_event(event: AgentEvent) -> None:
    if event.type == EventType.LLM_THINKING:
        print(f"{DIM}… agent is thinking{RESET}")

    elif event.type == EventType.TOOL_CALL_STARTED:
        args = _short_args(event.arguments)
        print(f"{BOLD_GREEN}▶ tool called:{RESET} {GREEN}{event.tool_name}{RESET}"
              f"{DIM}({args}){RESET}")

    elif event.type == EventType.TOOL_CALL_REJECTED:
        print(f"{RED}✗ rejected before sending:{RESET} {event.tool_name} - {event.detail}")

    elif event.type == EventType.TOOL_CALL_SUCCEEDED:
        print(f"{GREEN}✓ {event.tool_name} succeeded{RESET} {DIM}({event.duration_s:.1f}s){RESET}")

    elif event.type == EventType.TOOL_CALL_STALE:
        print(f"{YELLOW}⚠ {event.tool_name} failed live, served cached result{RESET} {DIM}({event.detail}){RESET}")

    elif event.type == EventType.TOOL_CALL_FAILED:
        print(f"{RED}✗ {event.tool_name} failed:{RESET} {event.detail}")

    elif event.type == EventType.LOOP_GUARD_TRIGGERED:
        if event.extra.get("severity") == "soft":
            print(f"{YELLOW}⟳ auto-nudge: {event.detail}{RESET} {DIM}(letting the agent try to self-correct){RESET}")
        else:
            print(f"{CYAN}⟳ loop guard: {event.detail}{RESET}")

    elif event.type == EventType.FINAL_ANSWER:
        pass  # printed by main() itself once run_turn() returns


async def confirm_via_terminal(reason: str) -> bool:
    print(f"\n{CYAN}[loop guard]{RESET} {reason}")
    answer = await asyncio.to_thread(input, "Keep going anyway? [y/N]: ")
    return answer.strip().lower() in ("y", "yes")


async def main() -> None:
    config = ClientConfig()
    agent = Agent(config, confirm_callback=confirm_via_terminal, on_event=render_event)

    print(f"Connecting to MCP server: {config.server.command!r} ...")
    await agent.start()
    print(f"Connected. {len(agent.transport.tools)} tools available. Type 'exit' to quit.\n")

    try:
        while True:
            user_input = await asyncio.to_thread(input, "you> ")
            if user_input.strip().lower() in ("exit", "quit"):
                break
            if not user_input.strip():
                continue

            print()  # blank line before the step stream starts
            answer = await agent.run_turn(user_input)
            print(f"\nassistant> {answer}\n")
    finally:
        await agent.shutdown()


if __name__ == "__main__":
    asyncio.run(main())