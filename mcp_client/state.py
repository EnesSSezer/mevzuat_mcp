"""
mcp_client/state.py

Feature 2: conversational state with rollback. In-memory only (no disk persistence implemented).

The rollback point is the *start of a user turn*. If something goes wrong
mid-workflow badly enough that the conversation can't be left in a coherent
state (e.g. an unexpected exception while orchestrating tool calls - NOT a
routine single tool failure, which is handled as a normal tool-error message
so the model can react to it), we roll back to exactly how things looked
before the user's message was processed, and report a clean error - so the
user can just retry without a corrupted transcript sitting in context.
"""
import copy
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class ConversationState:
    def __init__(self, system_prompt: str):
        self.messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]
        self._checkpoint: List[Dict[str, Any]] | None = None

    def add_user_message(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_message(self, message: Dict[str, Any]) -> None:
        self.messages.append(message)

    def checkpoint(self) -> None:
        """Snapshot the current state. Call this once, right before starting to process a new user turn."""
        self._checkpoint = copy.deepcopy(self.messages)

    def rollback(self) -> None:
        """Restore to the last checkpoint, discarding everything added since."""
        if self._checkpoint is None:
            logger.warning("rollback() called with no checkpoint set - nothing to restore.")
            return
        self.messages = copy.deepcopy(self._checkpoint)
        logger.info("Conversation state rolled back to last checkpoint.")

    def commit(self) -> None:
        """Turn completed cleanly - checkpoint no longer needed until the next turn."""
        self._checkpoint = None
