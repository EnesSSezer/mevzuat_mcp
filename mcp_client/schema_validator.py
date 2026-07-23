"""
mcp_client/schema_validator.py

Feature 1: compile the JSON Schemas the server hands back from `tools/list`,
and validate the LLM's arguments *before* they ever reach the transport.
A bad call never leaves the process - the model just gets a clean error
message back on the next turn and self-corrects.
"""
import logging
from typing import Any, Dict, List, Optional

from jsonschema import Draft7Validator
from jsonschema.validators import validator_for

logger = logging.getLogger(__name__)


class ToolSchemaValidator:
    """Holds one compiled validator per tool name."""

    def __init__(self):
        self._validators: Dict[str, Any] = {}
        self._schemas: Dict[str, dict] = {}

    def register(self, tool_name: str, input_schema: Optional[dict]) -> None:
        schema = input_schema or {"type": "object", "properties": {}}
        try:
            cls = validator_for(schema) or Draft7Validator
            cls.check_schema(schema)
            self._validators[tool_name] = cls(schema)
        except Exception as e:
            # A malformed schema from the server shouldn't crash client startup -
            # just skip validation for that one tool and log loudly.
            logger.warning(f"Could not compile schema for tool '{tool_name}': {e}")
            self._validators[tool_name] = None
        self._schemas[tool_name] = schema

    def validate(self, tool_name: str, arguments: Dict[str, Any]) -> Optional[str]:
        """
        Returns None if arguments are valid (or unvalidatable), otherwise a
        short, clean error string suitable for feeding straight back to the LLM.
        """
        validator = self._validators.get(tool_name)
        if validator is None:
            return None  # unknown tool or uncompilable schema - let the server reject it

        errors = sorted(validator.iter_errors(arguments or {}), key=lambda e: list(e.path))
        if not errors:
            return None

        # Keep it to the first couple of errors - a wall of jsonschema output
        # is worse for a model to parse than one clear sentence.
        messages = []
        for err in errors[:3]:
            if err.validator == "required":
                # err.message is already like "'repo_name' is a required property"
                missing = err.message.split("'")[1] if "'" in err.message else err.message
                messages.append(f"Missing required property '{missing}'")
            else:
                path = ".".join(str(p) for p in err.path) or "(root)"
                messages.append(f"Invalid value for '{path}': {err.message}")

        return "; ".join(messages)

    def known_tools(self) -> List[str]:
        return list(self._schemas.keys())
