"""ToolResult: return type for tool functions that need to update state and return a message."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    """Return type for tool functions that need to update state AND return a message to the AI.

    When a tool needs to both mutate graph state fields *and* communicate something back
    to the language model, return a ``ToolResult`` instead of a plain string.

    Attributes:
        message: The text response to return to the AI (used as the tool result content).
        state: Optional dict mapping state field names to their new values.
            Only fields present in the dict are updated; other fields are left unchanged.

    Example::

        class MyState(AgentState):
            jd_name: str = ""


        def update_context(state: MyState, jd_name: str) -> ToolResult:
            return ToolResult(
                message=f"JD name updated to '{jd_name}'",
                state={"jd_name": jd_name},
            )
    """

    message: Any
    state: dict[str, Any] | None = field(default=None)
