"""ToolNode package.

This package provides a modularized implementation of ToolNode. Public API:

- ToolNode
- HAS_FASTMCP, HAS_MCP

Backwards-compatible import path: ``from agentflow.graph.tool_node import ToolNode``
"""

from agentflow.state.tool_result import ToolResult

from .base import ToolNode
from .deps import HAS_FASTMCP, HAS_MCP


__all__ = ["HAS_FASTMCP", "HAS_MCP", "ToolNode", "ToolResult"]
