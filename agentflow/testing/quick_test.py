"""QuickTest - Simplified testing for Agentflow agents.

Reduces test boilerplate from ~20 lines to ~3 lines with preset test scenarios.
"""

import logging
from typing import Any

from agentflow.graph import StateGraph
from agentflow.state import Message
from agentflow.utils.constants import END

from .test_agent import TestAgent
from .test_result import TestResult


logger = logging.getLogger("agentflow.testing")


class QuickTest:
    """Simplified testing interface for common test patterns.

    Provides one-liner test methods for:
    - Single-turn Q&A
    - Multi-turn conversations
    - Agent with tool calls
    - Custom scenarios

    Example:
        ```python
        # Single turn test
        result = await QuickTest.single_turn(agent_response="Hello!", user_message="Hi there")
        result.assert_contains("Hello!")

        # Multi-turn test
        result = await QuickTest.multi_turn(
            [
                ("Hello", "Hi!"),
                ("How are you?", "Great!"),
            ]
        )

        # With tools
        result = await QuickTest.with_tools(
            query="Weather in NYC?",
            response="It's sunny!",
            tools=["get_weather"],
        )
        result.assert_tool_called("get_weather")
        ```
    """

    @classmethod
    async def single_turn(
        cls,
        agent_response: str,
        user_message: str = "Hello",
        model: str = "test-model",
        config: dict[str, Any] | None = None,
    ) -> TestResult:
        """Test a single user-agent interaction.

        Args:
            agent_response: What the agent should respond
            user_message: User's input message
            model: Model identifier for compatibility
            config: Optional config for graph execution

        Returns:
            TestResult with assertions
        """
        # Create test agent
        agent = TestAgent(model=model, responses=[agent_response])

        # Build simple graph
        graph = StateGraph()
        graph.add_node("MAIN", agent)
        graph.set_entry_point("MAIN")
        graph.add_edge("MAIN", END)

        # Compile and run
        compiled = graph.compile()
        result = await compiled.ainvoke(
            {"messages": [Message.text_message(user_message)]},
            config=config or {},
        )

        # Extract response
        final_response = cls._extract_response(result)

        return TestResult(
            final_response=final_response,
            messages=result.get("messages", []),
            tool_calls=[],
            state=result,
        )

    @classmethod
    async def multi_turn(
        cls,
        conversation: list[tuple[str, str]],
        model: str = "test-model",
        config: dict[str, Any] | None = None,
    ) -> TestResult:
        """Test a multi-turn conversation.

        Args:
            conversation: List of (user_message, agent_response) tuples
            model: Model identifier
            config: Optional config

        Returns:
            TestResult with all messages
        """
        # Extract responses
        responses = [response for _, response in conversation]

        # Create test agent
        agent = TestAgent(model=model, responses=responses)

        # Build graph
        graph = StateGraph()
        graph.add_node("MAIN", agent)
        graph.set_entry_point("MAIN")
        graph.add_edge("MAIN", END)

        compiled = graph.compile()

        result = {}

        # Run each turn
        messages = []
        for user_msg, _ in conversation:
            messages.append(Message.text_message(user_msg, role="user"))

            result = await compiled.ainvoke(
                {"messages": messages},
                config=config or {},
            )

            messages = result.get("messages", [])

        # Extract final response
        final_response = cls._extract_response(result)

        return TestResult(
            final_response=final_response,
            messages=messages,
            tool_calls=[],
            state=result,
        )

    @classmethod
    async def with_tools(
        cls,
        query: str,
        response: str,
        tools: list[str | Any],
        tool_responses: dict[str, str] | None = None,
        model: str = "test-model",
        config: dict[str, Any] | None = None,
    ) -> TestResult:
        """Test agent with tool calls.

        Args:
            query: User query
            response: Agent's final response
            tools: List of tool names or functions
            tool_responses: Dict mapping tool names to their responses
            model: Model identifier
            config: Optional config

        Returns:
            TestResult with tool call tracking
        """
        from agentflow.graph import ToolNode

        # Create mock tools
        tool_funcs = []
        tool_calls = []

        for tool in tools:
            if isinstance(tool, str):
                # Create mock tool function
                def make_tool(name: str):
                    def tool_func(query: str = "", **kwargs: Any) -> str:
                        tool_calls.append(
                            {
                                "name": name,
                                "args": {"query": query, **kwargs},
                            }
                        )
                        if tool_responses and name in tool_responses:
                            return tool_responses[name]
                        return f"Mock result from {name}"

                    tool_func.__name__ = name
                    return tool_func

                tool_funcs.append(make_tool(tool))
            else:
                tool_funcs.append(tool)

        # Create tool node
        tool_node = ToolNode(tool_funcs)

        # Create test agent with tool calls in response
        agent = TestAgent(model=model, responses=[response], tools=tool_funcs)

        # Build graph with tool routing
        graph = StateGraph()
        graph.add_node("MAIN", agent)
        graph.add_node("TOOL", tool_node)
        graph.set_entry_point("MAIN")

        # Simple routing: MAIN -> END (tools handled internally by TestAgent)
        graph.add_edge("MAIN", END)

        compiled = graph.compile()

        result = await compiled.ainvoke(
            {"messages": [Message.text_message(query)]},
            config=config or {},
        )

        final_response = cls._extract_response(result)

        return TestResult(
            final_response=final_response,
            messages=result.get("messages", []),
            tool_calls=tool_calls,
            state=result,
        )

    @classmethod
    async def custom(
        cls,
        agent: Any,
        user_message: str,
        graph_setup: Any = None,
        config: dict[str, Any] | None = None,
    ) -> TestResult:
        """Test with custom agent and graph setup.

        Args:
            agent: Custom agent instance (TestAgent or real Agent)
            user_message: User's message
            graph_setup: Optional callable to customize graph
            config: Optional config

        Returns:
            TestResult
        """
        graph = StateGraph()
        graph.add_node("MAIN", agent)
        graph.set_entry_point("MAIN")
        graph.add_edge("MAIN", END)

        # Allow custom graph modifications
        if graph_setup:
            graph = graph_setup(graph)

        compiled = graph.compile()
        result = await compiled.ainvoke(
            {"messages": [Message.text_message(user_message)]},
            config=config or {},
        )

        final_response = cls._extract_response(result)

        return TestResult(
            final_response=final_response,
            messages=result.get("messages", []),
            tool_calls=[],
            state=result,
        )

    @staticmethod
    def _extract_response(result: dict[str, Any]) -> str:
        """Extract final response from graph result."""
        messages = result.get("messages", [])
        if not messages:
            return ""

        # Get last assistant message
        for msg in reversed(messages):
            if hasattr(msg, "role") and msg.role == "assistant":
                if hasattr(msg, "text"):
                    return msg.text()
                return str(msg.content) if hasattr(msg, "content") else str(msg)

        return ""
