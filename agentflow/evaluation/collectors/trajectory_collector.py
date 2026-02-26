"""
Trajectory collector for capturing execution paths during agent runs.

This module provides classes for collecting the execution trajectory from
EventModel events during graph execution. It hooks into graph.compile()
via a CallbackManager rather than requiring changes at construction time.

Key classes:
    NodeResponse        — per-node input/output snapshot (DeepEval-style span)
    PublisherCallback   — wraps a BasePublisher as an AfterInvokeCallback
    TrajectoryCollector — collects trajectory via on_event(EventModel)
    EventCollector      — stores all raw events for debugging
    make_trajectory_callback — helper to wire everything into a CallbackManager

Example:
    collector = TrajectoryCollector()
    _, callback_mgr = make_trajectory_callback(collector, config={"thread_id": "run-1"})

    compiled = graph.compile(callback_manager=callback_mgr)
    await compiled.ainvoke(state, config)

    print(collector.tool_calls)
    print(collector.node_visits)
    print(f"duration: {collector.duration:.3f}s")
"""

import logging
import time
from dataclasses import dataclass
from dataclasses import field as datafield
from typing import Any

from agentflow.evaluation.dataset.eval_set import StepType, ToolCall, TrajectoryStep
from agentflow.publisher.base_publisher import BasePublisher
from agentflow.publisher.events import ContentType, Event, EventModel, EventType
from agentflow.utils.callbacks import (
    AfterInvokeCallback,
    CallbackContext,
    CallbackManager,
    InvocationType,
)


logger = logging.getLogger("agentflow.evaluation.collectors")


@dataclass
class NodeResponse:
    """Snapshot of input and output at a single AI node invocation.

    Mirrors DeepEval's @observe + update_current_span(output=...) pattern so
    evaluation tests can assert on intermediate LLM reasoning, not just the
    final answer.

    input_messages is extracted from input_data["state"].context BEFORE
    process_node_result updates the state (callback fires at line 342,
    process_node_result runs at line 345 in invoke_node_handler.py).

    response_text is extracted via output_data.invoke() because the LLM
    response is NOT yet written to state at callback time.

    Attributes:
        node_name:       Name of the graph node that ran (e.g. "MAIN").
        input_messages:  Conversation history going into this node, as
                         list of {"role": ..., "content": ...} dicts.
        response_text:   LLM text output; empty when this is a tool-call turn.
        has_tool_calls:  True when the LLM decided to call a tool this turn.
        tool_call_names: Names of tools requested (usually one per turn).
        is_final:        True when no further tool calls — last text turn.
        timestamp:       Wall-clock time when this node invocation completed.
    """

    node_name: str
    input_messages: list[dict] = datafield(default_factory=list)
    response_text: str = ""
    has_tool_calls: bool = False
    tool_call_names: list[str] = datafield(default_factory=list)
    is_final: bool = False
    timestamp: float = 0.0


class PublisherCallback(AfterInvokeCallback):
    """Wraps a BasePublisher as an AfterInvokeCallback for graph execution.

    Builds an EventModel from each callback invocation and calls
    publisher.publish() directly (awaited), avoiding the race condition
    of the internal publish_event() which fires as a background task.

    Registered for TOOL, MCP, and AI invocation types so tool calls and
    AI node visits are both captured.

    Attributes:
        _publisher: The BasePublisher that receives each EventModel.
        _config:    Optional thread_id / run_id stamped on every EventModel.

    Example:
        ```python
        collector = TrajectoryCollector()
        cb = PublisherCallback(collector, config={"thread_id": "run-1"})

        mgr = CallbackManager()
        mgr.register_after_invoke(InvocationType.AI, cb)
        mgr.register_after_invoke(InvocationType.TOOL, cb)
        ```
    """

    def __init__(self, publisher: BasePublisher, config: dict | None = None):
        """Initialize the callback with a publisher and optional config.

        Args:
            publisher: BasePublisher instance that receives published events.
            config:    Optional dict with thread_id / run_id to stamp on events.
        """
        self._publisher = publisher
        self._config = config or {}

    async def __call__(self, context: CallbackContext, input_data: Any, output_data: Any) -> Any:
        """Handle a post-invocation callback from the graph executor.

        For AI nodes, pre-extracts the LLM Message from output_data before
        building the event so input_messages and response_text are available.

        Args:
            context:     Metadata about the invocation (type, node name, etc.).
            input_data:  Raw input passed to the node or tool function.
            output_data: Raw return value from the node or tool function.

        Returns:
            output_data unchanged — the graph executor expects it back.
        """
        node_message = None
        if context.invocation_type == InvocationType.AI:
            node_message = await self._extract_node_message(output_data)
        event = self._build_event(context, input_data, output_data, node_message=node_message)
        if event:
            await self._publisher.publish(event)
        return output_data

    async def _extract_node_message(self, output_data: Any) -> Any:
        """Safely extract a Message from a ModelResponseConverter or result dict.

        Handles two pathways:
        1. ``ModelResponseConverter`` — returned by normal node functions that
           wrap raw LLM responses.  ``.invoke()`` is idempotent when the
           response is already computed (not callable).
        2. ``dict`` with a ``"messages"`` key — returned by
           ``_call_agent_node()`` which consumes the converter before the
           callback fires.  The last Message is the LLM response.

        Returns None on any failure so the callback never raises.

        Args:
            output_data: Return value from an AI node (ModelResponseConverter,
                         or dict with ``"messages"`` list).

        Returns:
            Converted Message object, or None if extraction failed.
        """
        try:
            from agentflow.adapters.llm.model_response_converter import ModelResponseConverter

            if isinstance(output_data, ModelResponseConverter):
                return await output_data.invoke()
        except Exception:
            logger.debug("ModelResponseConverter extraction failed", exc_info=True)

        # Agent-node pathway: _call_agent_node() returns a dict with
        # {"state": ..., "messages": [Message], "next_node": ...}
        # The converter was already consumed, so extract the Message directly.
        try:
            if isinstance(output_data, dict):
                messages = output_data.get("messages")
                if messages and len(messages) > 0:
                    return messages[-1]
        except Exception:
            logger.debug("Agent-node message extraction failed", exc_info=True)

        return None

    def _build_event(
        self,
        context: CallbackContext,
        input_data: Any,
        output_data: Any,
        node_message: Any = None,
    ) -> EventModel | None:
        """Build an EventModel from callback arguments.

        Produces a TOOL_EXECUTION event for tool/MCP invocations and a
        NODE_EXECUTION event for AI node invocations. Returns None for
        unrecognised invocation types.

        Args:
            context:      Callback context with invocation type and node name.
            input_data:   Raw input to the node or tool.
            output_data:  Raw output from the node or tool.
            node_message: Pre-extracted Message from an AI node (may be None).

        Returns:
            A populated EventModel, or None if the invocation type is not
            handled.
        """
        if context.invocation_type in (InvocationType.TOOL, InvocationType.MCP):
            return EventModel(
                event=Event.TOOL_EXECUTION,
                event_type=EventType.END,
                node_name=context.function_name or context.node_name or "",
                data={
                    "function_name": context.function_name,
                    "args": input_data if isinstance(input_data, dict) else {},
                    "result": str(output_data) if output_data is not None else "",
                    "tool_call_id": (context.metadata or {}).get("tool_call_id", ""),
                },
                content_type=[ContentType.TOOL_RESULT],
                thread_id=self._config.get("thread_id", ""),
                run_id=self._config.get("run_id", ""),
                timestamp=time.time(),
            )
        if context.invocation_type == InvocationType.AI:
            # Extract inputs from state (available before node ran)
            state = (
                input_data.get("state", input_data) if isinstance(input_data, dict) else input_data
            )
            input_messages: list[dict] = []
            if hasattr(state, "context") and state.context:
                input_messages = [
                    {"role": m.role, "content": m.text() or ""}
                    for m in state.context
                    if hasattr(m, "role")
                ]

            # Extract output from converted message (not yet written to state)
            response_text = ""
            has_tool_calls = False
            tool_call_names: list[str] = []
            if node_message is not None:
                response_text = node_message.text() or ""
                tc = getattr(node_message, "tools_calls", None) or []
                has_tool_calls = bool(tc)
                # Handle both dict-like and Pydantic ToolCall objects
                for t in tc:
                    if isinstance(t, dict):
                        tool_call_names.append(t.get("name", ""))
                    elif hasattr(t, "name"):
                        tool_call_names.append(t.name)
                    elif hasattr(t, "model_dump"):
                        tool_call_names.append(t.model_dump().get("name", ""))

            return EventModel(
                event=Event.NODE_EXECUTION,
                event_type=EventType.END,
                node_name=context.node_name or "",
                data={
                    "input_messages": input_messages,
                    "response_text": response_text,
                    "has_tool_calls": has_tool_calls,
                    "tool_call_names": tool_call_names,
                    "is_final": not has_tool_calls,
                },
                content_type=[ContentType.MESSAGE],
                thread_id=self._config.get("thread_id", ""),
                run_id=self._config.get("run_id", ""),
                timestamp=time.time(),
            )
        return None


class TrajectoryCollector(BasePublisher):
    """Collects execution trajectory from graph events.

    This class captures the sequence of nodes visited, tools called, and
    LLM outputs produced during agent graph execution. It can be used to
    track the actual execution path for comparison with expected trajectories,
    and captures intermediate LLM reasoning at each node invocation.

    Extends BasePublisher so it can be used as the target for PublisherCallback.
    Preserves the on_event(EventModel) interface for compatibility with the
    previous collector API.

    Attributes:
        trajectory:      Complete execution trajectory as TrajectoryStep objects.
        tool_calls:      List of tool calls made during execution.
        node_visits:     List of node names visited in order.
        node_responses:  Per-node input/output snapshots (one per node visit).
        final_response:  LLM text from the last non-tool-call node invocation.
        events:          All raw EventModel objects captured (when
                         capture_all_events=True), useful for debugging.
        start_time:      When the first event was received.
        end_time:        When the last event was received.

    Example:
        ```python
        collector = TrajectoryCollector()

        # Wire into graph compilation
        _, mgr = make_trajectory_callback(collector, config={"thread_id": "run-1"})
        compiled = graph.compile(callback_manager=mgr)
        await compiled.ainvoke(state, config)

        # Analyse collected trajectory
        print(f"Visited nodes: {collector.node_visits}")
        print(f"Tool calls: {collector.tool_calls}")
        print(f"Final response: {collector.final_response}")
        print(f"Duration: {collector.duration:.3f}s")
        ```
    """

    def __init__(self, capture_all_events: bool = False):
        """Initialize the trajectory collector.

        Args:
            capture_all_events: If True, store all EventModel objects in
                                 self.events for debugging purposes.
        """
        super().__init__(config={})
        self.trajectory: list[TrajectoryStep] = []
        self.tool_calls: list[ToolCall] = []
        self.node_visits: list[str] = []
        self.node_responses: list[NodeResponse] = []
        self.final_response: str = ""
        self.events: list[EventModel] = []
        self.capture_all_events = capture_all_events
        self.start_time: float | None = None
        self.end_time: float | None = None
        self._pending_tool_calls: dict[str, dict] = {}  # call_id -> pending info

    async def publish(self, event: EventModel) -> None:
        """Receive an event from PublisherCallback and delegate to on_event.

        Args:
            event: The EventModel published by PublisherCallback.
        """
        await self.on_event(event)

    async def on_event(self, event: EventModel) -> None:
        """Process an incoming event and update the trajectory.

        This method should be passed as a callback to graph execution, or
        called indirectly via publish() when used with PublisherCallback.

        Args:
            event: The event to process.
        """
        now = time.time()
        if self.start_time is None:
            self.start_time = now
        self.end_time = now

        if self.capture_all_events:
            self.events.append(event)

        if event.event == Event.NODE_EXECUTION:
            self._process_node_event(event)
        elif event.event == Event.TOOL_EXECUTION:
            self._process_tool_event(event)
        elif event.event == Event.GRAPH_EXECUTION:
            self._process_graph_event(event)

    def on_event_sync(self, event: EventModel) -> None:
        """Synchronous version of on_event for non-async contexts.

        Args:
            event: The event to process.
        """
        now = time.time()
        if self.start_time is None:
            self.start_time = now
        self.end_time = now

        if self.capture_all_events:
            self.events.append(event)

        if event.event == Event.NODE_EXECUTION:
            self._process_node_event(event)
        elif event.event == Event.TOOL_EXECUTION:
            self._process_tool_event(event)
        elif event.event == Event.GRAPH_EXECUTION:
            self._process_graph_event(event)

    def _process_node_event(self, event: EventModel) -> None:
        """Process a node execution event and record a NodeResponse."""
        node_name = event.node_name or event.data.get("node_name", "")
        if not node_name:
            logger.debug("Dropping node event with empty node_name: %s", event.event_type)
            return

        self.node_visits.append(node_name)
        self.trajectory.append(
            TrajectoryStep.node(
                name=node_name,
                timestamp=event.timestamp,
                event_type=event.event_type.value
                if hasattr(event.event_type, "value")
                else str(event.event_type),
            )
        )
        response_text = event.data.get("response_text", "")
        has_tool_calls = event.data.get("has_tool_calls", False)
        is_final = event.data.get("is_final", not has_tool_calls)
        nr = NodeResponse(
            node_name=node_name,
            input_messages=event.data.get("input_messages", []),
            response_text=response_text,
            has_tool_calls=has_tool_calls,
            tool_call_names=event.data.get("tool_call_names", []),
            is_final=is_final,
            timestamp=event.timestamp or time.time(),
        )
        self.node_responses.append(nr)
        if is_final and response_text:
            self.final_response = response_text  # last final turn wins

    def _process_tool_event(self, event: EventModel) -> None:
        """Process a tool execution event and record a ToolCall.

        Handles both the PublisherCallback pathway (END-only events with
        args + result in a single event) and the direct on_event pathway
        where START and END are separate events.  START events are parked
        in ``_pending_tool_calls``; the ToolCall is only committed once an
        END event arrives (or if the event has no event_type distinction).
        """
        function_name = event.data.get("function_name", event.node_name or "")
        args = {
            k: v
            for k, v in (event.data.get("args", {})).items()
            if k not in ("tool_call_id", "state")
        }
        tool_call_id = event.data.get("tool_call_id", "")
        result = event.data.get("result", "")

        if not function_name:
            logger.debug("Dropping tool event with empty function_name")
            return

        # --- START / END merging ---
        is_start = getattr(event, "event_type", None) == EventType.START
        is_end = getattr(event, "event_type", None) == EventType.END

        if is_start and tool_call_id:
            # Park args for later merge with END event
            self._pending_tool_calls[tool_call_id] = {
                "name": function_name,
                "args": args,
            }
            return  # don't create ToolCall yet

        # For END events, merge with pending START if available
        if is_end and tool_call_id and tool_call_id in self._pending_tool_calls:
            pending = self._pending_tool_calls.pop(tool_call_id)
            # Prefer START args (usually richer), fall back to END args
            args = pending.get("args", {}) or args
            function_name = pending.get("name", "") or function_name

        tc = ToolCall(
            name=function_name,
            args=args,
            call_id=tool_call_id if tool_call_id else None,
            result=result,  # preserve empty strings as-is
        )
        self.tool_calls.append(tc)
        self.trajectory.append(
            TrajectoryStep.tool(
                name=function_name,
                args=args,
                timestamp=event.timestamp,
                tool_call_id=tool_call_id,
            )
        )

    def _process_graph_event(self, event: EventModel) -> None:
        """Process a graph execution event (marks the end of a run)."""
        if event.event_type == EventType.END:
            self.end_time = time.time()

    def get_tool_names(self) -> list[str]:
        """Get list of tool names called in order."""
        return [tc.name for tc in self.tool_calls]

    def get_trajectory_steps(self, step_type: StepType | None = None) -> list[TrajectoryStep]:
        """Get trajectory steps, optionally filtered by type.

        Args:
            step_type: If provided, only return steps of this type.

        Returns:
            List of trajectory steps matching the filter, or all steps if
            step_type is None.
        """
        if step_type is None:
            return self.trajectory
        return [s for s in self.trajectory if s.step_type == step_type]

    def reset(self) -> None:
        """Reset the collector to initial state."""
        self.trajectory.clear()
        self.tool_calls.clear()
        self.node_visits.clear()
        self.node_responses.clear()
        self.final_response = ""
        self.events.clear()
        self._pending_tool_calls.clear()
        self.start_time = None
        self.end_time = None

    @property
    def duration(self) -> float:
        """Get duration of the collected execution in seconds.

        start_time is set on the first callback received; end_time is updated
        on every subsequent callback. Returns 0.0 if no events were captured.
        """
        if self.start_time is None or self.end_time is None:
            return 0.0
        return self.end_time - self.start_time

    def to_dict(self) -> dict[str, Any]:
        """Convert collector data to a dictionary for serialization."""
        return {
            "trajectory": [s.model_dump() for s in self.trajectory],
            "tool_calls": [tc.model_dump() for tc in self.tool_calls],
            "node_visits": self.node_visits,
            "node_responses": [
                {
                    "node_name": nr.node_name,
                    "input_messages": nr.input_messages,
                    "response_text": nr.response_text,
                    "has_tool_calls": nr.has_tool_calls,
                    "tool_call_names": nr.tool_call_names,
                    "is_final": nr.is_final,
                    "timestamp": nr.timestamp,
                }
                for nr in self.node_responses
            ],
            "final_response": self.final_response,
            "duration_seconds": self.duration,
        }

    async def close(self) -> None:
        """Explicitly mark the end of collection."""
        self.end_time = time.time()

    def sync_close(self) -> None:
        """Synchronous version of close."""
        self.end_time = time.time()

    def __repr__(self) -> str:
        return (
            f"TrajectoryCollector("
            f"nodes={len(self.node_visits)}, "
            f"tools={len(self.tool_calls)}, "
            f"responses={len(self.node_responses)}, "
            f"steps={len(self.trajectory)})"
        )


def make_trajectory_callback(
    collector: TrajectoryCollector,
    config: dict | None = None,
) -> tuple[TrajectoryCollector, CallbackManager]:
    """Wire a TrajectoryCollector into a CallbackManager.

    Registers hooks for TOOL, MCP, and AI invocation types so all tool calls
    and node visits are captured automatically during graph execution.

    Args:
        collector: The TrajectoryCollector that will receive events.
        config:    Optional dict with thread_id / run_id to stamp on each
                   EventModel (useful for correlating events across runs).

    Returns:
        Tuple of (collector, callback_manager). Pass the callback_manager
        to graph.compile().

    Example:
        ```python
        collector = TrajectoryCollector()
        _, mgr = make_trajectory_callback(collector, config={"thread_id": "eval-1"})
        compiled = graph.compile(callback_manager=mgr)
        await compiled.ainvoke(state, config)
        ```
    """
    cb = PublisherCallback(collector, config=config)
    mgr = CallbackManager()
    mgr.register_after_invoke(InvocationType.TOOL, cb)
    mgr.register_after_invoke(InvocationType.MCP, cb)
    mgr.register_after_invoke(InvocationType.AI, cb)
    return collector, mgr


class EventCollector:
    """Simple collector that stores all raw events for debugging and analysis.

    Useful when you need to inspect every EventModel fired during a run,
    including events that TrajectoryCollector doesn't process (e.g.
    GRAPH_EXECUTION start events).

    Attributes:
        events: List of all captured EventModel objects in arrival order.

    Example:
        ```python
        ec = EventCollector()
        # pass ec.on_event as a callback, then inspect:
        node_events = ec.filter_by_event(Event.NODE_EXECUTION)
        end_events = ec.filter_by_event_type(EventType.END)
        ```
    """

    def __init__(self) -> None:
        """Initialize the event collector."""
        self.events: list[EventModel] = []

    def reset(self) -> None:
        """Reset the collector to initial state."""
        self.events.clear()

    async def on_event(self, event: EventModel) -> None:
        """Capture an event.

        Args:
            event: The event to store.
        """
        self.events.append(event)

    def on_event_sync(self, event: EventModel) -> None:
        """Synchronous version of on_event for non-async contexts.

        Args:
            event: The event to store.
        """
        self.events.append(event)

    def filter_by_event(self, event_type: Event) -> list[EventModel]:
        """Filter events by event source type (NODE_EXECUTION, TOOL_EXECUTION, etc.).

        Args:
            event_type: The Event enum value to filter by.

        Returns:
            List of matching EventModel objects.
        """
        return [e for e in self.events if e.event == event_type]

    def filter_by_event_type(self, event_type: EventType) -> list[EventModel]:
        """Filter events by event phase (START, END, etc.).

        Args:
            event_type: The EventType enum value to filter by.

        Returns:
            List of matching EventModel objects.
        """
        return [e for e in self.events if e.event_type == event_type]

    def filter_by_node(self, node_name: str) -> list[EventModel]:
        """Filter events by node name.

        Args:
            node_name: The node name to filter by.

        Returns:
            List of matching EventModel objects.
        """
        return [e for e in self.events if e.node_name == node_name]

    def __len__(self) -> int:
        return len(self.events)

    def __repr__(self) -> str:
        return f"EventCollector(events={len(self.events)})"
