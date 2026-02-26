"""
Trajectory and event collectors for agent evaluation.

Hooks into graph execution via graph.compile(callback_manager=...) to capture
the execution path — tool calls, node visits, and timing — without requiring
a publisher wired at StateGraph construction time.

Example:
    ```python
    from agentflow.evaluation.collectors import TrajectoryCollector, make_trajectory_callback

    collector = TrajectoryCollector()
    _, mgr = make_trajectory_callback(collector, config={"thread_id": "eval-1"})

    compiled = graph.compile(callback_manager=mgr)
    await compiled.ainvoke(state, config)

    print(collector.tool_calls)  # [ToolCall(name="get_weather", ...)]
    print(collector.node_visits)  # ["PLANNER", "ANALYST"]
    print(collector.duration)  # 3.42
    ```
"""

from .trajectory_collector import (
    EventCollector,
    PublisherCallback,
    TrajectoryCollector,
    make_trajectory_callback,
)


__all__ = [
    "TrajectoryCollector",
    "EventCollector",
    "PublisherCallback",
    "make_trajectory_callback",
]
