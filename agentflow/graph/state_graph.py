import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar, Union

from injectq import InjectQ

from agentflow.checkpointer import BaseCheckpointer
from agentflow.exceptions import GraphError
from agentflow.publisher import BasePublisher
from agentflow.state import AgentState, BaseContextManager
from agentflow.store import BaseStore
from agentflow.utils import END, START, CallbackManager
from agentflow.utils.background_task_manager import BackgroundTaskManager
from agentflow.utils.id_generator import BaseIDGenerator, DefaultIDGenerator

from .agent import Agent
from .base_agent import BaseAgent
from .edge import Edge
from .node import Node
from .tool_node import ToolNode


if TYPE_CHECKING:
    from .compiled_graph import CompiledGraph


# Generic type variable bound to AgentState for state
StateT = TypeVar("StateT", bound=AgentState)

logger = logging.getLogger("agentflow.graph")


class StateGraph[StateT: AgentState]:
    """Main graph class for orchestrating multi-agent workflows.

    This class provides the core functionality for building and managing stateful
    agent workflows. It is similar to LangGraph's StateGraph
    integration with support for dependency injection.

    The graph is generic over state types to support custom AgentState subclasses,
    allowing for type-safe state management throughout the workflow execution.

    Attributes:
        state (StateT): The current state of the graph workflow.
        nodes (dict[str, Node]): Collection of nodes in the graph.
        edges (list[Edge]): Collection of edges connecting nodes.
        entry_point (str | None): Name of the starting node for execution.
        context_manager (BaseContextManager[StateT] | None): Optional context manager
            for handling cross-node state operations.
        dependency_container (DependencyContainer): Container for managing
            dependencies that can be injected into node functions.
        compiled (bool): Whether the graph has been compiled for execution.

    Example:
        >>> graph = StateGraph()
        >>> graph.add_node("process", process_function)
        >>> graph.add_edge(START, "process")
        >>> graph.add_edge("process", END)
        >>> compiled = graph.compile()
        >>> result = compiled.invoke({"input": "data"})
    """

    def __init__(
        self,
        state: StateT | None = None,
        context_manager: BaseContextManager[StateT] | None = None,
        publisher: BasePublisher | None = None,
        id_generator: BaseIDGenerator = DefaultIDGenerator(),
        container: InjectQ | None = None,
    ):
        """Initialize a new StateGraph instance.

        Args:
            state: Initial state for the graph. If None, a default AgentState
                will be created.
            context_manager: Optional context manager for handling cross-node
                state operations and advanced state management patterns.
            dependency_container: Container for managing dependencies that can
                be injected into node functions. If None, a new empty container
                will be created.
            publisher: Publisher for emitting events during execution

        Note:
            START and END nodes are automatically added to the graph upon
            initialization and accept the full node signature including
            dependencies.

        Example:
            # Basic usage with default AgentState
            >>> graph = StateGraph()

            # With custom state
            >>> custom_state = MyCustomState()
            >>> graph = StateGraph(custom_state)

            # Or using type hints for clarity
            >>> graph = StateGraph[MyCustomState](MyCustomState())
        """
        logger.info("Initializing StateGraph")
        logger.debug(
            "StateGraph init with state=%s, context_manager=%s",
            type(state).__name__ if state else "default AgentState",
            type(context_manager).__name__ if context_manager else None,
        )

        # State handling – accept both a class (e.g. AgentState) and an instance
        if state is not None:
            if isinstance(state, type) and issubclass(state, AgentState):
                self._state: StateT = state()  # type: ignore[assignment]
            else:
                self._state: StateT = state  # type: ignore[assignment]
        else:
            self._state: StateT = AgentState()  # type: ignore[assignment]

        # Graph structure
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []
        self.entry_point: str | None = None

        # Services
        self._publisher: BasePublisher | None = publisher
        self._id_generator: BaseIDGenerator = id_generator
        self._context_manager: BaseContextManager[StateT] | None = context_manager
        # save container for dependency injection
        # if any container is passed then we will activate that
        # otherwise we can skip it and use the default one
        if container is None:
            self._container = InjectQ.get_instance()
            logger.debug("No container provided, using global singleton instance")
        else:
            logger.debug("Using provided dependency container instance")
            self._container = container
            self._container.activate()

        # Register task_manager, for async tasks
        # This will be used to run background tasks
        self._task_manager = BackgroundTaskManager()

        # now setup the graph
        self._setup()

        # Add START and END nodes (accept full node signature including dependencies)
        logger.debug("Adding default START and END nodes")
        self.nodes[START] = Node(START, lambda state, config, **deps: state, self._publisher)  # type: ignore
        self.nodes[END] = Node(END, lambda state, config, **deps: state, self._publisher)
        logger.debug("StateGraph initialized with %d nodes", len(self.nodes))

    def _setup(self):
        """Setup the graph before compilation.

        This method can be used to perform any necessary setup or validation
        before compiling the graph for execution.
        """
        logger.info("Setting up StateGraph before compilation")
        # Placeholder for any setup logic needed before compilation
        # register dependencies

        # register state and context manager as singletons (these are nullable)
        self._container.bind_instance(
            BaseContextManager,
            self._context_manager,
            allow_none=True,
            allow_concrete=True,
        )
        self._container.bind_instance(
            BasePublisher,
            self._publisher,
            allow_none=True,
            allow_concrete=True,
        )

        # register id generator as factory
        self._container.bind_instance(
            BaseIDGenerator,
            self._id_generator,
            allow_concrete=True,
        )
        self._container.bind("generated_id_type", self._id_generator.id_type)
        # Allow async method also
        self._container.bind_factory(
            "generated_id",
            lambda: self._id_generator.generate(),
        )

        # Save BackgroundTaskManager
        self._container.bind_instance(
            BackgroundTaskManager,
            self._task_manager,
            allow_concrete=False,
        )

    def add_node(
        self,
        name_or_func: str | Callable,
        func: Union[Callable, "ToolNode", "Agent", None] = None,
    ) -> "StateGraph":
        """Add a node to the graph.

        This method supports multiple calling patterns:
        1. Pass a callable as the first argument (name inferred from function name)
        2. Pass a name string and callable/ToolNode/Agent as separate arguments
        3. Pass an Agent instance directly as the second argument

        Args:
            name_or_func: Either the node name (str) or a callable function.
                If callable, the function name will be used as the node name.
            func: The function, ToolNode, or Agent to execute. Required if name_or_func
                is a string, ignored if name_or_func is callable.

        Returns:
            StateGraph: The graph instance for method chaining.

        Raises:
            ValueError: If invalid arguments are provided.

        Example:
            >>> # Method 1: Function name inferred
            >>> graph.add_node(my_function)
            >>> # Method 2: Explicit name and function
            >>> graph.add_node("process", my_function)
            >>> # Method 3: Agent instance
            >>> graph.add_node("agent", agent_instance)
        """
        if callable(name_or_func) and func is None:
            # Function passed as first argument
            name = name_or_func.__name__
            func = name_or_func
            logger.debug("Adding node '%s' with inferred name from function", name)
        elif isinstance(name_or_func, str) and (
            callable(func) or isinstance(func, ToolNode | BaseAgent)
        ):
            # Name and function/ToolNode/Agent passed separately
            name = name_or_func
            node_type = (
                "Agent"
                if isinstance(func, BaseAgent)
                else "ToolNode"
                if isinstance(func, ToolNode)
                else "callable"
            )
            logger.debug(
                "Adding node '%s' with explicit name and %s",
                name,
                node_type,
            )
        else:
            error_msg = "Invalid arguments for add_node"
            logger.error(error_msg)
            raise ValueError(error_msg)

        self.nodes[name] = Node(name, func)
        logger.info("Added node '%s' to graph (total nodes: %d)", name, len(self.nodes))
        return self

    def add_edge(
        self,
        from_node: str,
        to_node: str,
    ) -> "StateGraph":
        """Add a static edge between two nodes.

        Creates a direct connection from one node to another. If the source
        node is START, the target node becomes the entry point for the graph.

        Args:
            from_node: Name of the source node.
            to_node: Name of the target node.

        Returns:
            StateGraph: The graph instance for method chaining.

        Example:
            >>> graph.add_edge("node1", "node2")
            >>> graph.add_edge(START, "entry_node")  # Sets entry point
        """
        logger.debug("Adding edge from '%s' to '%s'", from_node, to_node)
        # Set entry point if edge is from START
        if from_node == START:
            self.entry_point = to_node
            logger.info("Set entry point to '%s'", to_node)
        self.edges.append(Edge(from_node, to_node))
        logger.debug("Added edge (total edges: %d)", len(self.edges))
        return self

    def add_conditional_edges(
        self,
        from_node: str,
        condition: Callable,
        path_map: dict[str, str] | None = None,
    ) -> "StateGraph":
        """Add conditional routing between nodes based on runtime evaluation.

        Creates dynamic routing logic where the next node is determined by evaluating
        a condition function against the current state. This enables complex branching
        logic, decision trees, and adaptive workflow routing.

        Args:
            from_node: Name of the source node where the condition is evaluated.
            condition: Callable function that takes the current AgentState and returns
                a value used for routing decisions. Should be deterministic and
                side-effect free.
            path_map: Optional dictionary mapping condition results to destination nodes.
                If provided, the condition's return value is looked up in this mapping.
                If None, the condition should return the destination node name directly.

        Returns:
            StateGraph: The graph instance for method chaining.

        Raises:
            ValueError: If the condition function or path_map configuration is invalid.

        Example:
            ```python
            # Direct routing - condition returns node name
            def route_by_priority(state):
                priority = state.data.get("priority", "normal")
                return "urgent_handler" if priority == "high" else "normal_handler"


            graph.add_conditional_edges("classifier", route_by_priority)


            # Mapped routing - condition result mapped to nodes
            def get_category(state):
                return state.data.get("category", "default")


            category_map = {
                "finance": "finance_processor",
                "legal": "legal_processor",
                "default": "general_processor",
            }
            graph.add_conditional_edges("categorizer", get_category, category_map)
            ```

        Note:
            The condition function receives the current AgentState and should return
            consistent results for the same state. If using path_map, ensure the
            condition's return values match the map keys exactly.
        """
        """Add conditional edges from a node based on a condition function.

        Creates edges that are traversed based on the result of a condition
        function. The condition function receives the current state and should
        return a value that determines which edge to follow.

        Args:
            from_node: Name of the source node.
            condition: Function that evaluates the current state and returns
                a value to determine the next node.
            path_map: Optional mapping from condition results to target nodes.
                If provided, creates multiple conditional edges. If None,
                creates a single conditional edge.

        Returns:
            StateGraph: The graph instance for method chaining.

        Example:
            >>> def route_condition(state):
            ...     return "success" if state.success else "failure"
            >>> graph.add_conditional_edges(
            ...     "processor",
            ...     route_condition,
            ...     {"success": "next_step", "failure": "error_handler"},
            ... )
        """
        # Create edges based on possible returns from condition function
        logger.debug(
            "Node '%s' adding conditional edges with path_map: %s",
            from_node,
            path_map,
        )
        if path_map:
            logger.debug(
                "Node '%s' adding conditional edges with path_map: %s", from_node, path_map
            )
            for condition_result, target_node in path_map.items():
                edge = Edge(from_node, target_node, condition)
                edge.condition_result = condition_result
                self.edges.append(edge)
        else:
            # Single conditional edge
            logger.debug("Node '%s' adding single conditional edge", from_node)
            self.edges.append(Edge(from_node, "", condition))
        return self

    def set_entry_point(self, node_name: str) -> "StateGraph":
        """Set the entry point for the graph."""
        self.entry_point = node_name
        self.add_edge(START, node_name)
        logger.info("Set entry point to '%s'", node_name)
        return self

    def override_node(
        self,
        name: str,
        func: Union[Callable, "ToolNode", "Agent"],
    ) -> "StateGraph":
        """Override an existing node with a different function.

        Use this in tests to swap production nodes with test doubles.
        The node must already exist in the graph.

        Args:
            name: Name of the existing node to override
            func: New function, ToolNode, or Agent to use

        Returns:
            StateGraph: The graph instance for method chaining.

        Raises:
            KeyError: If the node doesn't exist

        Example:
            ```python
            # Production
            graph = StateGraph()
            graph.add_node("MAIN", production_agent)

            # Test - override with test agent
            graph.override_node("MAIN", test_agent)
            ```
        """
        if name not in self.nodes:
            raise KeyError(f"Node '{name}' does not exist. Use add_node() first.")

        # Create new Node with same name but different function
        self.nodes[name] = Node(name, func, self._publisher)
        logger.debug("Overrode node '%s' with new function", name)
        return self

    def compile(
        self,
        checkpointer: BaseCheckpointer[StateT] | None = None,
        store: BaseStore | None = None,
        interrupt_before: list[str] | None = None,
        interrupt_after: list[str] | None = None,
        callback_manager: CallbackManager = CallbackManager(),
        shutdown_timeout: float = 30.0,
    ) -> "CompiledGraph[StateT]":
        """Compile the graph for execution.

        Args:
            checkpointer: Checkpointer for state persistence
            store: Store for additional data
            debug: Enable debug mode
            interrupt_before: List of node names to interrupt before execution
            interrupt_after: List of node names to interrupt after execution
            callback_manager: Callback manager for executing hooks
            shutdown_timeout: Timeout in seconds for graceful shutdown (default: 30.0)
        """
        logger.info(
            "Compiling graph with %d nodes, %d edges, entry_point='%s'",
            len(self.nodes),
            len(self.edges),
            self.entry_point,
        )
        logger.debug(
            "Compile options: interrupt_before=%s, interrupt_after=%s",
            interrupt_before,
            interrupt_after,
        )

        if not self.entry_point:
            error_msg = "No entry point set. Use set_entry_point() or add an edge from START."
            logger.error(error_msg)
            raise GraphError(
                message=error_msg,
                error_code="GRAPH_002",
                context={"nodes": list(self.nodes.keys())},
            )

        # Validate graph structure
        logger.debug("Validating graph structure")
        self._validate_graph()
        logger.debug("Graph structure validated successfully")

        # Validate interrupt node names
        interrupt_before = interrupt_before or []
        interrupt_after = interrupt_after or []

        all_interrupt_nodes = set(interrupt_before + interrupt_after)
        invalid_nodes = all_interrupt_nodes - set(self.nodes.keys())
        if invalid_nodes:
            error_msg = f"Invalid interrupt nodes: {invalid_nodes}. Must be existing node names."
            logger.error(error_msg)
            raise GraphError(
                message=error_msg,
                error_code="GRAPH_004",
                context={
                    "invalid_nodes": list(invalid_nodes),
                    "valid_nodes": list(self.nodes.keys()),
                },
            )

        self.compiled = True
        logger.info("Graph compilation completed successfully")
        # Import here to avoid circular import at module import time
        # Now update Checkpointer
        if checkpointer is None:
            from agentflow.checkpointer import InMemoryCheckpointer

            checkpointer = InMemoryCheckpointer[StateT]()
            logger.debug("No checkpointer provided, using InMemoryCheckpointer")

        # Import the CompiledGraph class
        from .compiled_graph import CompiledGraph

        # Setup dependencies
        self._container.bind_instance(
            BaseCheckpointer,
            checkpointer,
            allow_concrete=True,
        )  # not null as we set default
        self._container.bind_instance(
            BaseStore,
            store,
            allow_none=True,
            allow_concrete=True,
        )
        self._container.bind_instance(
            CallbackManager,
            callback_manager,
            allow_concrete=True,
        )  # not null as we set default
        self._container.bind_instance(StateGraph, self)

        # Bind Nodes
        self._container.bind_factory("get_node", lambda x: self.nodes[x])
        self._container.bind_factory("get_entry_point_node", lambda: self.nodes[self.entry_point])  # type: ignore

        app = CompiledGraph(
            state=self._state,
            interrupt_after=interrupt_after,
            interrupt_before=interrupt_before,
            state_graph=self,
            checkpointer=checkpointer,
            publisher=self._publisher,
            store=store,
            task_manager=self._task_manager,
            shutdown_timeout=shutdown_timeout,
        )

        self._container.bind(CompiledGraph, app)
        # Compile the Graph, so it will optimize the dependency graph
        self._container.compile()
        return app

    def _validate_graph(self):
        """Validate the graph structure."""
        # Check for orphaned nodes
        connected_nodes = set()
        for edge in self.edges:
            connected_nodes.add(edge.from_node)
            connected_nodes.add(edge.to_node)

        all_nodes = set(self.nodes.keys())
        orphaned = all_nodes - connected_nodes
        if orphaned - {START, END}:  # START and END can be orphaned
            logger.error("Orphaned nodes detected: %s", orphaned - {START, END})
            raise GraphError(
                message=f"Orphaned nodes detected: {orphaned - {START, END}}",
                error_code="GRAPH_003",
                context={
                    "orphaned_nodes": list(orphaned - {START, END}),
                    "all_nodes": list(all_nodes),
                    "connected_nodes": list(connected_nodes),
                },
            )

        # Check that all edge targets exist
        for edge in self.edges:
            if edge.to_node and edge.to_node not in self.nodes:
                logger.error("Edge '%s' targets non-existent node: %s", edge, edge.to_node)
                raise GraphError(
                    message=f"Edge targets non-existent node: {edge.to_node}",
                    error_code="GRAPH_004",
                    context={
                        "edge": str(edge),
                        "target_node": edge.to_node,
                        "available_nodes": list(self.nodes.keys()),
                    },
                )
