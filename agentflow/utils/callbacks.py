"""
Callback system for TAF.

This module provides a comprehensive callback framework that allows users to define
their own validation logic and custom behavior at key points in the execution flow:

- before_invoke: Called before AI/TOOL/MCP invocation for input validation and modification
- after_invoke: Called after AI/TOOL/MCP invocation for output validation and modification
- on_error: Called when errors occur during invocation for error handling and logging

The system is generic and type-safe, supporting different callback types for different
invocation contexts.
"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Union

from agentflow.state.message import Message


logger = logging.getLogger("agentflow.utils")


class BaseValidator(ABC):
    """Abstract base class for message validators.

    Validators are used to validate message content before processing.
    They provide a simpler interface than callbacks, focused specifically
    on message validation.

    Example:
        ```python
        class MyValidator(BaseValidator):
            async def validate(self, messages: list[Message]) -> bool:
                for msg in messages:
                    if "bad_word" in msg.text():
                        raise ValidationError("Bad word detected", "content_policy")
                return True


        # Register with callback manager
        from agentflow.utils.callbacks import CallbackManager

        callback_manager = CallbackManager()
        callback_manager.register_input_validator(MyValidator())
        ```
    """

    @abstractmethod
    async def validate(self, messages: list[Message]) -> bool:
        """Validate a list of messages.

        Args:
            messages: List of Message objects to validate

        Returns:
            True if validation passes

        Raises:
            ValidationError: If validation fails
        """
        ...


class InvocationType(str, Enum):
    """Types of invocations that can trigger callbacks."""

    AI = "ai"
    TOOL = "tool"
    MCP = "mcp"
    INPUT_VALIDATION = "input_validation"
    SKILL = "skill"


@dataclass
class CallbackContext:
    """Context information passed to callbacks."""

    invocation_type: InvocationType
    node_name: str
    function_name: str | None = None
    metadata: dict[str, Any] | None = None


class BeforeInvokeCallback[T, R](ABC):
    """Abstract base class for before_invoke callbacks.

    Called before the AI model, tool, or MCP function is invoked.
    Allows for input validation and modification.
    """

    @abstractmethod
    async def __call__(self, context: CallbackContext, input_data: T) -> T | R:
        """Execute the before_invoke callback.

        Args:
            context: Context information about the invocation
            input_data: The input data about to be sent to the invocation

        Returns:
            Modified input data (can be same type or different type)

        Raises:
            Exception: If validation fails or modification cannot be performed
        """
        ...


class AfterInvokeCallback[T, R](ABC):
    """Abstract base class for after_invoke callbacks.

    Called after the AI model, tool, or MCP function is invoked.
    Allows for output validation and modification.
    """

    @abstractmethod
    async def __call__(self, context: CallbackContext, input_data: T, output_data: Any) -> Any | R:
        """Execute the after_invoke callback.

        Args:
            context: Context information about the invocation
            input_data: The original input data that was sent
            output_data: The output data returned from the invocation

        Returns:
            Modified output data (can be same type or different type)

        Raises:
            Exception: If validation fails or modification cannot be performed
        """
        ...


class OnErrorCallback(ABC):
    """Abstract base class for on_error callbacks.

    Called when an error occurs during invocation.
    Allows for error handling and logging.
    """

    @abstractmethod
    async def __call__(
        self, context: CallbackContext, input_data: Any, error: Exception
    ) -> Any | None:
        """Execute the on_error callback.

        Args:
            context: Context information about the invocation
            input_data: The input data that caused the error
            error: The exception that occurred

        Returns:
            Optional recovery value or None to re-raise the error

        Raises:
            Exception: If error handling fails or if the error should be re-raised
        """
        ...


# Type aliases for cleaner type hints
BeforeInvokeCallbackType = Union[
    BeforeInvokeCallback[Any, Any], Callable[[CallbackContext, Any], Union[Any, Awaitable[Any]]]
]

AfterInvokeCallbackType = Union[
    AfterInvokeCallback[Any, Any], Callable[[CallbackContext, Any, Any], Union[Any, Awaitable[Any]]]
]

OnErrorCallbackType = Union[
    OnErrorCallback,
    Callable[[CallbackContext, Any, Exception], Union[Any | None, Awaitable[Any | None]]],
]


class CallbackManager:
    """
    Manages registration and execution of callbacks for different invocation types.

    Handles before_invoke, after_invoke, and on_error callbacks for AI, TOOL, and MCP invocations.
    """

    def __init__(self):
        """
        Initialize the CallbackManager with empty callback registries.
        """
        self._before_callbacks: dict[InvocationType, list[BeforeInvokeCallbackType]] = {
            InvocationType.AI: [],
            InvocationType.TOOL: [],
            InvocationType.MCP: [],
            InvocationType.INPUT_VALIDATION: [],
            InvocationType.SKILL: [],
        }
        self._after_callbacks: dict[InvocationType, list[AfterInvokeCallbackType]] = {
            InvocationType.AI: [],
            InvocationType.TOOL: [],
            InvocationType.MCP: [],
            InvocationType.INPUT_VALIDATION: [],
            InvocationType.SKILL: [],
        }
        self._error_callbacks: dict[InvocationType, list[OnErrorCallbackType]] = {
            InvocationType.AI: [],
            InvocationType.TOOL: [],
            InvocationType.MCP: [],
            InvocationType.INPUT_VALIDATION: [],
            InvocationType.SKILL: [],
        }
        # Validator registry
        self._validators: list[BaseValidator] = []

    def register_before_invoke(
        self, invocation_type: InvocationType, callback: BeforeInvokeCallbackType
    ) -> None:
        """
        Register a before_invoke callback for a specific invocation type.

        Args:
            invocation_type (InvocationType): The type of invocation (AI, TOOL, MCP).
            callback (BeforeInvokeCallbackType): The callback to register.
        """
        self._before_callbacks[invocation_type].append(callback)

    def register_after_invoke(
        self, invocation_type: InvocationType, callback: AfterInvokeCallbackType
    ) -> None:
        """
        Register an after_invoke callback for a specific invocation type.

        Args:
            invocation_type (InvocationType): The type of invocation (AI, TOOL, MCP).
            callback (AfterInvokeCallbackType): The callback to register.
        """
        self._after_callbacks[invocation_type].append(callback)

    def register_on_error(
        self, invocation_type: InvocationType, callback: OnErrorCallbackType
    ) -> None:
        """
        Register an on_error callback for a specific invocation type.

        Args:
            invocation_type (InvocationType): The type of invocation (AI, TOOL, MCP).
            callback (OnErrorCallbackType): The callback to register.
        """
        self._error_callbacks[invocation_type].append(callback)

    async def execute_before_invoke(self, context: CallbackContext, input_data: Any) -> Any:
        """
        Execute all before_invoke callbacks for the given context.

        Args:
            context (CallbackContext): Context information about the invocation.
            input_data (Any): The input data to be validated or modified.

        Returns:
            Any: The modified input data after all callbacks.

        Raises:
            Exception: If any callback fails.
        """
        current_data = input_data

        for callback in self._before_callbacks[context.invocation_type]:
            try:
                if isinstance(callback, BeforeInvokeCallback):
                    current_data = await callback(context, current_data)
                elif callable(callback):
                    result = callback(context, current_data)
                    if hasattr(result, "__await__"):
                        current_data = await result
                    else:
                        current_data = result
            except Exception as e:
                await self.execute_on_error(context, input_data, e)
                raise

        return current_data

    async def execute_after_invoke(
        self,
        context: CallbackContext,
        input_data: Any,
        output_data: Any,
    ) -> Any:
        """
        Execute all after_invoke callbacks for the given context.

        Args:
            context (CallbackContext): Context information about the invocation.
            input_data (Any): The original input data sent to the invocation.
            output_data (Any): The output data returned from the invocation.

        Returns:
            Any: The modified output data after all callbacks.

        Raises:
            Exception: If any callback fails.
        """
        current_output = output_data

        for callback in self._after_callbacks[context.invocation_type]:
            try:
                if isinstance(callback, AfterInvokeCallback):
                    current_output = await callback(context, input_data, current_output)
                elif callable(callback):
                    result = callback(context, input_data, current_output)
                    if hasattr(result, "__await__"):
                        current_output = await result
                    else:
                        current_output = result
            except Exception as e:
                await self.execute_on_error(context, input_data, e)
                raise

        return current_output

    async def execute_on_error(
        self, context: CallbackContext, input_data: Any, error: Exception
    ) -> Message | None:
        """
        Execute all on_error callbacks for the given context.

        Args:
            context (CallbackContext): Context information about the invocation.
            input_data (Any): The input data that caused the error.
            error (Exception): The exception that occurred.

        Returns:
            Message | None: Recovery value from callbacks, or None if not handled.
        """
        recovery_value = None

        for callback in self._error_callbacks[context.invocation_type]:
            try:
                result = None
                if isinstance(callback, OnErrorCallback):
                    result = await callback(context, input_data, error)
                elif callable(callback):
                    result = callback(context, input_data, error)
                    if hasattr(result, "__await__"):
                        result = await result  # type: ignore

                if isinstance(result, Message) or result is None:
                    recovery_value = result
            except Exception as exc:
                logger.exception("Error callback failed: %s", exc)
                continue

        return recovery_value

    def register_input_validator(self, validator: BaseValidator) -> None:
        """
        Register a message validator for input validation.

        Validators provide a simpler interface for message validation
        compared to callbacks. They only need to implement validate(messages).

        Args:
            validator: BaseValidator instance to register

        Example:
            ```python
            from agentflow.utils.validators import PromptInjectionValidator

            callback_manager = CallbackManager()
            validator = PromptInjectionValidator()
            callback_manager.register_input_validator(validator)
            ```
        """
        self._validators.append(validator)
        logger.debug("Registered input validator: %s", validator.__class__.__name__)

    async def execute_validators(self, messages: list[Message]) -> bool:
        """
        Execute all registered validators on the given messages.

        Args:
            messages: List of Message objects to validate

        Returns:
            True if all validators pass

        Raises:
            ValidationError: If any validator fails
        """
        if not self._validators:
            logger.debug("No validators registered, skipping validation")
            return True

        logger.debug("Running %d validators on %d messages", len(self._validators), len(messages))

        for validator in self._validators:
            await validator.validate(messages)

        logger.debug("All validators passed")
        return True

    def clear_callbacks(self, invocation_type: InvocationType | None = None) -> None:
        """
        Clear callbacks for a specific invocation type or all types.

        Args:
            invocation_type (InvocationType | None): The invocation type to clear, or None for all.
        """
        if invocation_type:
            self._before_callbacks[invocation_type].clear()
            self._after_callbacks[invocation_type].clear()
            self._error_callbacks[invocation_type].clear()
        else:
            for inv_type in InvocationType:
                self._before_callbacks[inv_type].clear()
                self._after_callbacks[inv_type].clear()
                self._error_callbacks[inv_type].clear()

    def get_callback_counts(self) -> dict[str, dict[str, int]]:
        """
        Get count of registered callbacks by type for debugging.

        Returns:
            dict[str, dict[str, int]]: Counts of callbacks for each invocation type.
        """
        return {
            inv_type.value: {
                "before_invoke": len(self._before_callbacks[inv_type]),
                "after_invoke": len(self._after_callbacks[inv_type]),
                "on_error": len(self._error_callbacks[inv_type]),
            }
            for inv_type in InvocationType
        }
