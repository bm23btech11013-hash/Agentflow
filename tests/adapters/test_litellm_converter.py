"""Tests for LiteLLM converter functionality."""
import json
import logging
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch, MagicMock

import pytest

from agentflow.adapters.llm.litellm_converter import LiteLLMConverter
from agentflow.state.message import Message
from agentflow.state.message_block import ReasoningBlock, ToolCallBlock, TextBlock


class MockModelResponse:
    """Mock ModelResponse for testing."""
    
    def __init__(self, data):
        self.id = data.get("id", "test_id")
        self._data = data
    
    def model_dump(self):
        return self._data


class MockModelResponseStream:
    """Mock ModelResponseStream for testing."""
    
    def __init__(self, id="test_id", choices=None):
        self.id = id
        self.choices = choices or []


class MockDelta:
    """Mock delta for streaming response."""
    
    def __init__(self, content="", reasoning_content="", tool_calls=None):
        self.content = content
        self.reasoning_content = reasoning_content
        self.tool_calls = tool_calls or []


class MockChoice:
    """Mock choice for streaming response."""
    
    def __init__(self, delta=None):
        self.delta = delta


class MockToolCall:
    """Mock tool call for testing."""
    
    def __init__(self, id="test_tool_id", name="test_tool", arguments="{}"):
        self.id = id
        self.function = Mock()
        self.function.name = name
        self.function.arguments = arguments
    
    def model_dump(self):
        return {
            "id": self.id,
            "function": {
                "name": self.function.name,
                "arguments": self.function.arguments
            }
        }


class MockCustomStreamWrapper:
    """Mock CustomStreamWrapper for testing."""
    
    def __init__(self, chunks):
        self.chunks = chunks
        self.index = 0
    
    def __aiter__(self):
        return self
    
    async def __anext__(self):
        if self.index >= len(self.chunks):
            raise StopAsyncIteration
        chunk = self.chunks[self.index]
        self.index += 1
        return chunk
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.index >= len(self.chunks):
            raise StopIteration
        chunk = self.chunks[self.index]
        self.index += 1
        return chunk


class TestLiteLLMConverter:
    """Test class for LiteLLM converter."""
    
    @pytest.fixture
    def converter(self):
        """Create a LiteLLM converter instance."""
        return LiteLLMConverter()
    
    def test_converter_init(self, converter):
        """Test LiteLLM converter initialization."""
        assert isinstance(converter, LiteLLMConverter)
    
    @patch('agentflow.adapters.llm.litellm_converter.HAS_LITELLM', False)
    @pytest.mark.asyncio
    async def test_convert_response_no_litellm(self, converter):
        """Test convert_response raises ImportError when LiteLLM is not available."""
        response = MockModelResponse({"id": "test"})
        
        with pytest.raises(ImportError, match="litellm is not installed"):
            await converter.convert_response(response)
    
    @patch('agentflow.adapters.llm.litellm_converter.HAS_LITELLM', True)
    @pytest.mark.asyncio
    async def test_convert_response_basic(self, converter):
        """Test basic response conversion."""
        response_data = {
            "id": "test_id_123",
            "model": "gpt-3.5-turbo",
            "object": "chat.completion",
            "created": 1234567890,
            "choices": [
                {
                    "message": {
                        "content": "Hello, how can I help you?",
                        "role": "assistant"
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 15,
                "total_tokens": 25,
                "cache_creation_input_tokens": 2,
                "cache_read_input_tokens": 3,
                "completion_tokens_details": {"reasoning_tokens": 1}
            }
        }
        
        response = MockModelResponse(response_data)
        
        message = await converter.convert_response(response)
        
        assert isinstance(message, Message)
        assert message.message_id == "test_id_123"  # Should use the response ID
        assert message.role == "assistant"
        assert len(message.content) == 1
        assert isinstance(message.content[0], TextBlock)
        assert message.content[0].text == "Hello, how can I help you?"
        assert message.usages is not None
        assert message.usages.prompt_tokens == 10
        assert message.usages.completion_tokens == 15
        assert message.usages.total_tokens == 25
        assert message.usages.reasoning_tokens == 1
        assert message.metadata["model"] == "gpt-3.5-turbo"
        assert message.metadata["finish_reason"] == "stop"
    
    @patch('agentflow.adapters.llm.litellm_converter.HAS_LITELLM', True)
    @pytest.mark.asyncio
    async def test_convert_response_with_reasoning(self, converter):
        """Test response conversion with reasoning content."""
        response_data = {
            "id": "test_reasoning",
            "choices": [
                {
                    "message": {
                        "content": "Based on my analysis...",
                        "reasoning_content": "Let me think through this step by step..."
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {}
        }
        
        response = MockModelResponse(response_data)
        
        with patch('agentflow.state.message.generate_id', return_value="reasoning_id"):
            message = await converter.convert_response(response)
        
        assert len(message.content) == 2
        assert isinstance(message.content[0], TextBlock)
        assert isinstance(message.content[1], ReasoningBlock)
        assert message.content[1].summary == "Let me think through this step by step..."
        assert message.reasoning == "Let me think through this step by step..."

    @patch('agentflow.adapters.llm.litellm_converter.HAS_LITELLM', True)
    @pytest.mark.asyncio
    async def test_convert_response_with_thinking_blocks(self, converter):
        """Test response conversion with thinking_blocks when reasoning_content is empty."""
        response_data = {
            "id": "test_thinking_blocks",
            "choices": [
                {
                    "message": {
                        "content": "The answer is 42.",
                        "reasoning_content": "",
                        "thinking_blocks": [
                            {"type": "thinking", "thinking": "First, let me consider..."},
                            {"type": "thinking", "thinking": "Now combining results..."},
                        ],
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {},
        }

        response = MockModelResponse(response_data)

        with patch('agentflow.state.message.generate_id', return_value="thinking_id"):
            message = await converter.convert_response(response)

        assert len(message.content) == 2
        assert isinstance(message.content[0], TextBlock)
        assert message.content[0].text == "The answer is 42."
        assert isinstance(message.content[1], ReasoningBlock)
        assert message.content[1].summary == "First, let me consider...\n\nNow combining results..."
        assert message.reasoning == "First, let me consider...\n\nNow combining results..."

    @patch('agentflow.adapters.llm.litellm_converter.HAS_LITELLM', True)
    @pytest.mark.asyncio
    async def test_convert_response_thinking_blocks_ignored_when_reasoning_content_present(self, converter):
        """Test that thinking_blocks are ignored when reasoning_content is already present."""
        response_data = {
            "id": "test_reasoning_priority",
            "choices": [
                {
                    "message": {
                        "content": "Result",
                        "reasoning_content": "Direct reasoning",
                        "thinking_blocks": [
                            {"type": "thinking", "thinking": "Should be ignored"},
                        ],
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {},
        }

        response = MockModelResponse(response_data)

        with patch('agentflow.state.message.generate_id', return_value="priority_id"):
            message = await converter.convert_response(response)

        assert isinstance(message.content[1], ReasoningBlock)
        assert message.content[1].summary == "Direct reasoning"

    @patch('agentflow.adapters.llm.litellm_converter.HAS_LITELLM', True)
    @pytest.mark.asyncio
    async def test_convert_response_with_tool_calls(self, converter):
        """Test response conversion with tool calls."""
        response_data = {
            "id": "test_tools",
            "choices": [
                {
                    "message": {
                        "content": "I'll help you with that calculation.",
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "calculate",
                                    "arguments": '{"x": 5, "y": 10}'
                                }
                            }
                        ]
                    },
                    "finish_reason": "tool_calls"
                }
            ],
            "usage": {}
        }
        
        response = MockModelResponse(response_data)
        
        with patch('agentflow.state.message.generate_id', return_value="tools_id"):
            message = await converter.convert_response(response)
        
        assert len(message.content) == 2
        assert isinstance(message.content[0], TextBlock)
        assert isinstance(message.content[1], ToolCallBlock)
        assert message.content[1].name == "calculate"
        assert message.content[1].args == {"x": 5, "y": 10}
        assert message.content[1].id == "call_123"
        assert message.tools_calls is not None
        assert len(message.tools_calls) == 1
    
    @patch('agentflow.adapters.llm.litellm_converter.HAS_LITELLM', True)
    @pytest.mark.asyncio
    async def test_convert_response_empty_content(self, converter):
        """Test response conversion with empty content."""
        response_data = {
            "id": "empty_test",
            "choices": [
                {
                    "message": {
                        "content": "",
                        "role": "assistant"
                    },
                    "finish_reason": "length"
                }
            ],
            "usage": {}
        }
        
        response = MockModelResponse(response_data)
        
        with patch('agentflow.state.message.generate_id', return_value="empty_id"):
            message = await converter.convert_response(response)
        
        assert len(message.content) == 0
        assert message.metadata["finish_reason"] == "length"
    
    @patch('agentflow.adapters.llm.litellm_converter.HAS_LITELLM', True)
    @pytest.mark.asyncio
    async def test_convert_response_missing_fields(self, converter):
        """Test response conversion with missing optional fields."""
        response_data = {
            "id": "minimal_test",
            "choices": [{}],
            "usage": {}
        }
        
        response = MockModelResponse(response_data)
        
        message = await converter.convert_response(response)
        
        assert message.message_id == "minimal_test"  # Should use the response ID
        assert len(message.content) == 0
        assert message.metadata["finish_reason"] == "UNKNOWN"
    
    def test_process_chunk_empty_chunk(self, converter):
        """Test _process_chunk with empty/None chunk."""
        result = converter._process_chunk(None, 0, "", "", [], set())
        
        assert result == ("", "", [], 0, None)
    
    def test_process_chunk_no_choices(self, converter):
        """Test _process_chunk with chunk that has no choices."""
        chunk = MockModelResponseStream("test", [])
        result = converter._process_chunk(chunk, 0, "", "", [], set())
        
        assert result == ("", "", [], 0, None)
    
    def test_process_chunk_no_delta(self, converter):
        """Test _process_chunk with choice that has no delta."""
        choice = MockChoice(None)
        chunk = MockModelResponseStream("test", [choice])
        result = converter._process_chunk(chunk, 0, "", "", [], set())
        
        assert result == ("", "", [], 0, None)
    
    def test_process_chunk_text_content(self, converter):
        """Test _process_chunk with text content."""
        delta = MockDelta(content="Hello world")
        choice = MockChoice(delta)
        chunk = MockModelResponseStream("test_123", [choice])
        
        with patch('agentflow.state.message.generate_id', return_value="chunk_id"):
            result = converter._process_chunk(chunk, 1, "Previous ", "", [], set())
        
        accumulated_content, accumulated_reasoning, tool_calls, seq, message = result
        assert accumulated_content == "Previous Hello world"
        assert message is not None
        assert len(message.content) == 1
        assert isinstance(message.content[0], TextBlock)
        assert message.content[0].text == "Hello world"
        assert message.delta is True
    
    def test_process_chunk_reasoning_content(self, converter):
        """Test _process_chunk with reasoning content."""
        delta = MockDelta(reasoning_content="Let me think...")
        choice = MockChoice(delta)
        chunk = MockModelResponseStream("test_reasoning", [choice])
        
        with patch('agentflow.state.message.generate_id', return_value="reasoning_chunk_id"):
            result = converter._process_chunk(chunk, 1, "", "Previous reasoning ", [], set())
        
        accumulated_content, accumulated_reasoning, tool_calls, seq, message = result
        assert accumulated_reasoning == "Previous reasoning Let me think..."
        assert message is not None
        assert len(message.content) == 1
        assert isinstance(message.content[0], ReasoningBlock)
        assert message.content[0].summary == "Let me think..."

    def test_process_chunk_thinking_blocks_in_delta(self, converter):
        """Test _process_chunk with thinking_blocks in delta when reasoning_content is empty."""
        delta = MockDelta(content="", reasoning_content="")
        # Attach thinking_blocks attribute to the delta
        delta.thinking_blocks = [
            {"type": "thinking", "thinking": "Delta thinking step"},
        ]
        choice = MockChoice(delta)
        chunk = MockModelResponseStream("test_thinking_delta", [choice])

        with patch('agentflow.state.message.generate_id', return_value="thinking_delta_id"):
            result = converter._process_chunk(chunk, 1, "", "", [], set())

        accumulated_content, accumulated_reasoning, tool_calls, seq, message = result
        assert accumulated_reasoning == "Delta thinking step"
        assert message is not None
        assert len(message.content) == 1
        assert isinstance(message.content[0], ReasoningBlock)
        assert message.content[0].summary == "Delta thinking step"

    def test_process_chunk_thinking_blocks_ignored_when_reasoning_content_present(self, converter):
        """Test that thinking_blocks in delta are ignored when reasoning_content is present."""
        delta = MockDelta(content="", reasoning_content="Direct delta reasoning")
        delta.thinking_blocks = [
            {"type": "thinking", "thinking": "Should be ignored"},
        ]
        choice = MockChoice(delta)
        chunk = MockModelResponseStream("test_priority_delta", [choice])

        with patch('agentflow.state.message.generate_id', return_value="priority_delta_id"):
            result = converter._process_chunk(chunk, 1, "", "", [], set())

        accumulated_content, accumulated_reasoning, tool_calls, seq, message = result
        assert accumulated_reasoning == "Direct delta reasoning"
        assert message.content[0].summary == "Direct delta reasoning"

    def test_process_chunk_tool_calls(self, converter):
        """Test _process_chunk with tool calls."""
        tool_call = MockToolCall("call_456", "test_function", '{"param": "value"}')
        delta = MockDelta(tool_calls=[tool_call])
        choice = MockChoice(delta)
        chunk = MockModelResponseStream("test_tools", [choice])
        tool_ids = set()
        
        with patch('agentflow.state.message.generate_id', return_value="tool_chunk_id"):
            result = converter._process_chunk(chunk, 1, "", "", [], tool_ids)
        
        accumulated_content, accumulated_reasoning, tool_calls_result, seq, message = result
        assert len(tool_calls_result) == 1
        assert "call_456" in tool_ids
        assert message is not None
        assert len(message.content) == 1
        assert isinstance(message.content[0], ToolCallBlock)
        assert message.content[0].name == "test_function"
        assert message.content[0].args == {"param": "value"}
    
    def test_process_chunk_duplicate_tool_call(self, converter):
        """Test _process_chunk ignores duplicate tool call IDs."""
        tool_call = MockToolCall("duplicate_id", "test_function", "{}")
        delta = MockDelta(tool_calls=[tool_call])
        choice = MockChoice(delta)
        chunk = MockModelResponseStream("test_duplicate", [choice])
        tool_ids = {"duplicate_id"}  # Already exists
        
        result = converter._process_chunk(chunk, 1, "", "", [], tool_ids)
        
        accumulated_content, accumulated_reasoning, tool_calls_result, seq, message = result
        assert len(tool_calls_result) == 0  # Should not add duplicate
    
    def test_process_chunk_combined_content(self, converter):
        """Test _process_chunk with text, reasoning, and tool calls."""
        tool_call = MockToolCall("combined_call", "combined_tool", '{"test": true}')
        delta = MockDelta(
            content="Text content",
            reasoning_content="Reasoning content",
            tool_calls=[tool_call]
        )
        choice = MockChoice(delta)
        chunk = MockModelResponseStream("combined_test", [choice])
        
        with patch('agentflow.state.message.generate_id', return_value="combined_id"):
            result = converter._process_chunk(chunk, 1, "", "", [], set())
        
        accumulated_content, accumulated_reasoning, tool_calls_result, seq, message = result
        assert accumulated_content == "Text content"
        assert accumulated_reasoning == "Reasoning content"
        assert len(tool_calls_result) == 1
        assert message is not None
        assert len(message.content) == 3  # Text, Reasoning, ToolCall


class TestLiteLLMConverterStreaming:
    """Test class for LiteLLM converter streaming functionality."""
    
    @pytest.fixture
    def converter(self):
        """Create a LiteLLM converter instance."""
        return LiteLLMConverter()
    
    @pytest.mark.asyncio
    async def test_handle_stream_async(self, converter):
        """Test _handle_stream with async iteration."""
        chunks = [
            MockModelResponseStream("stream_1", [MockChoice(MockDelta(content="Hello "))]),
            MockModelResponseStream("stream_2", [MockChoice(MockDelta(content="world!"))]),
        ]
        stream = MockCustomStreamWrapper(chunks)
        config = {"thread_id": "test_thread"}
        
        messages = []
        with patch('agentflow.state.message.generate_id', side_effect=["msg1", "msg2", "final"]):
            async for message in converter._handle_stream(config, "test_node", stream):
                messages.append(message)
        
        assert len(messages) == 3  # 2 chunks + 1 final
        assert messages[0].content[0].text == "Hello "
        assert messages[1].content[0].text == "world!"
        assert messages[2].content[0].text == "Hello world!"
        assert messages[2].delta is False  # Final message
        assert messages[2].metadata["thread_id"] == "test_thread"
    
    @pytest.mark.asyncio
    async def test_handle_stream_with_reasoning(self, converter):
        """Test _handle_stream with reasoning content."""
        chunks = [
            MockModelResponseStream("r1", [MockChoice(MockDelta(reasoning_content="Step 1"))]),
            MockModelResponseStream("r2", [MockChoice(MockDelta(reasoning_content=" Step 2"))]),
        ]
        stream = MockCustomStreamWrapper(chunks)
        
        messages = []
        with patch('agentflow.state.message.generate_id', side_effect=["r1", "r2", "final"]):
            async for message in converter._handle_stream({}, "reasoning_node", stream):
                messages.append(message)
        
        assert len(messages) == 3
        assert messages[2].reasoning == "Step 1 Step 2"
    
    @pytest.mark.asyncio
    async def test_handle_stream_with_tools(self, converter):
        """Test _handle_stream with tool calls."""
        tool_call = MockToolCall("stream_tool", "stream_function", '{"stream": "data"}')
        chunks = [
            MockModelResponseStream("t1", [MockChoice(MockDelta(content="Using tool"))]),
            MockModelResponseStream("t2", [MockChoice(MockDelta(tool_calls=[tool_call]))]),
        ]
        stream = MockCustomStreamWrapper(chunks)
        
        messages = []
        with patch('agentflow.state.message.generate_id', side_effect=["t1", "t2", "final"]):
            async for message in converter._handle_stream({}, "tool_node", stream):
                messages.append(message)
        
        assert len(messages) == 3
        final_message = messages[2]
        assert len(final_message.content) == 2  # Text + ToolCall
        assert isinstance(final_message.content[0], TextBlock)
        assert isinstance(final_message.content[1], ToolCallBlock)
        assert final_message.content[1].name == "stream_function"
    
    @patch('agentflow.adapters.llm.litellm_converter.HAS_LITELLM', False)
    @pytest.mark.asyncio
    async def test_convert_streaming_response_no_litellm(self, converter):
        """Test convert_streaming_response raises ImportError when LiteLLM unavailable."""
        response = Mock()
        
        with pytest.raises(ImportError, match="litellm is not installed"):
            async for _ in converter.convert_streaming_response({}, "test", response):
                pass
    
    @patch('agentflow.adapters.llm.litellm_converter.HAS_LITELLM', True)
    @patch('agentflow.adapters.llm.litellm_converter.CustomStreamWrapper', MockCustomStreamWrapper)
    @pytest.mark.asyncio
    async def test_convert_streaming_response_with_stream(self, converter):
        """Test convert_streaming_response with CustomStreamWrapper."""
        chunks = [
            MockModelResponseStream("s1", [MockChoice(MockDelta(content="Streaming "))]),
            MockModelResponseStream("s2", [MockChoice(MockDelta(content="test"))]),
        ]
        stream = MockCustomStreamWrapper(chunks)
        
        messages = []
        with patch('agentflow.state.message.generate_id', side_effect=["s1", "s2", "final"]):
            async for message in converter.convert_streaming_response({}, "stream_node", stream):
                messages.append(message)
        
        assert len(messages) == 3
        assert messages[2].content[0].text == "Streaming test"
    
    # @patch('agentflow.adapters.llm.litellm_converter.HAS_LITELLM', True)
    # @patch('agentflow.adapters.llm.litellm_converter.ModelResponse', MockModelResponse)
    # @pytest.mark.asyncio
    # async def test_convert_streaming_response_with_model_response(self, converter):
    #     """Test convert_streaming_response with ModelResponse."""
    #     response_data = {
    #         "id": "non_stream",
    #         "choices": [{"message": {"content": "Non-streaming response"}}],
    #         "usage": {}
    #     }
    #     response = MockModelResponse(response_data)
        
    #     messages = []
    #     with patch('agentflow.state.message.generate_id', return_value="non_stream_id"):
    #         async for message in converter.convert_streaming_response({}, "model_node", response):
    #             messages.append(message)
        
    #     assert len(messages) == 1
    #     assert messages[0].content[0].text == "Non-streaming response"
    
    # @patch('agentflow.adapters.llm.litellm_converter.HAS_LITELLM', True)
    # @pytest.mark.asyncio
    # async def test_convert_streaming_response_unsupported_type(self, converter):
    #     """Test convert_streaming_response with unsupported response type."""
    #     unsupported_response = "unsupported_string"
        
    #     with pytest.raises(Exception, match="Unsupported response type"):
    #         async for _ in converter.convert_streaming_response({}, "test", unsupported_response):
    #             pass


class TestLiteLLMConverterEdgeCases:
    """Test class for LiteLLM converter edge cases."""
    
    @pytest.fixture
    def converter(self):
        """Create a LiteLLM converter instance."""
        return LiteLLMConverter()
    
    @patch('agentflow.adapters.llm.litellm_converter.HAS_LITELLM', True)
    @pytest.mark.asyncio
    async def test_convert_response_invalid_tool_call_data(self, converter):
        """Test response conversion with invalid tool call data."""
        response_data = {
            "id": "invalid_tools",
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {"id": None, "function": {"name": "test", "arguments": "{}"}},  # No ID
                            {"id": "valid_id", "function": {"name": None, "arguments": "{}"}},  # No name
                            {"id": "valid_id2", "function": {"name": "test", "arguments": None}},  # No args
                        ]
                    }
                }
            ],
            "usage": {}
        }
        
        response = MockModelResponse(response_data)
        
        with patch('agentflow.state.message.generate_id', return_value="invalid_id"):
            message = await converter.convert_response(response)
        
        # Should skip all invalid tool calls
        assert len(message.content) == 0
        assert message.tools_calls is None
    
    @pytest.mark.asyncio
    async def test_handle_stream_sync_fallback(self, converter):
        """Test _handle_stream falls back to sync iteration."""
        class SyncOnlyStream:
            def __init__(self, chunks):
                self.chunks = chunks
                self.index = 0
            
            def __iter__(self):
                return self
            
            def __next__(self):
                if self.index >= len(self.chunks):
                    raise StopIteration
                chunk = self.chunks[self.index]
                self.index += 1
                return chunk
        
        chunks = [
            MockModelResponseStream("sync1", [MockChoice(MockDelta(content="Sync "))]),
            MockModelResponseStream("sync2", [MockChoice(MockDelta(content="test"))]),
        ]
        stream = SyncOnlyStream(chunks)
        
        messages = []
        with patch('agentflow.state.message.generate_id', side_effect=["sync1", "sync2", "final"]):
            async for message in converter._handle_stream({}, "sync_node", stream):
                messages.append(message)
        
        assert len(messages) == 3
        assert messages[2].content[0].text == "Sync test"
    
    @pytest.mark.asyncio
    async def test_handle_stream_awaitable_stream(self, converter):
        """Test _handle_stream with awaitable stream."""
        chunks = [
            MockModelResponseStream("await1", [MockChoice(MockDelta(content="Awaitable"))]),
        ]
        
        async def awaitable_stream():
            return MockCustomStreamWrapper(chunks)
        
        messages = []
        with patch('agentflow.state.message.generate_id', side_effect=["await1", "final"]):
            async for message in converter._handle_stream({}, "await_node", awaitable_stream()):
                messages.append(message)
        
        assert len(messages) == 2
        assert messages[1].content[0].text == "Awaitable"
    
    @pytest.mark.asyncio
    async def test_handle_stream_exception_handling(self, converter):
        """Test _handle_stream handles exceptions gracefully."""
        class FailingStream:
            def __aiter__(self):
                return self
            
            async def __anext__(self):
                raise RuntimeError("Stream failed")
            
            def __iter__(self):
                return self
            
            def __next__(self):
                raise RuntimeError("Stream failed")
        
        stream = FailingStream()
        
        messages = []
        with patch('agentflow.state.message.generate_id', return_value="final"):
            async for message in converter._handle_stream({}, "fail_node", stream):
                messages.append(message)
        
        # Should still yield final message even if stream fails
        assert len(messages) == 1
        assert len(messages[0].content) == 0  # Empty content due to failure
    
    def test_process_chunk_none_tool_call(self, converter):
        """Test _process_chunk handles None tool calls."""
        delta = MockDelta(tool_calls=[None])  # None tool call
        choice = MockChoice(delta)
        chunk = MockModelResponseStream("none_tool", [choice])
        
        result = converter._process_chunk(chunk, 1, "", "", [], set())
        
        accumulated_content, accumulated_reasoning, tool_calls, seq, message = result
        assert len(tool_calls) == 0  # Should skip None tool calls
    
    @patch('agentflow.adapters.llm.litellm_converter.HAS_LITELLM', True)
    @pytest.mark.asyncio
    async def test_convert_response_with_dict_tool_calls(self, converter):
        """Test response conversion with tool calls as plain dicts."""
        response_data = {
            "id": "dict_tools_test",
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "function": {"name": "test_tool", "arguments": '{"x": 1}'}
                            }
                        ]
                    }
                }
            ],
            "usage": {}
        }
        
        response = MockModelResponse(response_data)
        message = await converter.convert_response(response)
        
        assert message.tools_calls is not None
        assert len(message.tools_calls) == 1
        assert message.tools_calls[0]["id"] == "call_123"