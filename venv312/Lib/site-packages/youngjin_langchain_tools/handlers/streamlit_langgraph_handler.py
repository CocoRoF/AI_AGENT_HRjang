# youngjin_langchain_tools/handlers/streamlit_langgraph_handler.py
"""
Streamlit handler for LangGraph agents.

This module provides a handler class that simplifies streaming
LangGraph agent responses in Streamlit applications.

Replaces the deprecated StreamlitCallbackHandler for LangGraph-based agents.
"""

from typing import Any, Dict, List, Optional, Generator
from dataclasses import dataclass
import logging
import re

# Configure logging
logger = logging.getLogger(__name__)


# ============================================================
# Error Patterns for User-Friendly Messages
# ============================================================
ERROR_PATTERNS = {
    # OpenAI errors
    r"AuthenticationError.*API key|openai.*api.*key|OPENAI_API_KEY": {
        "title": "🔑 OpenAI API Key 오류",
        "message": "OpenAI API 키가 설정되지 않았거나 유효하지 않습니다.",
        "solution": [
            "환경변수 `OPENAI_API_KEY`를 설정하거나 클라이언트 초기화 시 `api_key` 파라미터를 전달하세요.",
            "API 키 발급: https://platform.openai.com/api-keys",
        ],
    },
    r"RateLimitError|rate_limit|429": {
        "title": "⏱️ Rate Limit 초과",
        "message": "API 요청 한도를 초과했습니다.",
        "solution": [
            "잠시 후 다시 시도하거나, API 사용량 및 요금제를 확인하세요.",
        ],
    },
    r"InsufficientQuotaError|insufficient_quota|billing": {
        "title": "💳 크레딧 부족",
        "message": "API 크레딧이 부족합니다.",
        "solution": [
            "API 제공자의 결제 페이지에서 크레딧을 충전하세요.",
        ],
    },
    r"InvalidRequestError|invalid_request": {
        "title": "❌ 잘못된 요청",
        "message": "API 요청 형식이 올바르지 않습니다.",
        "solution": [
            "입력 데이터와 모델명이 올바른지 확인하세요.",
        ],
    },
    # Anthropic errors
    r"anthropic.*authentication|ANTHROPIC_API_KEY": {
        "title": "🔑 Anthropic API Key 오류",
        "message": "Anthropic API 키가 설정되지 않았거나 유효하지 않습니다.",
        "solution": [
            "환경변수 `ANTHROPIC_API_KEY`를 설정하세요.",
            "API 키 발급: https://console.anthropic.com/",
        ],
    },
    # Google errors
    r"google.*api.*key|GOOGLE_API_KEY": {
        "title": "🔑 Google API Key 오류",
        "message": "Google API 키가 설정되지 않았거나 유효하지 않습니다.",
        "solution": [
            "환경변수 `GOOGLE_API_KEY`를 설정하세요.",
            "API 키 발급: https://aistudio.google.com/apikey",
        ],
    },
    # Network errors
    r"ConnectionError|connection.*refused|network": {
        "title": "🌐 네트워크 오류",
        "message": "API 서버에 연결할 수 없습니다.",
        "solution": [
            "인터넷 연결 및 방화벽/프록시 설정을 확인하세요.",
        ],
    },
    r"TimeoutError|timeout|timed out": {
        "title": "⏰ 시간 초과",
        "message": "API 요청이 시간 초과되었습니다.",
        "solution": [
            "네트워크 연결을 확인하고 잠시 후 다시 시도하세요.",
        ],
    },
    # Model errors
    r"model.*not.*found|does not exist|invalid.*model": {
        "title": "🤖 모델 오류",
        "message": "지정된 모델을 찾을 수 없습니다.",
        "solution": [
            "모델명과 접근 권한을 확인하세요.",
        ],
    },
}


def _parse_error(error: Exception) -> Dict[str, Any]:
    """Parse an exception and return user-friendly error information."""
    error_str = str(error)
    error_type = type(error).__name__
    full_error = f"{error_type}: {error_str}"

    # Try to match known error patterns
    for pattern, info in ERROR_PATTERNS.items():
        if re.search(pattern, full_error, re.IGNORECASE):
            return {
                "matched": True,
                "title": info["title"],
                "message": info["message"],
                "solution": info["solution"],
                "original_error": error_str[:500],  # Truncate for display
            }

    # Unknown error - return generic info
    return {
        "matched": False,
        "title": "❗ 오류 발생",
        "message": f"{error_type}",
        "solution": ["에러 메시지를 확인하고 문제를 해결해주세요."],
        "original_error": error_str[:500],
    }


@dataclass
class StreamlitLanggraphHandlerConfig:
    """Configuration for StreamlitLanggraphHandler.

    Attributes:
        expand_new_thoughts: 도구 결과 expander를 펼친 상태로 표시할지 여부.
        collapse_completed_thoughts: 완료 시 status container를 접을지 여부.
        max_thought_containers: 표시할 최대 사고 과정 수 (초과 시 History로 이동).
        max_tool_content_length: 도구 결과 최대 표시 문자 수.
        show_tool_calls: 도구 호출 정보 표시 여부.
        show_tool_results: 도구 실행 결과 표시 여부.
    """

    expand_new_thoughts: bool = True
    """Whether to show tool result expanders in expanded state."""

    collapse_completed_thoughts: bool = True
    """Whether to collapse the status container when agent completes."""

    max_thought_containers: int = 4
    """Maximum number of thought containers to show at once.
    When exceeded, oldest thoughts are moved to a 'History' expander."""

    max_tool_content_length: int = 2000
    """Maximum length of tool output to display before truncating."""

    show_tool_calls: bool = True
    """Whether to display tool call information."""

    show_tool_results: bool = True
    """Whether to display tool execution results."""

    thinking_label: str = "🤔 Thinking..."
    """Label shown while the agent is processing."""

    complete_label: str = "✅ Complete!"
    """Label shown when processing is complete."""

    tool_call_emoji: str = "🔧"
    """Emoji for tool calls."""

    tool_complete_emoji: str = "✅"
    """Emoji for completed tool executions."""

    cursor: str = "▌"
    """Cursor character shown during streaming."""


class StreamlitLanggraphHandler:
    """
    Handler for streaming LangGraph agent responses in Streamlit.

    This class provides a simple interface to visualize LangGraph agent
    execution in Streamlit, similar to how StreamlitCallbackHandler worked
    for the older LangChain AgentExecutor.

    Features:
    - Real-time streaming of agent responses
    - Tool call visualization with expandable details
    - Tool execution results with collapsible output
    - Status indicator showing agent progress
    - Configurable display options

    Example:
        ```python
        import streamlit as st
        from youngjin_langchain_tools import StreamlitLanggraphHandler

        with st.chat_message("assistant"):
            handler = StreamlitLanggraphHandler(
                container=st.container(),
                expand_new_thoughts=True
            )
            response = handler.invoke(
                agent=my_agent,
                input={"messages": [{"role": "user", "content": prompt}]},
                config={"configurable": {"thread_id": thread_id}}
            )
            # response contains the final text
        ```

    For more control, use stream() method:
        ```python
        handler = StreamlitLanggraphHandler(st.container())
        for event in handler.stream(agent, input, config):
            # event contains streaming data if needed
            pass
        final_response = handler.get_response()
        ```
    """

    def __init__(
        self,
        container: Any,
        *,
        expand_new_thoughts: bool = True,
        collapse_completed_thoughts: bool = True,
        max_thought_containers: int = 4,
        max_tool_content_length: int = 2000,
        show_tool_calls: bool = True,
        show_tool_results: bool = True,
        thinking_label: str = "🤔 Thinking...",
        complete_label: str = "✅ Complete!",
        config: Optional[StreamlitLanggraphHandlerConfig] = None,
    ):
        """
        Initialize the StreamlitLanggraphHandler.

        Args:
            container: Streamlit container to render content in.
                       Usually st.container() or similar.
            expand_new_thoughts: Whether to expand status container
                                 to show tool calls. Defaults to True.
            collapse_completed_thoughts: Whether to collapse completed
                                         thought containers. Defaults to True.
            max_thought_containers: Maximum number of thought containers
                                    to show at once. Oldest are moved to
                                    'History' expander. Defaults to 4.
            max_tool_content_length: Maximum characters of tool output
                                     to display. Defaults to 2000.
            show_tool_calls: Whether to show tool call info. Defaults to True.
            show_tool_results: Whether to show tool results. Defaults to True.
            thinking_label: Label while processing. Defaults to "🤔 Thinking...".
            complete_label: Label when complete. Defaults to "✅ Complete!".
            config: Optional config object. If provided, overrides other params.
        """
        if config is not None:
            self._config = config
        else:
            self._config = StreamlitLanggraphHandlerConfig(
                expand_new_thoughts=expand_new_thoughts,
                collapse_completed_thoughts=collapse_completed_thoughts,
                max_thought_containers=max_thought_containers,
                max_tool_content_length=max_tool_content_length,
                show_tool_calls=show_tool_calls,
                show_tool_results=show_tool_results,
                thinking_label=thinking_label,
                complete_label=complete_label,
            )

        self._container = container
        self._final_response: str = ""
        self._status_container: Any = None
        self._thoughts_placeholder: Any = None  # Placeholder inside status for re-render
        self._response_placeholder: Any = None
        self._thought_history: List[Dict[str, Any]] = []  # History of old thoughts
        self._current_thoughts: List[Dict[str, Any]] = []  # Current visible thoughts
        self._thought_counter: int = 0  # Unique ID counter for thoughts

    @property
    def config(self) -> StreamlitLanggraphHandlerConfig:
        """Get the handler configuration."""
        return self._config

    def get_response(self) -> str:
        """
        Get the final response text after streaming completes.

        Returns:
            The accumulated response text from the agent.
        """
        return self._final_response

    def invoke(
        self,
        agent: Any,
        input: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Invoke the agent and stream the response with visualization.

        This is the main method for simple usage. It handles all the
        streaming complexity and returns the final response.

        Args:
            agent: The LangGraph agent (CompiledGraph) to invoke.
            input: Input dictionary, typically {"messages": [...]}.
            config: Optional config dict with "configurable" key for thread_id etc.

        Returns:
            The final response text from the agent.

        Example:
            ```python
            response = handler.invoke(
                agent=my_agent,
                input={"messages": [{"role": "user", "content": "Hello"}]},
                config={"configurable": {"thread_id": "123"}}
            )
            st.write(response)
            ```
        """
        # Consume the generator to completion
        for _ in self.stream(agent, input, config):
            pass
        return self._final_response

    def stream(
        self,
        agent: Any,
        input: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Stream agent execution with visualization.

        This method provides more control than invoke(), yielding
        each streaming event for custom processing.

        Args:
            agent: The LangGraph agent (CompiledGraph) to invoke.
            input: Input dictionary, typically {"messages": [...]}.
            config: Optional config dict.

        Yields:
            Dictionary with event information:
            - "type": "tool_call" | "tool_result" | "token" | "complete"
            - "data": Event-specific data

        Example:
            ```python
            for event in handler.stream(agent, input, config):
                if event["type"] == "token":
                    # Custom token handling
                    pass
            ```
        """
        # Import streamlit here to avoid import errors when not using streamlit
        try:
            import streamlit as st
        except ImportError:
            raise ImportError(
                "streamlit is required for StreamlitLanggraphHandler. "
                "Install it with: pip install streamlit"
            )

        # Reset state
        self._final_response = ""
        self._thought_history = []
        self._current_thoughts = []
        self._thought_counter = 0

        # Create UI components
        with self._container:
            self._status_container = st.status(
                self._config.thinking_label,
                expanded=True  # Always start expanded during processing
            )
            # Create placeholder INSIDE status for thought re-rendering
            with self._status_container:
                self._thoughts_placeholder = st.empty()
            self._response_placeholder = st.empty()

        # Stream from agent with error handling
        agent_config = config or {}

        try:
            for stream_mode, data in agent.stream(
                input,
                config=agent_config,
                stream_mode=["messages", "updates"]
            ):
                if stream_mode == "updates":
                    yield from self._handle_updates(data)
                elif stream_mode == "messages":
                    yield from self._handle_messages(data)

            # Mark as complete - collapse based on config
            self._status_container.update(
                label=self._config.complete_label,
                state="complete",
                expanded=not self._config.collapse_completed_thoughts
            )

        except Exception as e:
            # Parse error and display user-friendly message
            error_info = _parse_error(e)

            # Update status to show error
            self._status_container.update(
                label="❌ 오류 발생",
                state="error",
                expanded=True
            )

            # Display error in status container
            with self._status_container:
                st.error(f"**{error_info['title']}**")
                st.markdown(f"_{error_info['message']}_")

                st.markdown("**해결 방법:**")
                for solution in error_info["solution"]:
                    st.markdown(f"  {solution}")

                with st.expander("🔍 상세 에러 메시지", expanded=False):
                    st.code(error_info["original_error"], language="text")

            # Log the full error for debugging
            logger.error(f"Agent execution error: {e}", exc_info=True)

            # Yield error event
            yield {
                "type": "error",
                "data": {
                    "error_type": type(e).__name__,
                    "error_info": error_info,
                    "original_error": str(e),
                }
            }
            return  # Stop further processing

        # Final render without cursor
        if self._final_response:
            self._response_placeholder.markdown(self._final_response)

        yield {"type": "complete", "data": {"response": self._final_response}}

    def _add_thought(self, thought_type: str, data: Dict[str, Any]) -> None:
        """Add a thought and manage history if max_thought_containers exceeded."""
        self._thought_counter += 1
        thought = {
            "id": self._thought_counter,
            "type": thought_type,
            "data": data
        }
        self._current_thoughts.append(thought)

        # Check if we need to move old thoughts to history
        max_containers = self._config.max_thought_containers
        while len(self._current_thoughts) > max_containers:
            old_thought = self._current_thoughts.pop(0)
            self._thought_history.append(old_thought)

    def _render_thought_item(
        self,
        thought: Dict[str, Any],
        st_module: Any,
        in_history: bool = False
    ) -> None:
        """Render a single thought item.

        Args:
            thought: The thought dict with id, type, data.
            st_module: The streamlit module reference.
            in_history: Whether this thought is in history section.
        """
        thought_type = thought["type"]
        thought_data = thought["data"]

        if thought_type == "tool_call":
            # Tool call display: 🔧 tool_name: {args}
            st_module.markdown(
                f"{self._config.tool_call_emoji} "
                f"**{thought_data['name']}**: `{thought_data['args']}`"
            )

        elif thought_type == "tool_result":
            # Tool result display: ✅ tool_name 완료 + expander
            tool_name = thought_data['name']
            content = thought_data['content']

            st_module.markdown(
                f"{self._config.tool_complete_emoji} "
                f"**{tool_name}** 완료"
            )

            # Determine max length and expand state
            if in_history:
                max_len = 200
                should_expand = False  # History items always collapsed
            else:
                max_len = self._config.max_tool_content_length
                should_expand = self._config.expand_new_thoughts

            with st_module.expander(f"📋 {tool_name} 결과 보기", expanded=should_expand):
                if len(content) > max_len:
                    st_module.code(content[:max_len] + "\n... (truncated)", language="text")
                else:
                    st_module.code(content, language="text")

    def _render_thoughts_in_status(self) -> None:
        """Render all thoughts (history + current) inside the status container.

        Uses placeholder.container() to completely replace previous content.
        """
        import streamlit as st

        # Clear and re-render using placeholder
        with self._thoughts_placeholder.container():
            # Render history expander if there are old thoughts
            if self._thought_history:
                with st.expander(
                    f"📜 History ({len(self._thought_history)} items)",
                    expanded=False
                ):
                    for thought in self._thought_history:
                        self._render_thought_item(thought, st, in_history=True)

            # Render current thoughts
            for thought in self._current_thoughts:
                self._render_thought_item(thought, st, in_history=False)

    def _handle_updates(
        self,
        data: Dict[str, Any]
    ) -> Generator[Dict[str, Any], None, None]:
        """Handle 'updates' stream mode events."""
        for source, update in data.items():
            if not isinstance(update, dict):
                continue

            messages = update.get("messages", [])
            for msg in messages:
                # Handle tool calls
                if hasattr(msg, 'tool_calls') and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_name = tc.get('name', 'tool')
                        tool_args = tc.get('args', {})

                        # Only add to thoughts if show_tool_calls is True
                        if self._config.show_tool_calls:
                            self._add_thought(
                                "tool_call",
                                {"name": tool_name, "args": tool_args}
                            )
                            self._render_thoughts_in_status()

                        # Update status label to show current action
                        self._status_container.update(
                            label=f"{self._config.tool_call_emoji} {tool_name}...",
                            state="running"
                        )

                        yield {
                            "type": "tool_call",
                            "data": {"name": tool_name, "args": tool_args}
                        }

                # Handle tool results
                if source == "tools" and hasattr(msg, 'name'):
                    tool_name = msg.name
                    tool_content = str(msg.content) if hasattr(msg, 'content') else ""

                    # Only add to thoughts if show_tool_results is True
                    if self._config.show_tool_results:
                        self._add_thought(
                            "tool_result",
                            {"name": tool_name, "content": tool_content}
                        )
                        self._render_thoughts_in_status()

                    # Update status to show thinking again
                    self._status_container.update(
                        label=self._config.thinking_label,
                        state="running"
                    )

                    yield {
                        "type": "tool_result",
                        "data": {"name": tool_name, "content": tool_content}
                    }

    def _handle_messages(
        self,
        data: tuple
    ) -> Generator[Dict[str, Any], None, None]:
        """Handle 'messages' stream mode events."""
        chunk, metadata = data

        # Skip tool node messages
        if metadata.get("langgraph_node") == "tools":
            return

        # Handle content chunks
        if hasattr(chunk, 'content') and chunk.content:
            # Skip tool call chunks
            if hasattr(chunk, 'tool_call_chunks') and chunk.tool_call_chunks:
                return

            self._final_response += chunk.content
            self._response_placeholder.markdown(
                self._final_response + self._config.cursor
            )

            yield {
                "type": "token",
                "data": {"content": chunk.content, "accumulated": self._final_response}
            }
