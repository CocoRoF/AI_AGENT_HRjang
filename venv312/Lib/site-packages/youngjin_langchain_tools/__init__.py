# youngjin_langchain_tools/__init__.py
"""
youngjin-langchain-tools Library

A collection of LangChain/LangGraph utilities for AI applications.

Package Structure:
- handlers: UI framework handlers (Streamlit, etc.)
- utils: Utility functions and helpers

Usage:
    from youngjin_langchain_tools import StreamlitLanggraphHandler

    # Use in Streamlit with LangGraph agent
    with st.chat_message("assistant"):
        handler = StreamlitLanggraphHandler(st.container())
        response = handler.invoke(
            agent=my_agent,
            input={"messages": [{"role": "user", "content": prompt}]},
            config={"configurable": {"thread_id": thread_id}}
        )
"""

__version__ = "0.1.0"

# Import subpackages
from youngjin_langchain_tools import handlers
from youngjin_langchain_tools import utils

# Import core classes for convenience
from youngjin_langchain_tools.handlers import StreamlitLanggraphHandler
from youngjin_langchain_tools.handlers.streamlit_langgraph_handler import (
    StreamlitLanggraphHandlerConfig,
)
from youngjin_langchain_tools.utils import configure

__all__ = [
    "__version__",
    # Subpackages
    "handlers",
    "utils",
    # Core classes
    "StreamlitLanggraphHandler",
    "StreamlitLanggraphHandlerConfig",
    # Utility functions
    "configure",
]
