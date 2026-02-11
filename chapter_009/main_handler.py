import streamlit as st
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver
import uuid

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

from tools.search_ddgs import search_ddgs
from tools.fetch_page import fetch_page

from youngjin_langchain_tools import StreamlitLanggraphHandler

CUSTOM_SYSTEM_PROMPT = """
당신은 사용자의 요청에 따라 인터넷에서 정보를 조사하는 어시스턴트입니다.
이미 알고 있는 정보만으로 답변하지 말고, 반드시 검색을 수행한 뒤 답변하세요.
(사용자가 읽을 페이지를 지정하는 등 특별한 경우는 예외)

검색 결과만으로 정보가 부족하다면 다음을 시도하세요.
- 검색 결과의 링크를 열어 페이지 내용을 직접 확인
- 검색 쿼리를 변경해 재검색
- 공식 문서뿐 아니라 블로그, 커뮤니티 등도 참고
- 한 페이지가 길 경우 3페이지 이상 스크롤 금지 (메모리 부담)

사용자에게 링크만 던지지 말고, 직접적인 답변을 제공하세요.
(나쁜 예: "다음 페이지를 참고하세요" / 좋은 예: 구체적인 답변이나 코드를 직접 제시)

답변 마지막에는 참조한 페이지의 URL을 반드시 기재하세요.
사용자가 사용하는 언어로 답변하세요.
"""


def init_page():
    st.set_page_config(page_title="Web Browsing Agent", page_icon="🤗")
    st.header("Web Browsing Agent 🤗")
    st.sidebar.title("Options")


def init_messages():
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "안녕하세요! 무엇이든 질문해주세요!"}
        ]
        st.session_state["checkpointer"] = InMemorySaver()
        st.session_state["thread_id"] = str(uuid.uuid4())


def select_model():
    models = ("GPT-5.2", "Claude Sonnet 4.5", "Gemini 2.5 Flash")
    model = st.sidebar.radio("Choose a model:", models)

    if model == "GPT-5.2":
        return ChatOpenAI(temperature=0, model="gpt-5.2")
    elif model == "Claude Sonnet 4.5":
        return ChatAnthropic(temperature=0, model="claude-sonnet-4-5-20250929")
    elif model == "Gemini 2.5 Flash":
        return ChatGoogleGenerativeAI(temperature=0, model="gemini-2.5-flash")


def create_web_browsing_agent():
    tools = [search_ddgs, fetch_page]
    llm = select_model()

    summarization_middleware = SummarizationMiddleware(
        model=llm,
        max_tokens_before_summary=8000,
        messages_to_keep=10,
    )

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=CUSTOM_SYSTEM_PROMPT,
        checkpointer=st.session_state["checkpointer"],
        middleware=[summarization_middleware],
        debug=True,
    )

    return agent


def main():
    init_page()
    init_messages()
    web_browsing_agent = create_web_browsing_agent()

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input(placeholder="2025 한국시리즈 우승팀?"):
        st.chat_message("user").write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            handler = StreamlitLanggraphHandler(
                container=st.container(),
                expand_new_thoughts=True,
                max_thought_containers=4,
            )

            response = handler.invoke(
                agent=web_browsing_agent,
                input={"messages": [{"role": "user", "content": prompt}]},
                config={"configurable": {"thread_id": st.session_state["thread_id"]}},
            )

            if response:
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )


if __name__ == "__main__":
    main()
