# GitHub: https://github.com/naotaka1128/llm_app_codes/chapter_009/main.py
import streamlit as st
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

# models
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

# custom tools
from tools.search_ddg import search_ddg
from tools.fetch_page import fetch_page


CUSTOM_SYSTEM_PROMPT = """
당신은 사용자의 요청에 따라 인터넷에서 정보를 조사하는 어시스턴트입니다.
이미 알고 있는 정보에 의존하지 말고, 가능한 한 검색 도구를 사용해 정보를 수집한 뒤 답변하세요.
나무위키, 블로그, 커뮤니티 등 비공식 자료도 함께 참고할 수 있습니다.
(사용자가 특정 페이지를 지정한 경우에는 검색하지 않아도 됩니다.)

검색 결과만으로 정보가 부족하다고 판단되면,
- 검색 결과의 링크를 열어 실제 내용을 확인하고
- 필요하다면 검색어 또는 검색 언어를 변경해 다시 검색하세요.
단, 한 페이지에서 3페이지 이상 스크롤하지 마세요.

사용자는 바쁘므로,
참고 링크만 나열하지 말고 **직접적인 결론과 답변**을 제시하세요.

답변 마지막에는 **참조한 페이지의 URL을 반드시 포함**하세요.
사용자가 질문한 언어로 답변하세요.
"""


def init_page():
    st.set_page_config(page_title="Web Browsing Agent", page_icon="🤗")
    st.header("Web Browsing Agent 🤗")
    st.sidebar.title("Options")


def init_messages():
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "memory" not in st.session_state:
        st.session_state["memory"] = MemorySaver()
        st.session_state["thread_id"] = "default_thread"


def select_model():
    models = ("GPT-5.2", "Claude Sonnet 4.5", "Gemini 2.5 Flash")
    model = st.sidebar.radio("Choose a model:", models)
    if model == "GPT-5.2":
        return ChatOpenAI(temperature=0, model="gpt-5.2")
    elif model == "Claude Sonnet 4.5":
        return ChatAnthropic(temperature=0, model="claude-sonnet-4-5-20250929")
    elif model == "Gemini 2.5 Flash":
        return ChatGoogleGenerativeAI(temperature=0, model="gemini-2.5-flash")


def create_web_agent():
    tools = [search_ddg, fetch_page]
    llm = select_model()
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=CUSTOM_SYSTEM_PROMPT,
        checkpointer=st.session_state["memory"],
    )
    return agent


def main():
    init_page()
    init_messages()
    web_browsing_agent = create_web_agent()

    # 이전 메시지 표시
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input(placeholder="2025 한국시리즈 우승팀?"):
        # 사용자 메시지 저장 및 표시
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            config = {"configurable": {"thread_id": st.session_state["thread_id"]}}

            # 에이전트 실행
            response = web_browsing_agent.invoke(
                {"messages": [{"role": "user", "content": prompt}]}, config
            )

            # 응답 출력
            last_message = response["messages"][-1]
            st.write(last_message.content)

            # AI 응답 저장
            st.session_state.messages.append(
                {"role": "assistant", "content": last_message.content}
            )


if __name__ == "__main__":
    main()
