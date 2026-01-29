import streamlit as st
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import MemorySaver

# Models
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

# Custom tools
from tools.search_ddg import search_ddg
from tools.fetch_page import fetch_page

import uuid

CUSTOM_SYSTEM_PROMPT = """
당신은 사용자의 요청에 따라 인터넷에서 정보를 조사해
현실적인 판단으로 결론을 도출하는 어시스턴트입니다.

반드시 검색 도구를 사용해 정보를 확인하고,
나무위키, 위키백과, 언론 기사 등 비공식 자료도 참고할 수 있습니다.

다음 원칙을 따르세요.

1. 신뢰 가능한 출처 1곳에서라도 사실이 명확히 확인되면
   해당 내용은 확정된 사실로 단정하여 답변하세요.

2. 스포츠 우승, 선거 결과 등
   이미 종료된 사건에 대해서는 "확인 불가"와 같은 표현을 사용하지 마세요.

3. 먼저 결론을 한 문장으로 제시하고, 그 뒤에 간단한 근거를 덧붙이세요.

답변 마지막에는 참고한 URL을 포함하세요.
사용자가 질문한 언어로 답변하세요.
"""


def init_page():
    st.set_page_config(
        page_title="Web Browsing Agent",
        page_icon="🤗",
    )
    st.header("Web Browsing Agent 🤗")
    st.sidebar.title("Options")


def init_messages():
    clear_button = st.sidebar.button("Clear Conversation", key="clear")

    if clear_button or "memory" not in st.session_state:
        st.session_state["memory"] = MemorySaver()
        st.session_state["thread_id"] = str(uuid.uuid4())


def select_model():
    models = (
        "GPT-5.2",
        "Claude Sonnet 4.5",
        "Gemini 2.5 Flash",
    )

    model = st.sidebar.radio("Choose a model:", models)

    if model == "GPT-5.2":
        return ChatOpenAI(
            model="gpt-5.2",
            temperature=0,
        )

    if model == "Claude Sonnet 4.5":
        return ChatAnthropic(
            model="claude-sonnet-4-5-20250929",
            temperature=0,
        )

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
    )


def create_web_agent():
    tools = [search_ddg, fetch_page]
    llm = select_model()

    # 대화 기록 자동 요약 미들웨어 설정
    # 토큰 한도에 도달하면 오래된 메시지를 자동으로 요약하여 컨텍스트 유지
    summarization_middleware = SummarizationMiddleware(
        model=llm,  # 요약에 사용할 LLM (에이전트와 동일 모델 사용)
        max_tokens_before_summary=8000,  # 이 토큰 수를 초과하면 요약 시작
        messages_to_keep=10,  # 최근 10개 메시지는 유지하고, 이전 대화는 요약한다
    )

    return create_agent(
        model=llm,
        tools=tools,
        system_prompt=CUSTOM_SYSTEM_PROMPT,
        checkpointer=st.session_state["memory"],  # 대화 상태 저장용 체크포인터
        middleware=[summarization_middleware],  # 대화 요약 미들웨어 적용
    )


def main():
    init_page()
    init_messages()

    web_browsing_agent = create_web_agent()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 이전 메시지 출력
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input(placeholder="2025 한국시리즈 우승팀?"):
        # 사용자 메시지 저장
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            config = {"configurable": {"thread_id": st.session_state["thread_id"]}}

            response = web_browsing_agent.invoke(
                {"messages": [{"role": "user", "content": prompt}]},
                config,
            )

            last_message = response["messages"][-1]
            st.write(last_message.content)

            st.session_state.messages.append(
                {"role": "assistant", "content": last_message.content}
            )


if __name__ == "__main__":
    main()
