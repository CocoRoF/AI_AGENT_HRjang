import streamlit as st
from langsmith import uuid7

from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver

# models
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

# custom tools
from tools.fetch_qa_content import fetch_qa_content
from tools.fetch_stores_by_prefecture import fetch_stores_by_prefecture


CUSTOM_SYSTEM_PROMPT = """
당신은 한국의 저가 통신사 '영진모바일'의 고객센터 상담원입니다.
고객의 문의에 성실하고 정확하게 답변해 주세요.

통신사 고객센터 상담원으로서, 회사의 서비스와 휴대전화에 관한 정보에만 답변하세요.
그 외 주제의 질문에는 정중하게 거절해 주세요.

답변의 정확성을 위해, '영진모바일'에 대한 질문을 받으면
반드시 툴을 사용해 답을 찾아주세요.

고객이 사용한 언어로 답변해 주세요.
예를 들어 영어로 질문하면 영어로, 스페인어로 질문하면 스페인어로 답변합니다.

불분명한 부분이 있다면 반드시 고객에게 먼저 확인해 주세요.
고객의 의도를 정확히 파악해야 올바른 답변을 제공할 수 있습니다.

예를 들어 고객이 "매장은 어디에 있나요?"라고 질문한 경우,
먼저 거주 지역(시/도)을 물어보세요.
전국 매장을 알고 싶은 고객은 거의 없습니다.
고객의 의도를 파악하기 전까지 섣불리 답변하지 마세요.

위는 한 가지 예시일 뿐입니다.
다른 경우에도 항상 고객의 의도를 먼저 파악한 뒤 답변해 주세요.
"""


def init_page():
    st.set_page_config(page_title="고객 지원", page_icon="🐻")
    st.header("고객 지원🐻")
    st.sidebar.title("옵션")


def init_messages():
    clear_button = st.sidebar.button("대화 초기화", key="clear")
    if clear_button or "messages" not in st.session_state:
        welcome_message = (
            "영진모바일 고객지원에 오신 것을 환영합니다. 질문을 입력해 주세요🐻"
        )
        st.session_state.messages = [{"role": "assistant", "content": welcome_message}]
        st.session_state["checkpointer"] = InMemorySaver()
        st.session_state["thread_id"] = str(uuid7())


def select_model(temperature=0):
    models = ("GPT-5.2", "Claude Sonnet 4.5", "Gemini 2.5 Flash")
    model = st.sidebar.radio("사용할 모델 선택:", models)
    if model == "GPT-5.2":
        return ChatOpenAI(temperature=temperature, model="gpt-5.2")
    elif model == "Claude Sonnet 4.5":
        return ChatAnthropic(
            temperature=temperature, model="claude-sonnet-4-5-20250929"
        )
    elif model == "Gemini 2.5 Flash":
        return ChatGoogleGenerativeAI(temperature=temperature, model="gemini-2.5-flash")


def create_customer_support_agent():
    tools = [fetch_qa_content, fetch_stores_by_prefecture]
    llm = select_model()

    summarization_middleware = SummarizationMiddleware(
        model=llm,
        trigger=("tokens", 8000),
        keep=("messages", 10),
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
    agent = create_customer_support_agent()
    config = {"configurable": {"thread_id": st.session_state["thread_id"]}}

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input(placeholder="법인 명의로 계약이 가능한가요?"):
        st.chat_message("user").write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("답변 생성 중..."):
                result = agent.invoke({"messages": [("user", prompt)]}, config)
            answer = result["messages"][-1].content
            st.write(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
