# GitHub: https://github.com/naotaka1128/llm_app_codes/chapter_010/main.py

import streamlit as st
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor

from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_classic.memory import ConversationBufferWindowMemory

# models
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

# custom tools
from tools.fetch_qa_content import fetch_qa_content
from tools.fetch_stores_by_prefecture import fetch_stores_by_prefecture


###### dotenv 을 사용하지 않는 경우는 삭제해주세요 ######
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    import warnings

    warnings.warn(
        "dotenv not found. Please make sure to set your environment variables manually.",
        ImportWarning,
    )
################################################

CUSTOM_SYSTEM_PROMPT = """
당신은 일본의 저가 통신사 ‘영진모바일’의 고객센터(CS) 상담원입니다.
고객의 문의에 대해 성실하고 정확하게 답변해주세요.

통신사 CS로서, 회사의 서비스와 휴대전화에 관한 일반적인 정보에만 답변해야 합니다.
그 외의 주제에 관한 질문에는 정중하게 답변을 거절해주세요.

답변의 정확성을 보장하기 위해, ‘영진모바일’에 대한 질문을 받을 경우
반드시 툴을 사용해 답을 찾아주세요.

고객이 질문에 사용한 언어로 답변해주세요.
예를 들어 영어로 질문하면 영어로, 스페인어로 질문하면 스페인어로 답변해야 합니다.

답변 과정에서 불분명한 부분이 있다면 반드시 고객에게 확인해 주세요.
그렇게 해야 고객의 진짜 의도를 정확하게 파악하고 올바른 답변을 제공할 수 있습니다.

예를 들어 고객이 “매장은 어디에 있나요?”라고 질문한 경우,
먼저 고객이 거주하는 도도부현(지역)을 물어보세요.

일본 전국의 매장 위치를 알고 싶은 고객은 거의 없습니다.
고객은 자기 지역의 매장을 알고 싶은 것입니다.
따라서 전국 매장을 검색해 답변하는 일이 없도록 하며,
고객의 의도를 완전히 파악하기 전까지는 섣불리 답변하지 마세요!

위는 한 가지 예시에 불과합니다.
다른 경우에도 항상 고객의 의도를 파악하고 적절한 답변을 해주세요.
"""


def init_page():
    st.set_page_config(page_title="고객센터", page_icon="🐻")
    st.header("고객센터🐻")
    st.sidebar.title("옵션")


def init_messages():
    clear_button = st.sidebar.button("대화 초기화", key="clear")
    if clear_button or "messages" not in st.session_state:
        welcome_message = (
            "영진모바일 고객센터에 오신 것을 환영합니다. 무엇이든 문의해주세요🐻"
        )
        st.session_state.messages = [{"role": "assistant", "content": welcome_message}]
        st.session_state["memory"] = ConversationBufferWindowMemory(
            return_messages=True, memory_key="chat_history", k=10
        )


def select_model(temperature=0):
    models = ("GPT-5 mini", "GPT-5.2", "Claude Sonnet 4.5", "Gemini 2.5 Flash")
    model = st.sidebar.radio("사용할 모델 선택:", models)
    if model == "GPT-5 mini":
        return ChatOpenAI(temperature=temperature, model="gpt-5-mini")
    elif model == "GPT-5.2":
        return ChatOpenAI(temperature=temperature, model="gpt-5.2")
    elif model == "Claude Sonnet 4.5":
        return ChatAnthropic(
            temperature=temperature, model="claude-sonnet-4-5-20250929"
        )
    elif model == "Gemini 2.5 Flash":
        return ChatGoogleGenerativeAI(temperature=temperature, model="gemini-2.5-flash")


def create_agent():
    tools = [fetch_qa_content, fetch_stores_by_prefecture]
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CUSTOM_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    llm = select_model()
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent, tools=tools, verbose=True, memory=st.session_state["memory"]
    )


def main():
    init_page()
    init_messages()
    customer_support_agent = create_agent()

    for msg in st.session_state["memory"].chat_memory.messages:
        st.chat_message(msg.type).write(msg.content)

    if prompt := st.chat_input(placeholder="법인 명의로도 계약할 수 있어?"):
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
            response = customer_support_agent.invoke(
                {"input": prompt}, config=RunnableConfig({"callbacks": [st_cb]})
            )
            st.write(response["output"])


if __name__ == "__main__":
    main()
