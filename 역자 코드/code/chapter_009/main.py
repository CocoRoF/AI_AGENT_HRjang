
import streamlit as st
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_community.callbacks import StreamlitCallbackHandler

# models
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

# custom tools
from tools.search_ddg import search_ddg
from tools.fetch_page import fetch_page

###### dotenv를 사용하지 않는 경우에는 삭제해 주세요 ######
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    import warnings
    warnings.warn("dotenv not found. Please make sure to set your environment variables manually.", ImportWarning)
################################################

CUSTOM_SYSTEM_PROMPT = """
당신은 사용자의 요청에 따라 인터넷에서 정보를 조사하는 어시스턴트입니다
사용 가능한 도구를 활용해서 조사한 정보를 설명해 주세요
이미 알고 있는 것만을 사용해서 답변하지 말고 답변하기 전에 가능하다면 검색을 수행해 주세요.
(사용자가 읽을 페이지를 직접 지정하는 것 처럼 특별한 경우에는 검색할 필요없습니다.)

검색 결과 페이지만으로는 정보가 충분하지 않다고 판단되는 경우에는 다음의 두 가지 옵션을 고려해보세요.

검색 결과의 링크를 클릭해서 각 페이지의 콘텐츠에 접근해서 내용을 읽어 보세요.
한 페이지가 너무 긴 경우에는 3번 이상 가져오지 마세요 (메모리에 부담이 생길 수 있습니다).
검색어를 변경해서 새롭게 검색을 실행하세요.
검색할 내용에 따라 적절한 언어로 검색어를 변경해 주세요.
  예: 프로그래밍 관련 질문은 영어로 검색하는 것이 적절합니다

사용자는 매우 바쁘며 당신만큼 여유롭지 않습니다
그러므로 사용자의 수고를 덜기 위해 직접적인 답변을 제공해 주세요

=== 나쁜 답변의 예 ===
이 페이지들을 참고하세요
이 페이지들을 보면 코드를 작성할 수 있습니다
다음 페이지가 도움이 될 것입니다

=== 좋은 답변의 예 ===
다음은 샘플 코드입니다. — 여기에 샘플 코드 —
당신의 질문에 대한 답은 — 여기에 답변 —

답변 마지막에는 반드시 참고한 페이지의 URL을 기재해 주세요.
(사용자가 답변을 검증할 수 있도록 하기 위함입니다)

사용자가 사용하는 언어로 답변해 주세요
예를 들어, 사용자가 한글로 질문했다면 한글로, 스페인어로 질문했다면 스페인어로 답변해 주세요
"""


def init_page():
    st.set_page_config(
        page_title="Web Browsing Agent",
        page_icon="🤗"
    )
    st.header("Web Browsing Agent 🤗")
    st.sidebar.title("Options")


def init_messages():
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "안녕하세요! 무엇이든 질문해 주세요！"}
        ]
        st.session_state['memory'] = ConversationBufferWindowMemory(
            return_messages=True,
            memory_key="chat_history",
            k=10
        )

        # 이렇게도 쓸 수 있다
        # from langchain_community.chat_message_histories import StreamlitChatMessageHistory
        # msgs = StreamlitChatMessageHistory(key="special_app_key")
        # st.session_state['memory'] = ConversationBufferMemory(memory_key="history", chat_memory=msgs)


def select_model():
    models = ("GPT-4", "Claude 3.5 Sonnet", "Gemini 1.5 Pro", "GPT-3.5 (not recommended)")
    model = st.sidebar.radio("Choose a model:", models)
    if model == "GPT-3.5 (not recommended)":
        return ChatOpenAI(
            temperature=0, model_name="gpt-3.5-turbo")
    elif model == "GPT-4":
        return ChatOpenAI(
            temperature=0, model_name="gpt-4o")
    elif model == "Claude 3.5 Sonnet":
        return ChatAnthropic(
            temperature=0, model_name="claude-3-5-sonnet-20240620")
    elif model == "Gemini 1.5 Pro":
        return ChatGoogleGenerativeAI(
            temperature=0, model="gemini-1.5-pro-latest")


def create_agent():
    tools = [search_ddg, fetch_page]
    prompt = ChatPromptTemplate.from_messages([
        ("system", CUSTOM_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    llm = select_model()
    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        memory=st.session_state['memory']
    )


def main():
    init_page()
    init_messages()
    web_browsing_agent = create_agent()

    for msg in st.session_state['memory'].chat_memory.messages:
        st.chat_message(msg.type).write(msg.content)

    if prompt := st.chat_input(placeholder="2023 FIFA 여자 월드컵의 우승 국가는?"):
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            # 콜백 함수 설정 (에이전트 동작 시각화용)
            st_cb = StreamlitCallbackHandler(
                st.container(), expand_new_thoughts=True)

            # 에이전트 실행
            response = web_browsing_agent.invoke(
                {'input': prompt},
                config=RunnableConfig({'callbacks': [st_cb]})
            )
            st.write(response["output"])


if __name__ == '__main__':
    main()
