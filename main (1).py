import streamlit as st
from langchain.agents import create_agent
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.callbacks import StreamlitCallbackHandler

# models
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

# custom tools
from tools.search_ddg import search_ddg
from tools.fetch_page import fetch_page

###### dotenv를 사용하지 않는 경우는 삭제하세요 ######
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
당신은 사용자의 요청에 따라 인터넷에서 정보를 조사하는 어시스턴트입니다.
사용 가능한 도구를 활용하여 조사한 정보를 설명해주세요.
이미 알고 있는 정보만으로 답변하지 말고, 가능한 한 검색을 수행한 뒤 답변해주세요.
(사용자가 읽을 페이지를 지정하는 등 특별한 경우는 검색하지 않아도 됩니다.)

검색 결과 페이지만 확인했을 때 정보가 충분하지 않다고 판단되면 다음 옵션을 고려해 시도해 주세요.

- 검색 결과의 링크를 클릭해 각 페이지의 콘텐츠를 열람하고 내용을 확인하세요.
- 한 페이지가 너무 길 경우, 3페이지 이상 스크롤하지 마세요 (메모리 부담 때문).
- 검색 쿼리를 변경한 뒤 다시 검색을 시도하세요.
- 공식 문서뿐 아니라 블로그, 커뮤니티 등 비공식 자료도 함께 참고하세요.

사용자는 매우 바쁘며, 당신만큼 여유롭지 않습니다.
따라서 사용자의 수고를 덜어주기 위해 **직접적인 답변**을 제공해주세요.

=== 나쁜 답변 예시 ===
- 다음 페이지들을 참고하세요.
- 이 페이지들을 보고 코드를 작성할 수 있습니다.
- 다음 페이지가 도움이 될 것입니다.

=== 좋은 답변 예시 ===
- 이 문제의 해결 예시는 다음과 같습니다. -- 여기 코드 제시 --
- 질문에 대한 답은 다음과 같습니다. -- 여기 답변 제시 --

답변 마지막에는 **참조한 페이지의 URL을 반드시 기재**해주세요.
(사용자가 정보를 검증할 수 있도록)

사용자가 사용하는 언어로 답변해주세요.
사용자가 한국어로 질문하면 한국어로, 스페인어로 질문하면 스페인어로 답변해야 합니다.
"""

# 대화 기록 윈도우 크기 (최근 k개 대화만 유지)
MEMORY_WINDOW_SIZE = 10


def init_page():
    st.set_page_config(page_title="Web Browsing Agent", page_icon="🤗")
    st.header("Web Browsing Agent 🤗")
    st.sidebar.title("Options")


def init_messages():
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "chat_history" not in st.session_state:
        st.session_state.chat_history = InMemoryChatMessageHistory()
        st.session_state.messages = [
            {"role": "assistant", "content": "안녕하세요! 무엇이든 질문해주세요!"}
        ]


def get_chat_history_messages():
    """윈도우 크기만큼의 최근 대화 기록 반환"""
    messages = st.session_state.chat_history.messages
    # 최근 k*2개 메시지만 유지 (user + assistant 쌍)
    return messages[-(MEMORY_WINDOW_SIZE * 2):]


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
    """LangChain 1.0+ create_agent 함수를 사용한 에이전트 생성"""
    tools = [search_ddg, fetch_page]
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", CUSTOM_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    llm = select_model()
    
    # LangChain 1.0+: create_agent 단일 함수로 에이전트 생성
    return create_agent(
        llm,
        tools=tools,
        prompt=prompt,
        verbose=True
    )


def main():
    init_page()
    init_messages()
    web_browsing_agent = create_web_agent()

    # 저장된 메시지 표시
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if user_input := st.chat_input(placeholder="2025 한국시리즈 우승팀?"):
        st.chat_message("user").write(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            # 콜백 함수 설정 (에이전트 동작 시각화용)
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)

            # 에이전트 실행 (chat_history를 직접 전달)
            response = web_browsing_agent.invoke(
                {
                    "input": user_input,
                    "chat_history": get_chat_history_messages()
                },
                config=RunnableConfig({"callbacks": [st_cb]})
            )
            
            output = response["output"]
            st.write(output)
            
            # 대화 기록 저장
            st.session_state.chat_history.add_user_message(user_input)
            st.session_state.chat_history.add_ai_message(output)
            st.session_state.messages.append({"role": "assistant", "content": output})


if __name__ == "__main__":
    main()
