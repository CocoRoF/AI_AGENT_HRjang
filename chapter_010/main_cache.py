import streamlit as st
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_community.callbacks import StreamlitCallbackHandler

# models
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

# custom tools
from tools.fetch_qa_content import fetch_qa_content
from tools.fetch_stores_by_prefecture import fetch_stores_by_prefecture
from src.cache import Cache

###### dotenv を利用しない場合は消してください ######   # 원문이므로 그대로
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


@st.cache_data  # 캐시를 사용하도록 변경
def load_system_prompt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def init_page():
    st.set_page_config(page_title="고객 지원", page_icon="🐻")
    st.header("고객 지원🐻")
    st.sidebar.title("Options")


def init_messages():
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        welcome_message = (
            "영진모바일 고객지원에 오신 것을 환영합니다. 질문을 입력해 주세요 🐻"
        )
        st.session_state.messages = [{"role": "assistant", "content": welcome_message}]
        st.session_state["memory"] = ConversationBufferWindowMemory(
            return_messages=True, memory_key="chat_history", k=10
        )

    if len(st.session_state.messages) == 1:  # 환영 메시지뿐인 경우
        st.session_state["first_question"] = True  # 추가 부분
    else:
        st.session_state["first_question"] = False  # 추가 부분


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
    ## https://learn.deeplearning.ai/functions-tools-agents-langchain/lesson/7/conversational-agent
    tools = [fetch_qa_content, fetch_stores_by_prefecture]
    # 캐시를 사용하도록 변경
    custom_system_prompt = load_system_prompt("./prompt/system_prompt.txt")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", custom_system_prompt),
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

    # 캐시 초기화
    cache = Cache()

    for msg in st.session_state["memory"].chat_memory.messages:
        st.chat_message(msg.type).write(msg.content)

    if prompt := st.chat_input(placeholder="법인 명의로 계약이 가능한가요?"):
        st.chat_message("user").write(prompt)

        # 처음 질문인 경우 캐시를 확인
        if st.session_state["first_question"]:
            if cache_content := cache.search(query=prompt):
                st.chat_message("assistant").write(f"(cache) {cache_content}")
                st.session_state.messages.append(
                    {"role": "assistant", "content": cache_content}
                )
                st.stop()  # 캐시 내용을 출력한 경우 실행 종료

        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
            response = customer_support_agent.invoke(
                {"input": prompt}, config=RunnableConfig({"callbacks": [st_cb]})
            )
            st.write(response["output"])

        # 처음 질문인 경우 캐시에 저장
        if st.session_state["first_question"]:
            cache.save(prompt, response["output"])


if __name__ == "__main__":
    main()
