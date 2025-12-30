import streamlit as st
from langchain_core.tracers.context import collect_runs
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

# cache / feedback
from src.cache import Cache
from src.feedback import add_feedback

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
            "영진모바일 고객지원센터에 오신 것을 환영합니다. 무엇이든 질문해주세요🐻"
        )
        st.session_state.messages = [{"role": "assistant", "content": welcome_message}]
        st.session_state["memory"] = ConversationBufferWindowMemory(
            return_messages=True, memory_key="chat_history", k=10
        )

    if len(st.session_state.messages) == 1:  # welcome message만 있는 경우
        st.session_state["first_question"] = True
    else:
        st.session_state["first_question"] = False


def select_model():
    models = ("GPT-5.1", "Claude Sonnet 4.5", "Gemini 2.5 Flash")
    model = st.sidebar.radio("Choose a model:", models)
    if model == "GPT-5.1":
        return ChatOpenAI(temperature=0, model="gpt-5.1")
    elif model == "Claude Sonnet 4.5":
        return ChatAnthropic(temperature=0, model="claude-sonnet-4-5-20250929")
    elif model == "Gemini 2.5 Flash":
        return ChatGoogleGenerativeAI(temperature=0, model="gemini-2.5-flash")


def create_agent():
    tools = [fetch_qa_content, fetch_stores_by_prefecture]
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

    if prompt := st.chat_input(placeholder="법인으로 계약할 수 있어?"):
        st.chat_message("user").write(prompt)

        # 첫 번째 질문인 경우 캐시 확인
        if st.session_state["first_question"]:
            if cache_content := cache.search(query=prompt):
                with st.chat_message("assistant"):
                    st.write(f"(cache) {cache_content}")
                st.session_state.messages.append(
                    {"role": "assistant", "content": cache_content}
                )
                st.stop()  # 캐시 내용을 출력했다면 실행 종료

        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)

            with collect_runs() as cb:  # 수정된 부분
                response = customer_support_agent.invoke(
                    {"input": prompt}, config=RunnableConfig({"callbacks": [st_cb]})
                )
                st.session_state.run_id = cb.traced_runs[0].id
                st.write(response["output"])

        # 첫 번째 질문이면 캐시에 저장
        if st.session_state["first_question"]:
            cache.save(prompt, response["output"])

    if st.session_state.get("run_id"):
        add_feedback()


if __name__ == "__main__":
    main()
