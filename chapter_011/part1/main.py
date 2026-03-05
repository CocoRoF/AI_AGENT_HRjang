import re

import streamlit as st
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

# models
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langsmith import uuid7

# custom tools
from src.code_interpreter import CodeInterpreterClient
from tools.code_interpreter import code_interpreter_tool, set_client


@st.cache_data
def load_system_prompt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def csv_upload():
    with st.form("my-form", clear_on_submit=True):
        file = st.file_uploader(label="Upload your CSV here😇", type="csv")
        submitted = st.form_submit_button("Upload CSV")
        if submitted and file is not None:
            if not file.name in st.session_state.uploaded_files:
                uploaded_filename, uploaded_filepath = (
                    st.session_state.code_interpreter_client.upload_file(
                        file.read(), file.name
                    )
                )
                st.session_state.custom_system_prompt += f"\n업로드한 파일명: {uploaded_filename}\n (Code Interpreter Sandbox path: {uploaded_filepath})\n"
                st.session_state.uploaded_files.append(file.name)
        else:
            st.write("데이터 분석하고 싶은 파일을 업로드해줘")

    if st.session_state.uploaded_files:
        st.sidebar.markdown("## Uploaded files:")
        for file_name in st.session_state.uploaded_files:
            st.sidebar.markdown(f"- {file_name}")


def init_page():
    st.set_page_config(page_title="Data Analysis Agent", page_icon="🤗")
    st.header("Data Analysis Agent 🤗", divider="rainbow")
    st.sidebar.title("Options")

    # 세션 초기화
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = []
        # 대화가 리셋될 때 Code Interpreter 세션 재시작
        st.session_state.code_interpreter_client = CodeInterpreterClient()
        set_client(st.session_state.code_interpreter_client)
        st.session_state["checkpointer"] = InMemorySaver()
        st.session_state["thread_id"] = str(uuid7())
        st.session_state.custom_system_prompt = load_system_prompt(
            "./prompt/system_prompt.txt"
        )
        st.session_state.uploaded_files = []


def select_model():
    models = ("GPT-5.2", "Claude Sonnet 4.5", "Gemini 2.5 Flash")
    model = st.sidebar.radio("Choose a model:", models)
    if model == "GPT-5.2":
        return ChatOpenAI(temperature=0, model="gpt-5.2")
    elif model == "Claude Sonnet 4.5":
        return ChatAnthropic(temperature=0, model="claude-sonnet-4-5-20250929")
    elif model == "Gemini 2.5 Flash":
        return ChatGoogleGenerativeAI(temperature=0, model="gemini-2.5-flash")


def create_data_analysis_agent():
    tools = [code_interpreter_tool]
    llm = select_model()

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=st.session_state.custom_system_prompt,
        checkpointer=st.session_state["checkpointer"],
        debug=True,
    )

    return agent


def main():
    init_page()
    csv_upload()
    data_analysis_agent = create_data_analysis_agent()
    config = {"configurable": {"thread_id": st.session_state["thread_id"]}}

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if prompt := st.chat_input(placeholder="분석하고 싶은 내용을 입력해주세요."):
        st.chat_message("user").write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("분석 중..."):
                result = data_analysis_agent.invoke(
                    {"messages": [("user", prompt)]}, config
                )
            answer = result["messages"][-1].content
            st.write(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
