import re
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
from src.code_interpreter import CodeInterpreterClient
from tools.code_interpreter import code_interpreter_tool


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
                assistant_api_file_id = (
                    st.session_state.code_interpreter_client.upload_file(file.read())
                )
                st.session_state.custom_system_prompt += f"\업로드한 파일명: {file.name} (Code Interpreter에서의 path: /mnt/data/{assistant_api_file_id})\n"
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

    # 메시지 초기화 / python runtime 초기화
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = []
        # 대화가 리셋될 때 Code Interpreter의 세션도 다시 생성
        st.session_state.code_interpreter_client = CodeInterpreterClient()
        st.session_state["memory"] = ConversationBufferWindowMemory(
            return_messages=True, memory_key="chat_history", k=10
        )
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


def create_agent():
    # tools 이외의 부분은 이전 장과 완전히 동일
    tools = [code_interpreter_tool]
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", st.session_state.custom_system_prompt),
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


def parse_response(response):
    """
    response에서 text와 image_paths를 가져옵니다

    response의 예
    ===
    비트코인의 종가 차트를 생성했습니다. 아래 이미지에서 확인할 수 있습니다.
    <img src="./files/file-s4W0rog1pjneOAtWeq21lbDy.png" alt="Bitcoin Closing Price Chart">

    image_path를 가져온 후에는 img 태그를 삭제해둡니다
    """
    # img 태그를 가져오기 위한 정규표현식 패턴
    img_pattern = re.compile(r'<img\s+[^>]*?src="([^"]+)"[^>]*?>')

    # img 태그를 검색하여 image_paths를 가져옴
    image_paths = img_pattern.findall(response)

    # img 태그를 삭제하여 텍스트를 가져옴
    text = img_pattern.sub("", response).strip()

    return text, image_paths


def display_content(content):
    text, image_paths = parse_response(content)
    st.write(text)
    for image_path in image_paths:
        st.image(image_path, caption="")


def main():
    init_page()
    csv_upload()
    data_analysis_agent = create_agent()

    for msg in st.session_state["memory"].chat_memory.messages:
        with st.chat_message(msg.type):
            display_content(msg.content)

    if prompt := st.chat_input(placeholder="분석하고 싶은 내용을 입력해주세요."):
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
            response = data_analysis_agent.invoke(
                {"input": prompt}, config=RunnableConfig({"callbacks": [st_cb]})
            )
            display_content(response["output"])


if __name__ == "__main__":
    main()
