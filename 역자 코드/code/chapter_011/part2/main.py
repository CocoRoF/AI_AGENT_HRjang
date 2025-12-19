
import re
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
from src.code_interpreter import CodeInterpreterClient
from tools.code_interpreter import code_interpreter_tool
from tools.bigquery import BigQueryClient

###### dotenv를 사용하지 않는 경우에는 삭제해 주세요 ######
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    import warnings
    warnings.warn("dotenv not found. Please make sure to set your environment variables manually.", ImportWarning)
################################################


@st.cache_data  # 캐시를 활용하도록 변경
def load_system_prompt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def init_page():
    st.set_page_config(
        page_title="Data Analysis Agent",
        page_icon="🤗"
    )
    st.header("Data Analysis Agent 🤗", divider='rainbow')
    st.sidebar.title("Options")

	# message 초기화  / python runtime 초기화 
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = []
		# 대화가 리셋될 때 Code Interpreter의 세션도 다시 생성
        st.session_state.code_interpreter_client = CodeInterpreterClient()
        st.session_state['memory'] = ConversationBufferWindowMemory(
            return_messages=True,
            memory_key="chat_history",
            k=10
        )
        st.session_state.custom_system_prompt = load_system_prompt(
            "./prompt/system_prompt.txt")
        st.session_state.uploaded_files = []


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


def create_agent(bq_client):
    tools = [
        bq_client.get_table_info_tool(),
        bq_client.exec_query_tool(),
        code_interpreter_tool
    ]
    prompt = ChatPromptTemplate.from_messages([
        ("system", st.session_state.custom_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    llm = select_model()
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        memory=st.session_state['memory']
    )


def parse_response(response):
    """
	response에서 text와 image_paths를 추출

	response의 예시
    ===    
	비트코인의 종가 차트를 작성했습니다. 아래 이미지에서 확인하실 수 있습니다
    <img src="./files/file-s4W0rog1pjneOAtWeq21lbDy.png" alt="Bitcoin Closing Price Chart">

	image_path를 추출한 후에는 img 태그를 삭제해 둔다
    """
	# img 태그를 찾기 위한 정규표현식 패턴
    img_pattern = re.compile(r'<img\s+[^>]*?src="([^"]+)"[^>]*?>')

	# img 태그를 검색해서 image_paths를 추출
    image_paths = img_pattern.findall(response)
    image_paths = [path.replace("/mnt/data/", "./files/") for path in image_paths]	# img 태그를 삭제하고 텍스트를 추출
    text = img_pattern.sub('', response).strip()

    return text, image_paths


def display_content(content):
    text, image_paths = parse_response(content)
    st.write(text)
    for image_path in image_paths:
        st.image(image_path, caption="")


def main():
    init_page()
    bq_client = BigQueryClient(st.session_state.code_interpreter_client)
    data_analysis_agent = create_agent(bq_client)

    for msg in st.session_state['memory'].chat_memory.messages:
        with st.chat_message(msg.type):
            display_content(msg.content)

    if prompt := st.chat_input(placeholder="분석하고 싶은 내용을 입력해 주세요"):
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(
                st.container(), expand_new_thoughts=True)
            response = data_analysis_agent.invoke(
                {'input': prompt},
                config=RunnableConfig({'callbacks': [st_cb]})
            )
            display_content(response["output"])


if __name__ == '__main__':
    main()
