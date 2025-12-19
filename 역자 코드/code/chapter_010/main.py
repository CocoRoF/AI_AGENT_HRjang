

import streamlit as st
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain.agents import create_tool_calling_agent, AgentExecutor

from langchain.agents import AgentExecutor
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain.memory import ConversationBufferWindowMemory

# models
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

# custom tools
from tools.fetch_qa_content import fetch_qa_content
from tools.fetch_stores_by_prefecture import fetch_stores_by_prefecture


###### dotenv를 사용하지 않는 경우에는 삭제해 주세요 ######
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    import warnings
    warnings.warn("dotenv not found. Please make sure to set your environment variables manually.", ImportWarning)
################################################

CUSTOM_SYSTEM_PROMPT = """
당신은 한국의 저가 이동통신사 '베어모바일'의 고객지원(CS) 담당자입니다.
고객의 문의에 성실하고 정확한 답변을 제공해 주세요.

이동통신사의 CS로서 자사 서비스나 휴대전화의 일반적인 지식만 답변합니다.
그 외 주제의 질문에는 정중히 답변을 사양해 주세요.

답변의 정확성을 보장하기 위해 '베어모바일'에 관한 질문을 받았을 때는
반드시 도구를 사용해서 답변을 찾아야 합니다.

고객이 질문에 사용한 언어로 답변해 주세요
예를 들어서 고객이 영어로 질문했다면 반드시 영어로,
스페인어로 질문했다면 반드시 스페인어로 답변해 주세요.

답변을 할 때에 불분명한 점이 있으면 고객에게 확인해 주세요.
이를 통해 고객의 의도를 파악하고 적절한 답변을 제공할 수 있습니다.

예를 들어 사용자가 '지점은 어디에 있나요?'라고 질문했을 경우에
먼저 사용자의 거주 지역을 물어보세요.

전국의 지점을 알고 싶어하는 사용자는 거의 없습니다.
대부분은 자신이 사는 지역의 지점을 알고 싶어합니다.
따라서 한국의 전 지점을 검색해서 답변하지 말고
고객의 의도를 충분히 이해할 때까지 답변하지 마세요!

이것은 어디까지나 한 예시일 뿐입니다.
그 외의 경우에도 반드시 고객의 의도를 파악하고 적절한 답변을 해 주세요.
"""


def init_page():
    st.set_page_config(
        page_title="고객지원",
        page_icon="🐻"
    )
    st.header("고객지원🐻")
    st.sidebar.title("Options")


def init_messages():
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        welcome_message = "베어모바일 고객지원에 오신 것을 환영합니다. 질문해 주세요🐻"
        st.session_state.messages = [
            {"role": "assistant", "content": welcome_message}
        ]
        st.session_state['memory'] = ConversationBufferWindowMemory(
            return_messages=True,
            memory_key="chat_history",
            k=10
        )


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
    ## https://learn.deeplearning.ai/functions-tools-agents-langchain/lesson/7/conversational-agent
    tools = [fetch_qa_content, fetch_stores_by_prefecture]
    prompt = ChatPromptTemplate.from_messages([
        ("system", CUSTOM_SYSTEM_PROMPT),
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


def main():
    init_page()
    init_messages()
    customer_support_agent = create_agent()

    for msg in st.session_state['memory'].chat_memory.messages:
        st.chat_message(msg.type).write(msg.content)

    if prompt := st.chat_input(placeholder="법인 명의로 계약할 수 있나요?"):
        st.chat_message("user").write(prompt)

        with st.chat_message("assistant"):
            st_cb = StreamlitCallbackHandler(
                st.container(), expand_new_thoughts=True)
            response = customer_support_agent.invoke(
                {'input': prompt},
                config=RunnableConfig({'callbacks': [st_cb]})
            )
            st.write(response["output"])


if __name__ == '__main__':
    main()
