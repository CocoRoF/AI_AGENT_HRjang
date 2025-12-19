import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

# .env 파일에 저장된 API KEY 등을 자동으로 불러오는 부분
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    import warnings
    warnings.warn("dotenv not found. Please set environment variables manually.", ImportWarning)


def main():
    # 웹페이지 기본 설정
    st.set_page_config(
        page_title="My Great ChatGPT",
        page_icon="🤗"
    )
    st.header("My Great ChatGPT 🤗")

    # 채팅 이력 초기화
    if "message_history" not in st.session_state:
        st.session_state.message_history = []

    # 사용할 LLM 모델 설정
    llm = ChatOpenAI(temperature=0)

    # Prompt 템플릿
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 친절하고 유용한 도움을 주는 어시스턴트입니다."),
        MessagesPlaceholder(variable_name="history"),
        ("user", "{user_input}")
    ])

    output_parser = StrOutputParser()

    # LCEL 체인 구성
    chain = prompt | llm | output_parser

    # 입력 UI를 Form + TextArea 방식으로 변경
    container = st.container()
    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_area(label='Message: ', height=100)
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
        # 내용을 입력하고 Submit 버튼이 눌리면 실행된다
            with st.spinner("ChatGPT가 답변 중 ..."):
                response = chain.invoke({
                    "history": st.session_state.message_history,
                    "user_input": user_input
                })

            # 히스토리에 저장
            st.session_state.message_history.append(("user", user_input))
            st.session_state.message_history.append(("assistant", response))

    # 과거 대화 출력
    for role, message in st.session_state.get("message_history", []):
        st.chat_message(role).markdown(message)


if __name__ == '__main__':
    main()
