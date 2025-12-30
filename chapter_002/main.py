import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

###### dotenv를 사용하지 않는 경우 삭제하세요 ######
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


def main():
    st.set_page_config(page_title="My Great ChatGPT", page_icon="🤗")
    st.header("My Great ChatGPT 🤗")

    # 챗 히스토리 초기화: message_history가 없다면 생성
    if "message_history" not in st.session_state:
        st.session_state.message_history = [
            # System Prompt 설정 ('system'은 System Prompt를 의미)
            ("system", "당신은 친절하고 유용한 도움을 주는 어시스턴트입니다.")
        ]

    # ChatGPT에 질문을 보내고 답변을 받아 파싱하는 처리 생성 (1.-4.의 처리)
    # 1. ChatGPT 모델 설정
    #    (기본적으로 GPT-3.5 Turbo가 호출됨)
    llm = ChatOpenAI(temperature=0)

    # 2. 사용자 입력을 받아 ChatGPT에 전달하는 템플릿 생성
    #    템플릿에는 과거 챗 히스토리를 포함하도록 설정
    prompt = ChatPromptTemplate.from_messages(
        [
            *st.session_state.message_history,
            ("user", "{user_input}"),  # 여기에 나중에 사용자 입력이 들어감
        ]
    )

    # 3. ChatGPT의 응답을 파싱하는 처리 호출
    output_parser = StrOutputParser()

    # 4. 사용자 입력을 ChatGPT에 전달하고 응답을 가져오는 연속 처리(chain)를 생성
    #    각 요소를 | (파이프)로 연결해 연속 처리를 만드는 것이 LCEL의 특징
    chain = prompt | llm | output_parser

    # 사용자 입력 감시
    if user_input := st.chat_input("궁금한 것을 입력해주세요."):
        with st.spinner("ChatGPT가 답변 중 ..."):
            response = chain.invoke({"user_input": user_input})

        # 사용자의 질문을 히스토리에 추가 ('user'는 사용자 질문 의미)
        st.session_state.message_history.append(("user", user_input))

        # ChatGPT의 답변을 히스토리에 추가 ('assistant'는 ChatGPT 답변 의미)
        st.session_state.message_history.append(("ai", response))

    # 챗 히스토리 출력
    for role, message in st.session_state.get("message_history", []):
        st.chat_message(role).markdown(message)


if __name__ == "__main__":
    main()
