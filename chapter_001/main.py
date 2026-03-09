import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser


def main():
    # 웹페이지 기본 설정
    st.set_page_config(page_title="My Great ChatGPT", page_icon="🤗")
    st.header("My Great ChatGPT 🤗")

    # 1. 채팅 이력 초기화
    if "message_history" not in st.session_state:
        st.session_state.message_history = []

    # 2. LLM 모델 설정 (기본값: gpt-4o-mini)
    llm = ChatOpenAI(temperature=0)

    # 3. LLM에 전달할 프롬프트 템플릿 정의
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "당신은 친절하고 유용한 도움을 주는 어시스턴트입니다."),
            MessagesPlaceholder(variable_name="history"),
            ("user", "{user_input}"),
        ]
    )

    # LLM 응답을 텍스트로 변환해주는 파서
    output_parser = StrOutputParser()

    # 4. 사용자 질문을 ChatGPT로 전달해 응답을 받는 체인을 생성
    # 각 요소를 | (파이프)로 연결해서 연속적인 처리를 만드는 것이 LCEL의 특징
    chain = prompt | llm | output_parser

    # 5. 사용자 입력 처리
    if user_input := st.chat_input("궁금한 것을 입력해주세요."):
        # 입력을 받으면 이 부분이 실행
        with st.spinner("ChatGPT가 답변 중 ..."):
            # invoke 실행 시 히스토리와 사용자 입력을 프롬프트에 전달
            response = chain.invoke(
                {"history": st.session_state.message_history, "user_input": user_input}
            )

        # 사용자의 질문을 이력에 추가 ('user'는 사용자의 질문을 의미한다).
        st.session_state.message_history.append(("user", user_input))

        # ChatGPT의 답변을 이력에 추가 ('assistant'는 ChatGPT의 답변을 의미한다)
        st.session_state.message_history.append(("assistant", response))

    # 6. 대화 이력 출력
    for role, message in st.session_state.get("message_history", []):
        st.chat_message(role).markdown(message)


if __name__ == "__main__":
    main()
