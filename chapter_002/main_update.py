# 필요한 라이브러리 호출
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

    warnings.warn(
        "dotenv not found. Please set environment variables manually.", ImportWarning
    )


def main():
    # 웹페이지 기본 설정
    st.set_page_config(page_title="My Great ChatGPT", page_icon="🤗")
    st.header("My Great ChatGPT 🤗")

    # 채팅 이력 초기화: message_history가 없다면 새로 생성
    if "message_history" not in st.session_state:
        st.session_state.message_history = []

    # 2️. 사용할 LLM 모델 설정
    llm = ChatOpenAI(temperature=0)

    # 3️. LLM Prompt 템플릿 정의
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "당신은 친절하고 유용한 도움을 주는 어시스턴트입니다."),
            MessagesPlaceholder(variable_name="history"),
            ("user", "{user_input}"),
        ]
    )

    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser

    # 5️. Streamlit 전용 입력 창
    if user_input := st.chat_input("궁금한 것을 입력해주세요."):
        with st.spinner("ChatGPT가 답변 중 ..."):
            response = chain.invoke(
                {
                    "history": st.session_state.message_history,  # dict 형식 전달
                    "user_input": user_input,
                }
            )

        # 대화 히스토리 기록 (dict 구조)
        st.session_state.message_history.append({"role": "user", "content": user_input})
        st.session_state.message_history.append(
            {"role": "assistant", "content": response}
        )

    # UI에 과거 대화 모두 출력
    for msg in st.session_state.get("message_history", []):
        st.chat_message(msg["role"]).markdown(msg["content"])


if __name__ == "__main__":
    main()
