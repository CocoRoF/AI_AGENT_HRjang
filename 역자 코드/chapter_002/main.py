
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

###### dotenv를 사용하지 않는 경우에는 삭제해 주세요 ######
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    import warnings
    warnings.warn("dotenv not found. Please make sure to set your environment variables    manually.", ImportWarning)
################################################


def main():
    st.set_page_config(
        page_title="My Great ChatGPT",
        page_icon="🤗"
    )
    st.header("My Great ChatGPT 🤗")

	#채팅 이력 초기화: message_history가 없다면 새로 생성
    if "message_history" not in st.session_state:
        st.session_state.message_history = [
			# System Prompt 를 설정 ('system' 은 System Prompt을 의미한다)
            ("system", "You are a helpful assistant.")
        ]


	# ChatGPT에게 질문을 하고 답변을 받아오는(파싱하는) 처리를 작성 (1~4번 처리)
    # 1. ChatGPT 모델을 호출하도록 설정
    #   (기본 설정으로는 GPT-3.5 Turbo가 호출된다.)

    llm = ChatOpenAI(temperature=0)


	# 2. 사용자의 질문을 받아서 ChatGPT에 전달하기 위한 템플릿을 작성
    #     템플릿에는 과거의 채팅 이력도 포함되도록 설정

    prompt = ChatPromptTemplate.from_messages([
        *st.session_state.message_history,
		("user", "{user_input}") # 이 곳에 나중에 사용자의 입력이 들어간다
    ])

	# 3. ChatGPT의 응답을 파싱하기 위한 처리를 호출
    output_parser = StrOutputParser()


	# 4. 사용자의 질문을 ChatGPT에 전달하고, 응답을 받아오는 연속적인 처리(체인)를 생성
    #    각 요소를 | (파이프)로 연결해서 연속적인 처리를 만드는 것이 LCEL의 특징
    chain = prompt | llm | output_parser

	# 사용자의 입력 모니터링
    if user_input := st.chat_input("궁금한 것을 입력해주세요."):
	    # 입력을 받으면 이 부분이 실행됩니다
        with st.spinner("ChatGPT is typing ..."):
            response = chain.invoke({"user_input": user_input})

		# 사용자의 질문을 이력에 추가 ('user'는 사용자의 질문을 의미한다).
        st.session_state.message_history.append(("user", user_input))

		# ChatGPT의 답변을 이력에 추가 ('assistant'는 ChatGPT의 답변을 의미한다)
        st.session_state.message_history.append(("ai", response))

	# 채팅 이력 표시
    for role, message in st.session_state.get("message_history", []):
        st.chat_message(role).markdown(message)


if __name__ == '__main__':
    main()
