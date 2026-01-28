import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# models
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI


def init_page():
    st.set_page_config(page_title="Ask My PDF(s)", page_icon="🧐")
    st.sidebar.title("옵션")


def select_model(temperature=0):
    models = ("GPT-5 mini", "GPT-5.1", "Claude Sonnet 4.5", "Gemini 2.5 Flash")
    model = st.sidebar.radio("Choose a model:", models)
    if model == "GPT-5 mini":
        return ChatOpenAI(temperature=temperature, model="gpt-5-mini")
    elif model == "GPT-5.1":
        return ChatOpenAI(temperature=temperature, model="gpt-5.1")
    elif model == "Claude Sonnet 4.5":
        return ChatAnthropic(
            temperature=temperature, model="claude-sonnet-4-5-20250929"
        )
    elif model == "Gemini 2.5 Flash":
        return ChatGoogleGenerativeAI(temperature=temperature, model="gemini-2.5-flash")


def init_qa_chain():
    llm = select_model()
    prompt = ChatPromptTemplate.from_template(
        """
    다음 배경 지식을 사용해서 사용자 질문에 답변해 주세요.
    
    ===
    배경 지식
    {context}
    
    ===
    사용자 질문
    {question}
    """
    )

    retriever = st.session_state.vectorstore.as_retriever(
        # "mmr", "similarity_score_threshold" 등도 사용 가능
        search_type="similarity",
        # 몇 개의 문서를 가져올지 설정(기본값: 4)
        search_kwargs={"k": 10},
    )
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def page_ask_my_pdf():
    chain = init_qa_chain()

    if query := st.text_input("PDF에 대한 질문을 입력하세요: ", key="input"):
        st.markdown("## 답변")
        st.write_stream(chain.stream(query))


def main():
    init_page()
    st.title("PDF QA 🧐")
    if "vectorstore" not in st.session_state:
        st.warning("먼저 📄 Upload PDF(s)에서 PDF 파일을 업로드해 주세요")
    else:
        page_ask_my_pdf()


if __name__ == "__main__":
    main()
