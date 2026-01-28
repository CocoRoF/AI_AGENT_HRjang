import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# models
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

###### dotenv를 사용하지 않는 경우에는 삭제해 주세요 ######
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


def init_page():
    st.set_page_config(page_title="Ask My PDF(s)", page_icon="🧐")
    st.sidebar.title("Options")


def select_model(temperature=0):
    models = ("GPT-5.2", "GPT-4", "Claude 3.5 Sonnet", "Gemini 1.5 Pro")
    model = st.sidebar.radio("Choose a model:", models)
    if model == "GPT-5.2":
        return ChatOpenAI(temperature=temperature, model_name="gpt-5.2")
    elif model == "GPT-4":
        return ChatOpenAI(temperature=temperature, model_name="gpt-4o")
    elif model == "Claude 3.5 Sonnet":
        return ChatAnthropic(
            temperature=temperature, model_name="claude-3-5-sonnet-20240620"
        )
    elif model == "Gemini 1.5 Pro":
        return ChatGoogleGenerativeAI(
            temperature=temperature, model="gemini-1.5-pro-latest"
        )


def init_qa_chain():
    llm = select_model()  # select_model은 앞 장과 동일
    prompt = ChatPromptTemplate.from_template(
        """
    다음의 배경 지식을 바탕으로 사용자의 질문에 답해 주세요.

    ===
    배경 지식
    {context}

    ===
    사용자의 질문
    {question}
    """
    )
    retriever = st.session_state.vectorstore.as_retriever(
        # "mmr", "similarity_score_threshold" 도 있음
        search_type="similarity",
        # 문서를 몇 개 가져올지 설정 (기본값: 4)
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

    query = st.text_input("PDF에 관한 질문을 적어주세요: ", key="input")
    if query:
        st.markdown("## Answer")
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
