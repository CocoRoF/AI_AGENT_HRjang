import fitz  # PyMuPDF
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

###### dotenv를 사용하지 않는 경우에는 삭제해 주세요 ######
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    import warnings
    warnings.warn("dotenv not found. Please make sure to set your environment variables manually.", ImportWarning)
################################################


def init_page():
    st.set_page_config(
        page_title="Upload PDF(s)",
        page_icon="📄"
    )
    st.sidebar.title("Options")


def init_messages():
    clear_button = st.sidebar.button("Clear DB", key="clear")
    if clear_button and "vectorstore" in st.session_state:
        del st.session_state.vectorstore


def get_pdf_text():
    # file_uploader로 PDF를 업로드 한다
    # (file_uploader의 자세한 설명은 6장을 참고하세요)
    pdf_file = st.file_uploader(
        label='Upload your PDF 😇',
        type='pdf'  # PDF 파일만 업로드 가능
    )
    if pdf_file:
        pdf_text = ""
        with st.spinner("Loading PDF ..."):
            # PyMuPDF로 PDF를 읽기
            # (자세한 설명은 라이브러리의 공식 페이지 등을 참고해 주세요)			
            pdf_doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
            for page in pdf_doc:
                pdf_text += page.get_text()

        # RecursiveCharacterTextSplitter를 이용해서 청크로 분리한다
        # (자세한 설명은 6장을 참고하세요)
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="text-embedding-3-small",
            # 적절한 청크 크기는 질문 대상인 PDF에 따라 달라지므로 조정이 필요
            # 너무 크게 하면 질문에 답할 때 여러 청크의 정보를 참조할 수 없음
            # 반대로 너무 작으면 하나의 청크에 충분한 크기의 문맥이 담기지 않음
            chunk_size=500,
            chunk_overlap=0,
        )
        return text_splitter.split_text(pdf_text)
    else:
        return None


def build_vector_store(pdf_text):
    with st.spinner("Saving to vector store ..."):
        if 'vectorstore' in st.session_state:
            st.session_state.vectorstore.add_texts(pdf_text)
        else:
            # 벡터 DB의 초기화와 문서 추가를 동시에 수행
            # LangChain의 Document Loader를 사용하는 경우에는 from_documents를 사용
            st.session_state.vectorstore = FAISS.from_texts(
                pdf_text,
                OpenAIEmbeddings(model="text-embedding-3-small")
            )

            # FAISS의 기본 설정은 L2 거리
            # 사인 유사도로 설정하고 싶을 때는 아래와 같이 하면 됨
            # from langchain_community.vectorstores.utils import DistanceStrategy
            # st.session_state.vectorstore = FAISS.from_texts(
            #     pdf_text,
            #     OpenAIEmbeddings(model="text-embedding-3-small"),
            #     distance_strategy=DistanceStrategy.COSINE
            # )


def page_pdf_upload_and_build_vector_db():
    st.title("PDF Upload 📄")
    pdf_text = get_pdf_text()
    if pdf_text:
        build_vector_store(pdf_text)


def main():
    init_page()
    page_pdf_upload_and_build_vector_db()


if __name__ == '__main__':
    main()
