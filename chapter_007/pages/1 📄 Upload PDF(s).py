import fitz  # PyMuPDF
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

###### dotenv を利用しない場合は消してください ######
# dotenv를 사용하지 않는 경우는 삭제하세요
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
    st.set_page_config(page_title="Upload PDF(s)", page_icon="📄")
    st.sidebar.title("옵션")


def init_messages():
    clear_button = st.sidebar.button("DB 초기화", key="clear")
    if clear_button and "vectorstore" in st.session_state:
        del st.session_state.vectorstore


def get_pdf_text():
    # file_uploader로 PDF를 업로드한다
    # (file_uploader에 대한 자세한 설명은 6장을 참고하세요)
    pdf_file = st.file_uploader(
        label="PDF를 업로드하세요 😇", type="pdf"  # PDF 파일만 업로드 가능
    )
    if pdf_file:
        pdf_text = ""
        with st.spinner("PDF 로딩 중 ..."):
            # PyMuPDF로 PDF를 읽어들인다
            # (자세한 설명은 라이브러리 공식 문서를 참고하세요)
            pdf_doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
            for page in pdf_doc:
                pdf_text += page.get_text()

        # RecursiveCharacterTextSplitter로 텍스트를 청크 단위로 분할
        # (자세한 설명은 6장을 참고하세요)
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="text-embedding-3-small",
            # 적절한 chunk size는 PDF 종류에 따라 조정이 필요
            # 너무 크게 설정하면 여러 위치의 정보를 참조하기 어려워짐
            # 너무 작으면 하나의 청크에 충분한 문맥이 들어가지 않음
            chunk_size=500,
            chunk_overlap=0,
        )
        return text_splitter.split_text(pdf_text)
    else:
        return None


def build_vector_store(pdf_text):
    with st.spinner("벡터 스토어 저장 중 ..."):
        if "vectorstore" in st.session_state:
            st.session_state.vectorstore.add_texts(pdf_text)
        else:
            # 벡터 DB 초기화와 문서 추가를 동시에 수행
            # LangChain의 Document Loader를 사용할 경우 `from_documents` 사용
            st.session_state.vectorstore = FAISS.from_texts(
                pdf_text, OpenAIEmbeddings(model="text-embedding-3-small")
            )

            # FAISS 기본 설정은 L2 거리
            # 코사인 유사도를 사용하려면 아래처럼 설정
            # from langchain_community.vectorstores.utils import DistanceStrategy
            # st.session_state.vectorstore = FAISS.from_texts(
            #     pdf_text,
            #     OpenAIEmbeddings(model="text-embedding-3-small"),
            #     distance_strategy=DistanceStrategy.COSINE
            # )


def page_pdf_upload_and_build_vector_db():
    st.title("PDF 업로드 📄")
    pdf_text = get_pdf_text()
    if pdf_text:
        build_vector_store(pdf_text)


def main():
    init_page()
    page_pdf_upload_and_build_vector_db()


if __name__ == "__main__":
    main()


# 보완 사항
# 백테 데이터베이스에 대한 개념 추가
# text-embedding-3-small 말고도 정보 추가
