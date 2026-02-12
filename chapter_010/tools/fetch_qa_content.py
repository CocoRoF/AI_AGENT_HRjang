import streamlit as st
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from pydantic import BaseModel, Field


class FetchQAContentInput(BaseModel):
    """타입을 지정하기 위한 클래스"""

    query: str = Field()


@st.cache_resource(show_spinner=False)
def load_qa_vectorstore(vectorstore_path="./vectorstore/qa_vectorstore"):
    """'자주 묻는 질문' 벡터 DB를 로드"""
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(
        vectorstore_path, embeddings=embeddings, allow_dangerous_deserialization=True
    )


@tool(args_schema=FetchQAContentInput)
def fetch_qa_content(query):
    """
    '자주 묻는 질문' 리스트에서 사용자의 질문과 관련된 콘텐츠를 검색하는 도구입니다.
    '영진모바일'에 관한 구체적인 정보를 얻는 데 유용합니다.

    반환값:
    - similarity: 질문과의 유사도 (0~1). 값이 높을수록 관련성이 높으며,
                  0.5 미만은 반환되지 않습니다.
    - content: 자주 묻는 질문과 그에 대한 답변 텍스트

    빈 리스트가 반환된 경우 관련 답변을 찾지 못한 것이므로, 질문을 좀 더 구체적으로 다시 요청하세요.

    Returns
    -------
    List[Dict[str, Any]]:
      - similarity: float
      - content: str
    """
    db = load_qa_vectorstore()
    docs = db.similarity_search_with_score(query=query, k=5, score_threshold=0.5)
    return [
        {"similarity": 1 - similarity, "content": i.page_content}
        for i, similarity in docs
    ]
