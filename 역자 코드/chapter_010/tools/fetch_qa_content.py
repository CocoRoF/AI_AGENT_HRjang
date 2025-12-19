
import streamlit as st
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.pydantic_v1 import (BaseModel, Field)


class FetchQAContentInput(BaseModel):
    """ 형을 지정하기 위한 클래스 """
    query: str = Field()


@st.cache_resource
def load_qa_vectorstore(
    vectorstore_path="./vectorstore/qa_vectorstore"
):
    """‘자주 묻는 질문’ 벡터DB를 로드"""
    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(
        vectorstore_path,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )


@tool(args_schema=FetchQAContentInput)
def fetch_qa_content(query):
    """
    ‘자주 묻는 질문’ 목록에서 사용자의 질문과 관련된 콘텐츠를 찾아주는 도구입니다.
    '베어모바일'에 대한 구체적인 지식을 얻는 데 도움이 됩니다.

    이 도구는 similarity(유사도)와 content(내용)를 반환합니다.
    - similarity는 답변이 질문과 얼마나 관련이 있는지를 나타냅니다.
        값이 높을수록 질문과의 관련성이 높다는 의미입니다.
        similarity 값이 0.5 미만인 문서는 반환되지 않습니다.
    - content는 질문에 대한 답변 텍스트를 제공합니다.
        반적으로 자주 묻는 질문과 그에 대응하는 답변으로 구성됩니다.

    빈 리스트가 반환된 경우에는 사용자의 질문에 대한 적절한 답변을 찾을 수 없었다는 의미입니다.
    이럴 경우에는 사용자에게 질문 내용을 더 명확히 해달라고 요청하는 것이 좋습니다.

    Returns
    -------
    List[Dict[str, Any]]:
    - page_content
      - similarity: float
      - content: str
    """
    db = load_qa_vectorstore()
    docs = db.similarity_search_with_score(
        query=query,
        k=5,
        score_threshold=0.5
    )
    return [
        {
            "similarity": 1 - similarity,
            "content": i.page_content
        }
        for i, similarity in docs
    ]
