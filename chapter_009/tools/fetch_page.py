import requests
import html2text
from readability import Document
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langchain_text_splitters import RecursiveCharacterTextSplitter


class FetchPageInput(BaseModel):
    url: str = Field()
    page_num: int = Field(0, ge=0)


@tool(args_schema=FetchPageInput)
def fetch_page(url, page_num=0, timeout_sec=10):
    """
    지정된 URL의 웹페이지 콘텐츠를 가져오는 도구.

    `status`와 `page_content`(`title`, `content`, `has_next`)를 반환합니다.
    status가 200이 아니면 오류가 발생한 것이므로 다른 페이지를 시도하세요.

    기본적으로 최대 1,000 토큰 분량만 가져옵니다.
    콘텐츠가 더 있으면 `has_next`가 True가 되며,
    같은 URL에서 `page_num`을 1씩 증가시켜 다시 요청하세요(0부터 시작).
    단, 메모리 부담으로 3페이지 이상 조회하지 마세요.

    Returns
    -------
    Dict[str, Any]:
    - status: int
    - page_content: {title: str, content: str, has_next: bool}
    """
    try:
        response = requests.get(url, timeout=timeout_sec)
        response.encoding = "utf-8"
    except requests.exceptions.Timeout:
        return {
            "status": 500,
            "page_content": {
                "error_message": "타임아웃 오류. 다른 페이지를 시도하세요."
            },
        }

    if response.status_code != 200:
        return {
            "status": response.status_code,
            "page_content": {
                "error_message": "페이지 다운로드 실패. 다른 페이지를 시도하세요."
            },
        }

    try:
        doc = Document(response.text)
        title = doc.title()
        html_content = doc.summary()
        content = html2text.html2text(html_content)
    except:
        return {
            "status": 500,
            "page_content": {
                "error_message": "페이지 파싱 실패. 다른 페이지를 시도하세요."
            },
        }

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-3.5-turbo",
        chunk_size=1000,
        chunk_overlap=0,
    )
    chunks = text_splitter.split_text(content)

    if page_num >= len(chunks):
        return {
            "status": 500,
            "page_content": {
                "error_message": "잘못된 page_num. 다른 페이지를 시도하세요."
            },
        }
    elif page_num >= 3:
        return {
            "status": 503,
            "page_content": {
                "error_message": "메모리 초과 위험. 현재 정보로 답변을 작성하세요."
            },
        }
    else:
        return {
            "status": 200,
            "page_content": {
                "title": title,
                "content": chunks[page_num],
                "has_next": page_num < len(chunks) - 1,
            },
        }
