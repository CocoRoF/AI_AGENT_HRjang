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

    지정된 URL로부터 (와 페이지 번호로부터) 웹 페이지의 콘텐츠를 가져오는 툴.

    `status` 와 'page_content'('title', 'content', 'has_next' 인디케이터)를 반환합니다.
    status가 200이 아닌 경우에는 페이지를 가져올 때 에러가 발생한 것입니다.
    (기본적으로 페이지에서 최대 2,000 토큰의 콘텐츠만 가져옵니다.)
    페이지에 콘텐츠가 더 있는 경우에는 'has_next'는 True가 됩니다.
    계속을 읽으려면, 같은 URL에서 page_num 매개변수를 증가시켜 다시 입력해주세요.
    (페이징은 0부터 시작하고 다음 페이지는 1입니다)

    1페이지가 너무 길 경우에는 3번 이상 가져오지 마세요(메모리 부하가 발생할 수 있기
    때문입니다).

    Returns
    -------
    Dict[str, Any]:
    - status: str
    - page_content
        - title: str
        - content: str
        - has_next: bool
    """
    response = requests.get(url, timeout=timeout_sec)
    response.encoding = "utf-8"

    doc = Document(response.text)
    title = doc.title()
    html_content = doc.summary()
    content = html2text.html2text(html_content)

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-3.5-turbo",
        chunk_size=1000,
        chunk_overlap=0,
    )
    chunks = text_splitter.split_text(content)
    return {
        "status": 200,
        "page_content": {
            "title": title,
            "content": chunks[page_num],
            "has_next": page_num < len(chunks) - 1,
        },
    }
