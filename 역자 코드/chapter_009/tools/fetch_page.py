
import requests
import html2text
from readability import Document
from langchain_core.tools import tool
from langchain_core.pydantic_v1 import (BaseModel, Field)
from langchain_text_splitters import RecursiveCharacterTextSplitter


class FetchPageInput(BaseModel):
    url: str = Field()
    page_num: int = 0


@tool(args_schema=FetchPageInput)
def fetch_page(url, page_num=0, timeout_sec=10):
    """
    지정된 URL(와 페이지 번호)로부터 웹페이지의 콘텐츠를 가져오는 툴입니다

    `status` 와 `page_content`（`title`、`content`、`has_next`인디케이터）를 반환합니다.
    status가 200이 아닌 경우는 페이지를 가져오는 과정에서 오류가 발생한 것입니다. (다른 페이지를 시도해 보세요)
    
    기본적으로는 최대 2,000 토큰 분량의 콘텐츠만 가져옵니다
    페이지에 더 많은 콘텐츠가 있는 경우에는 has_next 값은 True가 됩니다
    이어서 읽으려면 같은 URL에 page_num 파라미터를 증가시켜서 다시 입력해 주세요
    (페이지 번호는 0부터 시작하고 다음 페이지는 1입니다)

    한 페이지가 너무 긴 경우에는 3번 이상 가져오지 마세요 (메모리에 부담이 생길 수 있습니다).

    Returns
    -------
    Dict[str, Any]:
    - status: str
    - page_content
      - title: str
      - content: str
      - has_next: bool
    """
    try:
        response = requests.get(url, timeout=timeout_sec)
        response.encoding = 'utf-8'
    except requests.exceptions.Timeout:
        return {
            "status": 500,
            "page_content": {'error_message': 'Could not download page due to Timeout Error. Please try to fetch other pages.'}
        }

    if response.status_code != 200:
        return {
            "status": response.status_code,
            "page_content": {'error_message': 'Could not download page. Please try to fetch other pages.'}
        }
    
    try:
        doc = Document(response.text)
        title = doc.title()
        html_content = doc.summary()
        content = html2text.html2text(html_content)
    except:
        return {
            "status": 500,
            "page_content": {'error_message': 'Could not parse page. Please try to fetch other pages.'}
        }

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name='gpt-3.5-turbo',
        chunk_size=1000,
        chunk_overlap=0,
    )
    chunks = text_splitter.split_text(content)
    if page_num >= len(chunks):
        return {
            "status": 500,
            "page_content": {'error_message': 'page_num parameter looks invalid. Please try to fetch other pages.'}
        }
    elif page_num >= 3:
        return {
            "status": 503,
            "page_content": {'error_message': "Reading more of the page_num's content will overload your memory. Please provide your response based on the information you currently have."}
        }
    else:
        return {
            "status": 200,
            "page_content": {
                "title": title,
                "content": chunks[page_num],
                "has_next": page_num < len(chunks) - 1
            }
        }
