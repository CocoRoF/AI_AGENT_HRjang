from itertools import islice
from ddgs import DDGS
from langchain_core.tools import tool
from pydantic import BaseModel, Field


"""
DDGS Python 라이브러리의 응답 예시
--------------------------------------------
[
    {
        "title": "2025년 한국시리즈 - 위키백과, 우리 모두의 백과사전",
        "snippet": "결과는 LG 트윈스가 시리즈 전적 4승 1패로 ...",
        "url": "https://ko.wikipedia.org/wiki/2025년_한국시리즈",
    }, ...
]
"""


class SearchDDGSInput(BaseModel):
    query: str = Field(description="검색할 키워드를 입력하세요")


@tool(args_schema=SearchDDGSInput)
def search_ddgs(query, max_result_num=5):
    """
    DDGS로 키워드 검색을 실행하는 도구.

    제목, 스니펫(설명), URL을 반환하며, 정보가 단순화되어 있어
    오래되었거나 부족할 수 있습니다.
    원하는 정보를 찾지 못했다면 'fetch_page'로 페이지 내용을 직접 확인하세요.
    문맥에 따라 가장 적합한 언어로 검색하세요(사용자의 언어와 다를 수 있음).

    Returns
    -------
    List[Dict[str, str]]: title, snippet, url
    """
    res = DDGS().text(query, region="ko-kr", safesearch="off")
    return [
        {
            "title": r.get("title", ""),
            "snippet": r.get("body", ""),
            "url": r.get("href", ""),
        }
        for r in islice(res, max_result_num)
    ]
