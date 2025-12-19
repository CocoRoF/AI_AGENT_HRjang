
from itertools import islice
from duckduckgo_search import DDGS
from langchain_core.tools import tool
from langchain_core.pydantic_v1 import (BaseModel, Field)


"""
Sample Response of DuckDuckGo python library
--------------------------------------------
[
    {
        'title': '일정·결과｜2025 프로야구｜한국 야구 협회',
        'href': 'https://www.koreabaseball.com/schedule/schedule.aspx',
        'body': '일정·결과｜2025 프로야구｜한국 야구 협회'
    }, ...
]
"""

class SearchDDGInput(BaseModel):
    query: str = Field(description="검색하고 싶은 키워드를 입력해주세요")


@tool(args_schema=SearchDDGInput)
def search_ddg(query, max_result_num=5):
    """
    DuckDuckGo 검색을 실행하기 위한 툴입니다
    검색하고 싶으 키워드를 입력해서 사용해 주세요
    검색 결과로는 각 페이지의 제목, 스니펫(설명), URL이 반환됩니다
    이 툴을 통해 얻는 정보는 매우 단순화되어 있으며 경우에 따라서는 오래된 정보일 수도 있습니다
    필요한 정보를 찾지 못한 경우에는 반드시 ‘fetch_page’를 사용해서 각 페이지의 내용을 확인해 주세요.
    문맥에 따라 가장 적절한 언어를 사용해 주세요 (사용자의 언어와 반드시 같을 필요는 없습니다).
    예를 들어, 프로그래밍 관련 질문은 영어로 검색하는 것이 가장 적절합니다

    Returns
    -------
    List[Dict[str, str]]:
    - title
    - snippet
    - url
    """
    res = DDGS().text(query, region='wt-wt', safesearch='off', backend="api")
    return [
        {
            "title": r.get('title', ""),
            "snippet": r.get('body', ""),
            "url": r.get('href', "")
        }
        for r in islice(res, max_result_num)
    ]
