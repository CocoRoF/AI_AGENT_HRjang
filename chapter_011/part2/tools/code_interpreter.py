from langchain_core.tools import tool
from pydantic import BaseModel, Field

# 모듈 레벨 변수로 Code Interpreter 클라이언트 참조 저장
_client = None


def set_client(client):
    """Code Interpreter 클라이언트를 설정합니다."""
    global _client
    _client = client


class ExecPythonInput(BaseModel):
    """에이전트 입력 타입 정의"""

    code: str = Field()


@tool(args_schema=ExecPythonInput)
def code_interpreter_tool(code):
    """
    Code Interpreter를 사용해 Python 코드를 실행합니다.

    - 데이터 가공, 수식 계산, 통계 및 텍스트 분석에 사용합니다.
    - 외부 인터넷 액세스나 추가 라이브러리 설치는 불가능합니다.

    오류 시:
    - 같은 코드를 반복하지 말고 다른 방식을 사용하여 최대 2회 재시도합니다.

    Returns:
        str: 파이썬 코드 실행 결과 텍스트
    """
    text_result = _client.run(code)
    return text_result
