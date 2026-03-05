import textwrap
import traceback

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from openai import OpenAI

load_dotenv()


class CodeInterpreterClient:
    """
    OpenAI Responses API의 Code Interpreter를 사용하여
    Python 코드를 가상 환경에서 실행하고 데이터를 분석하는 클래스입니다.

    - `upload_file`: 분석할 파일을 가상 환경에 업로드
    - `run`: Python 코드를 실행하고 결과(텍스트) 반환
    """

    def __init__(self):
        self.openai_client = OpenAI()
        self.container_id = self._create_container()
        # Code Interpreter 도구가 바인딩된 언어 모델 초기화
        self.llm = ChatOpenAI(
            model="gpt-5-mini",
            include=["code_interpreter_call.outputs"],
            temperature=0,
        ).bind_tools(
            [
                {
                    "type": "code_interpreter",
                    "container": self.container_id,
                }
            ]
        )

    def _create_container(self):
        """코드 실행과 파일 저장을 위한 가상 환경(Container) 생성"""
        container = self.openai_client.containers.create(
            name="code-interpreter-bigquery-session"
        )
        return container.id

    def upload_file(self, file_content, filename="uploaded_file.csv"):
        # Container에 파일 업로드
        response = self.openai_client.containers.files.create(
            container_id=self.container_id,
            file=(filename, file_content),
        )
        return filename, response.path

    def run(self, code):
        """
        Code Interpreter를 사용하여 Python 코드를 실행합니다.

        Args:
            code (str): 실행할 Python 코드
        Returns:
            str: 코드 실행 결과 텍스트
        """
        prompt = textwrap.dedent(
            f"""\
            다음 코드를 실행하고 결과를 반환해 주세요.
            파일 읽기에 실패한 경우, 가능한 범위 내에서 수정하고 다시 실행해 주세요.

            ```python
            {code}
            ```
            """
        )
        try:
            response = self.llm.invoke(prompt)
            text_parts = []
            for block in response.content:
                if (
                    isinstance(block, dict)
                    and block.get("type") == "code_interpreter_call"
                ):
                    for item in block.get("outputs", []):
                        if isinstance(item, dict):
                            logs = item.get("logs", "")
                            if logs:
                                text_parts.append(logs)

            output = "\n".join(text_parts).strip()

            return output

        except Exception as e:
            error_msg = f"[Code Interpreter 오류]\n{traceback.format_exc()}"
            return error_msg
