

import os
import magic
import traceback
import mimetypes
from openai import OpenAI
import re

class CodeInterpreterClient:
    """

    OpenAI의 Assistants API의 Code Interpreter Tool을 사용해서

    Python 코드를 실행하거나 파일을 읽어서 분석을 수행하는 클래스


    이 클래스는 다음과 같은 기능을 제공합니다:
    1. OpenAI Assistants API를 사용한 Python 코드 실행
    2. 파일 업로드 및 Assistants API에 등록
    3. 업로드한 파일을 사용한 데이터 분석 및 그래프 작성


    주요 메서드:

    - upload_file(file_content): 파일을 업로드해서 Assistants API에 등록한다

    - run(prompt): Assistants API를 사용해서 Python 코드를 실행하거나 파일 분석을 수행한다

    Example:
    ===============
    from src.code_interpreter import CodeInterpreter
    code_interpreter = CodeInterpreter()
    code_interpreter.upload_file(open('file.csv', 'rb').read())

    code_interpreter.run("file.csv의 내용을 읽고 그래프를 그려 주세요")
    """
    def __init__(self):
        self.file_ids = []
        self.openai_client = OpenAI()
        self.assistant_id = self._create_assistant_agent()
        self.thread_id = self._create_thread()
        self._create_file_directory()
        self.code_intepreter_instruction = """

        제공된 데이터 분석용 Python 코드를 실행해 주세요.

        실행한 결과를 반환해 주세요. 당신의 분석 결과는 필요하지 않습니다.

        다시 한 번 말합니다. 실행한 결과를 반환해 주세요.

        파일 경로 등이 조금 틀린 경우에는 적절히 수정해 주세요.

        수정한 경우에는 수정 내용을 설명해 주세요.
        """

    def _create_file_directory(self):
        directory = "./files/"
        os.makedirs(directory, exist_ok=True)

    def _create_assistant_agent(self):
        """
        OpenAI Assistants API Response Example:
        ===============
        Assistant(
            id='asst_hogehogehoge',
            created_at=1713525431,
            description=None,
            instructions='You are a python code runner. Write and run code to answer questions.',
            metadata={},
            model='gpt-4o',
            name='Python Code Runner',
            object='assistant',
            tools=[
                CodeInterpreterTool(type='code_interpreter')
            ],
            response_format='auto',
            temperature=1.0,
            tool_resources=ToolResources(
                code_interpreter=ToolResourcesCodeInterpreter(file_ids=[]),
                file_search=None
            ),
            top_p=1.0
        )
        """
        self.assistant = self.openai_client.beta.assistants.create(
            name="Python Code Runner",
            instructions="You are a python code runner. Write and run code to answer questions.",
            tools=[{"type": "code_interpreter"}],
            model="gpt-4o",
            tool_resources={
                "code_interpreter": {
                    "file_ids": self.file_ids
                }
            }
        )
        return self.assistant.id

    def _create_thread(self):
        """
        OpenAI Assistants API Response Example:
        Thread(
            id='thread_hoge',
            created_at=1713525580,
            metadata={},
            object='thread',
            tool_resources=ToolResources(code_interpreter=None, file_search=None))
        """
        thread = self.openai_client.beta.threads.create()
        return thread.id

    def upload_file(self, file_content):
        """
        Upload file to assistant agent

        OpenAI Assistants API Response Example:
        FileObject(
            id='file-hogehoge',
            bytes=18,
            created_at=1713525743,
            filename='test.csv',
            object='file',
            purpose='assistants',
            status='processed',
            status_details=None
        )

        Args:
            file_content (_type_): open('file.csv', 'rb').read()
        """
        file = self.openai_client.files.create(
            file=file_content,
            purpose='assistants'
        )
        self.file_ids.append(file.id)
        self._add_file_to_assistant_agent()  # Update assistant with new files
        return file.id

    def _add_file_to_assistant_agent(self):
        self.assistant = self.openai_client.beta.assistants.update(
            assistant_id=self.assistant_id,
            tool_resources={
                "code_interpreter": {
                    "file_ids": self.file_ids
                }
            }
        )

    def run(self, code):
        """
        Assistants API Response Example
        ===============
        Message(id='msg_mzx4vA5cS8kuzLfpeALC049M', assistant_id=None, attachments=[], completed_at=None, content=[TextContentBlock(text=Text(annotations=[], value='I need to solve the equation `3x + 11 = 14`. Can you help me?'), type='text')], created_at=1713526391, incomplete_at=None, incomplete_details=None, metadata={}, object='thread.message', role='user', run_id=None, status=None, thread_id='thread_dmhWy82iU3S97MMdWk5Bzkc7')
        Run(id='run_ox2vsSkPB0VMViuMOnVXGlzH', assistant_id='asst_tXog4eZKOLIal42dO5nQQISB', cancelled_at=None, completed_at=1713526496, created_at=1713526488, expires_at=None, failed_at=None, incomplete_details=None, instructions='Please address the user as Jane Doe. The user has a premium account.', last_error=None, max_completion_tokens=None, max_prompt_tokens=None, metadata={}, model='gpt-4o', object='thread.run', required_action=None, response_format='auto', started_at=1713526489, status='completed', thread_id='thread_dmhWy82iU3S97MMdWk5Bzkc7', tool_choice='auto', tools=[CodeInterpreterTool(type='code_interpreter')], truncation_strategy=TruncationStrategy(type='auto', last_messages=None), usage=Usage(completion_tokens=151, prompt_tokens=207, total_tokens=358), temperature=1.0, top_p=1.0, tool_resources={})

        
        >> message
        SyncCursorPage[Message](
            data=[
                Message(
                    id='msg_VLCN8oRK9qXoaRa41e8F9YjS',
                    assistant_id='asst_tXog4eZKOLIal42dO5nQQISB',
                    attachments=[],
                    completed_at=None,
                    content=[
                        ImageFileContentBlock(
                            image_file=ImageFile(file_id='file-oL7oQPvIcbmvD3oAqRR5eX6r'),
                            type='image_file'
                        ),
                        TextContentBlock(
                            text=Text(
                                annotations=[
                                    FilePathAnnotation(
                                        end_index=174,
                                        file_path=FilePath(file_id='file-NK7CrMtrEIZixhV6WIAiTdtk'),
                                        start_index=136,
                                        text='sandbox:/mnt/data/Fibonacci_Series.csv',
                                        type='file_path'
                                    )
                                ],
                                value="Here's the sine curve, \\( y = \\sin(x) \\), plotted over the range from \\(-2\\pi\\) to \\(2\\pi\\). The curve beautifully illustrates the periodic nature of the sine function. If you need any further analysis or another graph, feel free to let me know!"
                            ),
                            type='text'
                        )
                    ],
                    created_at=1713526821,
                    incomplete_at=None,
                    incomplete_details=None,
                    metadata={},
                    object='thread.message',
                    role='assistant',
                    run_id='run_LwPzADWdCMbwWsxB4i5VsMyu',
                    status=None,
                    thread_id='thread_dmhWy82iU3S97MMdWk5Bzkc7'
                )
            ],
            object='list',
            first_id='msg_VLCN8oRK9qXoaRa41e8F9YjS',
            last_id='msg_VLCN8oRK9qXoaRa41e8F9YjS',
            has_more=True
        )
        """

        prompt = f"""

        다음 코드를 실행하고 결과를 반환해 주세요

        파일 읽기에 실패한 경우, 가능한 범위 내에서 수정하고 다시 실행해 주세요.
        ```python
        {code}
        ```

        당신의 의견이나 감상은 필요 없으니 실행 결과만 반환해 주세요
        """

        # add message to thread
        self.openai_client.beta.threads.messages.create(
            thread_id=self.thread_id,
            role="user",
            content=prompt
        )

        # run assistant to get response
        run = self.openai_client.beta.threads.runs.create_and_poll(
            thread_id=self.thread_id,
            assistant_id=self.assistant_id,
            instructions=self.code_intepreter_instruction
        )
        if run.status == 'completed': 
            message = self.openai_client.beta.threads.messages.list(
                thread_id=self.thread_id,
                limit=1  # Get the last message
            )
            try:
                file_ids = []
                for content in message.data[0].content:
                    if content.type == "text":
                        text_content = content.text.value
                        file_ids.extend([
                            annotation.file_path.file_id
                            for annotation in content.text.annotations
                        ])
                    elif content.type == "image_file":
                        file_ids.append(content.image_file.file_id)
                    else:
                        raise ValueError("Unknown content type")
            except:
                print(traceback.format_exc())
                return None, None
        else:            print("💥 코드 실행 실패. code:\n", code)            print("📂 다운로드된 파일들:", self.file_ids)            print("🧪 run 상태:", run.status)
            raise ValueError("Run failed")

        file_names = []
        if file_ids:
            for file_id in file_ids:
                file_names.append(self._download_file(file_id, text_content))

        return text_content, file_names

    def _download_file(self, file_id, text_content):        data = self.openai_client.files.content(file_id)        data_bytes = data.read()        # 파일명 추출        match = re.search(r'([\w\-]+\.png)', text_content, re.IGNORECASE)        if match:            file_name = f"./files/{match.group(1)}"        else:            file_name = f"./files/{file_id}.csv"        with open(file_name, "wb") as file:            file.write(data_bytes)        return file_name
