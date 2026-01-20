# 2. 사용자의 질문을 받아서 ChatGPT에게 전달하기 위한 템플릿 생성
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "당신은 친절하고 유용한 도움을 주는 어시스턴트입니다."),
        ("user", "{input}"),
    ]
)
