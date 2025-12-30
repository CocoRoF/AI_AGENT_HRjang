text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    # 모델에 따라 토큰 수 계산 방식이 다르므로 model_name을 지정
    model_name="gpt-5",
    # chunk size는 토큰 수로 계산
    chunk_size=60,
    chunk_overlap=20,
)

print(
    text_splitter.split_text(
        "Lorem Ipsum is simply dummy text of the printing and typesetting industry. \n\n Lorem Ipsum has been the industry's standard dummy text ever since the 1500s"
    )
)
