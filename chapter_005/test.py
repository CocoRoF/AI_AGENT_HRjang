def init_chain():
    summarize_chain = init_summarize_chain()

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        # 모델마다 토큰 수를 계산하는 방식이 다르기 때문에,model_name을 지정한다
        # Claude 3를 사용할 때는 정확한 토큰 수를 계산할 수 없다는 점에 주의
        model_name="gpt-3.5-turbo",
        # 청크 크기는 토큰 수 기준으로 계산
        chunk_size=16000,
        chunk_overlap=0,
    )
    text_split = RunnableLambda(
        lambda x: [{"content": doc} for doc in text_splitter.split_text(x["content"])]
    )
    text_concat = RunnableLambda(lambda x: {"content": "\n".join(x)})
    map_reduce_chain = (
        text_split | summarize_chain.map() | text_concat | summarize_chain
    )

    def route(x):
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        token_count = len(encoding.encode(x["content"]))
        if token_count > 16000:
            return map_reduce_chain
        else:
            return summarize_chain

    chain = RunnableLambda(route)
