MODEL_PRICES = {
    "input": {
        "gpt-5-mini": 0.25 / 1_000_000,  # GPT-5 mini (빠르고 저렴)
        "gpt-5.1": 1.25 / 1_000_000,  # GPT-5.1 (코딩과 에이전트 작업용)
        "claude-sonnet-4-5-20250929": 3 / 1_000_000,  # Claude Sonnet 4.5
        "gemini-2.5-flash": 0.30 / 1_000_000,  # Gemini 2.5 Flash
    },
    "output": {
        "gpt-5-mini": 2 / 1_000_000,  # GPT-5 mini
        "gpt-5.1": 10 / 1_000_000,  # GPT-5.1
        "claude-sonnet-4-5-20250929": 15 / 1_000_000,  # Claude Sonnet 4.5
        "gemini-2.5-flash": 2.50 / 1_000_000,  # Gemini 2.5 Flash
    },
}


def get_message_counts(text):
    if "gemini" in st.session_state.model_name:
        return st.session_state.llm.get_num_tokens(text)
    else:
        if "gpt" in st.session_state.model_name:
            encoding = tiktoken.encoding_for_model(st.session_state.model_name)
        else:
            # Claude 계열 모델은 토크나이저를 공개하지 않아서, tiktoken을 사용해 토큰 수를 계산한다
            # 정확한 토큰 수는 아니지만, 대략적인 토큰 수를 파악할 수 있다
            encoding = tiktoken.encoding_for_model("gpt-4o")
        return len(encoding.encode(text))
