MODEL_PRICES = {
    "input": {
        "gpt-5-mini": 0.25 / 1_000_000,
        "gpt-5.1": 1.25 / 1_000_000,
        "claude-sonnet-4-5-20250929": 3 / 1_000_000,
        "gemini-2.5-flash": 0.30 / 1_000_000,
    },
    "output": {
        "gpt-5-mini": 2 / 1_000_000,
        "gpt-5.1": 10 / 1_000_000,
        "claude-sonnet-4-5-20250929": 15 / 1_000_000,
        "gemini-2.5-flash": 2.50 / 1_000_000,
    },
}


def get_message_counts(text):
    if "gemini" in st.session_state.model_name:
        return st.session_state.llm.get_num_tokens(text)
    else:
        if "gpt" in st.session_state.model_name:
            encoding = tiktoken.encoding_for_model(st.session_state.model_name)
        else:
            # Claude 계열 모델은 토크나이저를 공개하지 않아서 tiktoken으로 대략 계산
            encoding = tiktoken.encoding_for_model("gpt-4o")

        return len(encoding.encode(text))


def calc_and_display_costs():
    output_count = 0
    input_count = 0

    for msg in st.session_state.message_history:
        token_count = get_message_counts(msg["content"])
        if msg["role"] == "assistant":
            output_count += token_count
        else:
            input_count += token_count

    if not st.session_state.message_history:
        return

    model_name = st.session_state.model_name
    cost_input = MODEL_PRICES["input"][model_name] * input_count
    cost_output = MODEL_PRICES["output"][model_name] * output_count

    cost = cost_input + cost_output

    st.sidebar.markdown("## Costs")
    st.sidebar.markdown(f"**Total cost: ${cost:.5f}**")
    st.sidebar.markdown(f"- Input cost: ${cost_input:.5f}")
    st.sidebar.markdown(f"- Output cost: ${cost_output:.5f}")


def main():
    ...

    if user_input := st.chat_input("궁금한 내용을 입력해주세요."):
        ...

    calc_and_display_costs()
