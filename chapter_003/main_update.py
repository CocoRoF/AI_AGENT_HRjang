import tiktoken
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

# models
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    import warnings

    warnings.warn(
        "dotenv not found. Please make sure to set your environment variables manually.",
        ImportWarning,
    )

# 2025년 12월 기준 최신 모델 가격 (per 1M tokens)
MODEL_PRICES = {
    "input": {
        "gpt-5-mini": 0.25 / 1_000_000,  # GPT-5 mini (빠르고 저렴)
        "gpt-5.1": 1.25 / 1_000_000,  # GPT-5.1 (코딩과 에이전트 작업용)
        "claude-sonnet-4-5-20250929": 3 / 1_000_000,  # Claude Sonnet 4.5 최신
        "gemini-2.5-flash": 0.30 / 1_000_000,  # Gemini 2.5 Flash 최신
    },
    "output": {
        "gpt-5-mini": 2 / 1_000_000,  # GPT-5 mini
        "gpt-5.1": 10 / 1_000_000,  # GPT-5.1
        "claude-sonnet-4-5-20250929": 15 / 1_000_000,  # Claude Sonnet 4.5 최신
        "gemini-2.5-flash": 2.50 / 1_000_000,  # Gemini 2.5 Flash 최신
    },
}

SYSTEM_PROMPT = "당신은 친절하고 유용한 도움을 주는 어시스턴트입니다."


def init_page():
    st.set_page_config(page_title="My Great ChatGPT", page_icon="🤗")
    st.header("My Great ChatGPT 🤗")
    st.sidebar.title("Options")


def init_messages():
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "message_history" not in st.session_state:
        st.session_state.message_history = []


def select_model():
    temperature = st.sidebar.slider(
        "Temperature:", min_value=0.0, max_value=2.0, value=0.0, step=0.1
    )

    models = ("GPT-5 mini", "GPT-5.1", "Claude Sonnet 4.5", "Gemini 2.5 Flash")
    model = st.sidebar.radio("Choose a model:", models)

    if model == "GPT-5 mini":
        st.session_state.model_name = "gpt-5-mini"
        return ChatOpenAI(temperature=temperature, model=st.session_state.model_name)
    elif model == "GPT-5.1":
        st.session_state.model_name = "gpt-5.1"
        return ChatOpenAI(temperature=temperature, model=st.session_state.model_name)
    elif model == "Claude Sonnet 4.5":
        st.session_state.model_name = "claude-sonnet-4-5-20250929"
        return ChatAnthropic(temperature=temperature, model=st.session_state.model_name)
    elif model == "Gemini 2.5 Flash":
        st.session_state.model_name = "gemini-2.5-flash"
        return ChatGoogleGenerativeAI(
            temperature=temperature, model=st.session_state.model_name
        )


def init_chain():
    st.session_state.llm = select_model()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="history"),
            ("user", "{user_input}"),
        ]
    )

    parser = StrOutputParser()
    return prompt | st.session_state.llm | parser


def get_message_counts(text):
    if "gemini" in st.session_state.model_name:
        return st.session_state.llm.get_num_tokens(text)
    else:
        if "gpt" in st.session_state.model_name:
            encoding = tiktoken.encoding_for_model(st.session_state.model_name)
        else:
            # Claude 모델은 gpt-4o 인코딩 사용
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

    cost_input = MODEL_PRICES["input"][st.session_state.model_name] * input_count
    cost_output = MODEL_PRICES["output"][st.session_state.model_name] * output_count
    cost = cost_input + cost_output

    st.sidebar.markdown("## Costs")
    st.sidebar.markdown(f"**Total cost: ${cost:.5f}**")
    st.sidebar.markdown(f"- Input cost: ${cost_input:.5f}")
    st.sidebar.markdown(f"- Output cost: ${cost_output:.5f}")


def main():
    init_page()
    init_messages()
    chain = init_chain()

    for msg in st.session_state.message_history:
        st.chat_message(msg["role"]).markdown(msg["content"])

    if user_input := st.chat_input("궁금한 내용을 입력해줘!"):
        st.session_state.message_history.append({"role": "user", "content": user_input})
        st.chat_message("user").markdown(user_input)

        with st.chat_message("assistant"):
            response = st.write_stream(
                chain.stream(
                    {
                        "history": st.session_state.message_history,
                        "user_input": user_input,
                    }
                )
            )

        st.session_state.message_history.append(
            {"role": "assistant", "content": response}
        )
    calc_and_display_costs()


if __name__ == "__main__":
    main()
