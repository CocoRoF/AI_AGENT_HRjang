# Github: https://github.com/naotaka1128/llm_app_codes/chapter05/part2/map_reduce.py

import tiktoken
import traceback
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter

# models
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

from urllib.parse import urlparse
from langchain_community.document_loaders import YoutubeLoader  # Youtube용

###### dotenv 를 사용하지 않는 경우는 삭제해주세요 ######
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    import warnings

    warnings.warn(
        "dotenv not found. Please make sure to set your environment variables manually.",
        ImportWarning,
    )
################################################


SUMMARIZE_PROMPT = """아래 콘텐츠의 내용을 약 300자 정도로 알기 쉽게 요약해주세요.

========

{content}

========

한국어로 작성해줘!
"""


def init_page():
    st.set_page_config(page_title="Youtube Summarizer", page_icon="🤗")
    st.header("Youtube Summarizer 🤗")
    st.sidebar.title("Options")


def select_model(temperature=0):
    models = ("GPT-3.5", "GPT-4", "Claude 3.5 Sonnet", "Gemini 1.5 Pro")
    model = st.sidebar.radio("Choose a model:", models)
    if model == "GPT-3.5":
        return ChatOpenAI(temperature=temperature, model_name="gpt-3.5-turbo")
    elif model == "GPT-4":
        return ChatOpenAI(temperature=temperature, model_name="gpt-4o")
    elif model == "Claude 3.5 Sonnet":
        return ChatAnthropic(
            temperature=temperature, model_name="claude-3-5-sonnet-20240620"
        )
    elif model == "Gemini 1.5 Pro":
        return ChatGoogleGenerativeAI(
            temperature=temperature, model="gemini-1.5-pro-latest"
        )


def init_summarize_chain():
    llm = select_model()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("user", SUMMARIZE_PROMPT),
        ]
    )
    output_parser = StrOutputParser()
    return prompt | llm | output_parser


def init_chain():
    summarize_chain = init_summarize_chain()

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-3.5-turbo",
        chunk_size=500,
        chunk_overlap=0,
    )

    text_split = RunnableLambda(
        lambda x: [{"content": doc} for doc in text_splitter.split_text(x["content"])]
    )

    text_concat = RunnableLambda(lambda x: {"content": "\n".join(x)})

    map_reduce_chain = (
        text_split | summarize_chain.map() | text_concat | summarize_chain
    )

    # ✔ 요약 방식 기록 기능 추가
    def route(x):
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        token_count = len(encoding.encode(x["content"]))

        if token_count > 1000:
            st.session_state["mode"] = "map-reduce"
            return map_reduce_chain
        else:
            st.session_state["mode"] = "single"
            return summarize_chain

    chain = RunnableLambda(route)
    return chain


def validate_url(url):
    """URL이 유효한지 판단하는 함수"""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def get_content(url):
    with st.spinner("Youtube 불러오는 중..."):
        loader = YoutubeLoader.from_youtube_url(
            url, add_video_info=False, language=["ko", "en"]
        )
        res = loader.load()

        try:
            if res:
                content = res[0].page_content
                title = res[0].metadata.get("title", "YouTube Video")
                return f"Title: {title}\n\n{content}"
            else:
                return None
        except:
            st.write(traceback.format_exc())
            return None


def main():
    init_page()
    chain = init_chain()

    if url := st.text_input("URL: ", key="input"):
        is_valid_url = validate_url(url)

        if not is_valid_url:
            st.write("Please input valid url")
        else:
            if content := get_content(url):

                # 1) 먼저 route() 실행해 mode 설정
                _ = chain.invoke({"content": content})

                # 2) 요약 방식 표시
                if "mode" in st.session_state:
                    st.write(f"### 요약 방식: {st.session_state['mode']}")

                # 3) 실제 요약 스트리밍 실행
                st.markdown("## Summary")
                st.write_stream(chain.stream({"content": content}))

                st.markdown("---")
                st.markdown("## Original Text")
                st.write(content)


if __name__ == "__main__":
    main()
