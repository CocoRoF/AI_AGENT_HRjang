# Github: https://github.com/Youngjin-com/AI_AGENT/tree/main/chapter_005/part2/map_reduce.py

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


SUMMARIZE_PROMPT = """다음 콘텐츠의 내용을 약 300자 정도로 알기 쉽게 요약해주세요.
========
{content}
========
한국어로 작성해 주세요!
"""


def init_page():
    st.set_page_config(page_title="Youtube Summarizer", page_icon="🤗")
    st.header("Youtube Summarizer 🤗")
    st.sidebar.title("Options")


def select_model(temperature=0):
    models = ("GPT-5 mini", "GPT-5.1", "Claude Sonnet 4.5", "Gemini 2.5 Flash")
    model = st.sidebar.radio("Choose a model:", models)
    if model == "GPT-5 mini":
        return ChatOpenAI(temperature=temperature, model="gpt-5-mini")
    elif model == "GPT-5.1":
        return ChatOpenAI(temperature=temperature, model="gpt-5.1")
    elif model == "Claude Sonnet 4.5":
        return ChatAnthropic(
            temperature=temperature, model="claude-sonnet-4-5-20250929"
        )
    elif model == "Gemini 2.5 Flash":
        return ChatGoogleGenerativeAI(
            temperature=temperature,
            model="gemini-2.5-flash",
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
        # 모델에 따라 토큰 수 계산 방식이 다르므로 model_name을 지정
        model_name="gpt-5",
        # chunk size는 토큰 수로 계산
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
        encoding = tiktoken.encoding_for_model("gpt-5")
        token_count = len(encoding.encode(x["content"]))
        if token_count > 16000:
            return map_reduce_chain
        else:
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
    """
    Document:
        - page_content: str
        - metadata: dict
            - source: str
            - title: str
            - description: Optional[str],
            - view_count: int
            - thumbnail_url: Optional[str]
            - publish_date: str
            - length: int
            - author: str
    """
    with st.spinner("Youtube 불러오는 중..."):
        loader = YoutubeLoader.from_youtube_url(
            url,
            add_video_info=False,  # 중요: 에러 방지를 위해 메타데이터 가져오기 끔
            language=["ko", "en"],  # 한국어 우선, 없으면 영어
        )
        res = loader.load()  # list of `Document` (page_content, metadata)
        try:
            if res:
                return res[0].page_content
            else:
                return None
        except Exception as e:
            st.error(f"Error occurred: {e}")
            st.write(traceback.format_exc())
            return None


def main():
    init_page()
    chain = init_chain()

    # 사용자의 입력을 감시
    if url := st.text_input("URL: ", key="input"):
        is_valid_url = validate_url(url)
        if not is_valid_url:
            st.write("Please input valid url")
        else:
            if content := get_content(url):
                st.markdown("## Summary")
                st.write_stream(chain.stream({"content": content}))
                st.markdown("---")
                st.markdown("## Original Text")
                st.write(content)

    # 비용을 표시하려면 3장의 코드를 추가해 주세요
    # calc_and_display_costs()


if __name__ == "__main__":
    main()
