# Github: https://github.com/Youngjin-com/AI_AGENT/tree/main/chapter_005/part2/main.py

import traceback
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# models
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

from urllib.parse import urlparse
from langchain_community.document_loaders import YoutubeLoader


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


def init_chain():
    llm = select_model()
    prompt = ChatPromptTemplate.from_messages([("user", SUMMARIZE_PROMPT)])

    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
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
    """
    with st.spinner("Fetching Youtube ..."):
        try:
            loader = YoutubeLoader.from_youtube_url(
                url,
                add_video_info=False,  # 불필요한 메타데이터 요청 제거
                language=["ko", "en"],  # 한글 자막 우선, 없으면 영어
            )
            res = loader.load()

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

    # 사용자가 URL을 입력하면 요약을 수행
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


if __name__ == "__main__":
    main()
