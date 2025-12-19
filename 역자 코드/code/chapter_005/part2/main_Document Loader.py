# Github: https://github.com/naotaka1128/llm_app_codes/chapter05/part2/main.py

import traceback
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# models
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

from urllib.parse import urlparse
from langchain_community.document_loaders import YoutubeLoader  # Youtube用

###### dotenv를 사용하지 않는 경우에는 삭제해 주세요 ######
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    import warnings
    warnings.warn("dotenv not found. Please make sure to set your environment variables manually.", ImportWarning)
################################################


SUMMARIZE_PROMPT = """다음 콘텐츠의 내용을 300자 정도로 알기 쉽게 요약해 주세요.
========

{content}

========

한글로 써주세요
"""


def init_page():
    st.set_page_config(
        page_title="Youtube Summarizer",
        page_icon="🤗"
    )
    st.header("Youtube Summarizer 🤗")
    st.sidebar.title("Options")


def select_model(temperature=0):
    models = ("GPT-3.5", "GPT-4", "Claude 3.5 Sonnet", "Gemini 1.5 Pro")
    model = st.sidebar.radio("Choose a model:", models)
    if model == "GPT-3.5":
        return ChatOpenAI(
            temperature=temperature,
            model_name="gpt-3.5-turbo"
        )
    elif model == "GPT-4":
        return ChatOpenAI(
            temperature=temperature,
            model_name="gpt-4o"
        )
    elif model == "Claude 3.5 Sonnet":
        return ChatAnthropic(
            temperature=temperature,
            model_name="claude-3-5-sonnet-20240620"
        )
    elif model == "Gemini 1.5 Pro":
        return ChatGoogleGenerativeAI(
            temperature=temperature,
            model="gemini-1.5-pro-latest"
        )


def init_chain():
    llm = select_model()
    prompt = ChatPromptTemplate.from_messages([
        ("user", SUMMARIZE_PROMPT),
    ])
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    return chain


def validate_url(url):
    """ URL이 유효한지 판단하는 함수 """
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
    with st.spinner("Fetching Youtube ..."):
        loader = YoutubeLoader.from_youtube_url(
            url,
            add_video_info=True,  # 제목과 재생횟수도 가져올 수 있다
            language=['en', 'ko']  # 영어 - 한글 순서로 자막을 수집
        )
        res = loader.load()  # list of `Document` (page_content, metadata)
        try:
            if res:
                content = res[0].page_content
                title = res[0].metadata['title']
                return f"Title: {title}\n\n{content}"
            else:
                return None
        except:
            st.write(traceback.format_exc())  # 에러가 발생하면 에러 표시
            return None


def main():
    init_page()
    chain = init_chain()

    # 사용자 입력 감시
    if url := st.text_input("URL: ", key="input"):
        is_valid_url = validate_url(url)
        if not is_valid_url:
            st.write('Please input valid url')
        else:
            if content := get_content(url):
                st.markdown("## Summary")
                st.write_stream(chain.stream({"content": content}))
                st.markdown("---")
                st.markdown("## Original Text")
                st.write(content)

    # 비용을 표시하려면 3장의 코드를 추가하세요
    # calc_and_display_costs()


if __name__ == '__main__':
    main()
