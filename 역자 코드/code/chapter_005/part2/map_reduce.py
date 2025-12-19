
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
import os
import yt_dlp
import tempfile
import webvtt
import time

###### dotenv를 사용하지 않는 경우에는 삭제해 주세요 ######
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    import warnings
    warnings.warn("dotenv not found. Please make sure to set your environment variables manually.", ImportWarning)
################################################


SUMMARIZE_PROMPT = """다음 콘텐츠에 대해, 내용을 300자 정도로 알기 쉽게 요약해 주세요.
========

{content}

========

한글로 써줘
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


def init_summarize_chain():
    llm = select_model()
    prompt = ChatPromptTemplate.from_messages([
        ("user", SUMMARIZE_PROMPT),
    ])
    output_parser = StrOutputParser()
    return prompt | llm | output_parser


def init_chain():
    summarize_chain = init_summarize_chain()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=16000,
        chunk_overlap=0
    )

    text_split = RunnableLambda(
        lambda x: [
            {"content": doc} for doc
            in text_splitter.split_text(x['content'])
        ]
    )
    text_concat = RunnableLambda(
        lambda x: {"content": '\n'.join(x)})
    map_reduce_chain = (
        text_split
        | summarize_chain.map()
        | text_concat
        | summarize_chain
    )

    def route(x):
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        token_count = len(encoding.encode(x["content"]))
        if token_count > 16000:
            st.markdown("map_reduce_chain")
            return map_reduce_chain
        else:
            st.markdown("summarize_chain")
            return summarize_chain

    chain = RunnableLambda(route)

    return chain


def validate_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def download_youtube_subtitles(url):
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            for lang in ["en", "ko"]:
                ydl_opts = {
                    'skip_download': True,
                    'writesubtitles': True,
                    'subtitleslangs': [lang],
                    'subtitlesformat': 'vtt',
                    'outtmpl': os.path.join(tmpdir, '%(id)s.%(ext)s'),
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info_dict = ydl.extract_info(url, download=False)
                    video_id = info_dict.get("id")
                    vtt_path = os.path.join(tmpdir, f"{video_id}.{lang}.vtt")
                    ydl.download([url])

                    for _ in range(10):
                        if os.path.isfile(vtt_path):
                            persistent_vtt_path = os.path.join(".", f"{video_id}.{lang}.vtt")
                            with open(vtt_path, "rb") as f_in, open(persistent_vtt_path, "wb") as f_out:
                                f_out.write(f_in.read())
                            return persistent_vtt_path
                        time.sleep(0.5)
    except Exception:
        st.write(traceback.format_exc())
    return None


def vtt_to_text(vtt_path):
    try:
        if not os.path.exists(vtt_path):
            st.error(f"자막 파일이 존재하지 않습니다: {vtt_path}")
            return ""

        lines = []
        for caption in webvtt.read(vtt_path):
            lines.append(caption.text)
        return "\n".join(lines)

    except Exception:
        st.write(traceback.format_exc())
        return ""


def get_content(url):
    url = url.strip()
    with st.spinner("Fetching Youtube ..."):
        vtt_path = download_youtube_subtitles(url)
        if vtt_path:
            return vtt_to_text(vtt_path)
        else:
            return None


def main():
    init_page()
    chain = init_chain()

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
            else:
                st.error("자막을 가져올 수 없습니다.")


if __name__ == '__main__':
    main()
