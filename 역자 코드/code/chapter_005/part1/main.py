import traceback
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# models
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time


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
        page_title="Website Summarizer",
        page_icon="🤗"
    )
    st.header("Website Summarizer 🤗")
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
    """웹페이지에서 콘텐츠를 가져오되, 실패하면 Selenium으로 재시도"""
    try:
        # 1차 시도: requests 방식 (정적 HTML)
        with st.spinner("웹Fetching Website ..."):
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            text = extract_main_text(soup)
            if "JavaScript와 쿠키를 활성화하여 계속 진행하세요" not in text:
                return text
            else:
                st.warning(" JS 렌더링이 필요한 페이지입니다. Selenium으로 재시도합니다...")

    except Exception:
        st.warning("requests로는 페이지를 불러올 수 없습니다. Selenium으로 재시도합니다...")

    # 2차 시도: Selenium
    try:
        with st.spinner("Fetching Website ...(Selenium 사용)"):
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--no-sandbox")

            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=chrome_options)


            driver.get(url)
            time.sleep(3)  # 페이지 로딩 대기

            html = driver.page_source
            driver.quit()

            soup = BeautifulSoup(html, 'html.parser')
            return extract_main_text(soup)

    except Exception as e:
        st.error("웹사이트 내용을 가져오는 데 실패했습니다.")
        st.text(traceback.format_exc())
        return None


def extract_main_text(soup):
    """본문일 가능성이 높은 영역에서 텍스트 추출"""
    if soup.main:
        return soup.main.get_text()
    elif soup.article:
        return soup.article.get_text()
    elif soup.body:
        return soup.body.get_text()
    else:
        return "본문을 찾을 수 없습니다."

	
def main():
    init_page()
    chain = init_chain()

    if "message_history" not in st.session_state:
        st.session_state.message_history = []	

    # 사용자의 입력을 감시
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

    # 비용을 표시하려면 3장의 코드를 추가해 주세요
    # calc_and_display_costs()


if __name__ == '__main__':
    main()
