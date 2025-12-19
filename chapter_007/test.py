from langchain.tools import tool  # ★ 경로 최신화
from langchain_openai import ChatOpenAI
from langchain.output_parsers import JsonOutputToolsParser

llm = ChatOpenAI(model="gpt-4o-mini")  # ★ 최신 모델 추천


@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)


llm_with_tools = llm.bind_tools([get_word_length])
chain = llm_with_tools | JsonOutputToolsParser()

res = chain.invoke("abafeafafa는 몇글자야?")
print(res)
