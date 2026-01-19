from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

prompt = PromptTemplate.from_template("Say: {content}")


# 대문자로 변환하는 함수
def to_upper(input):
    return {"content": input["content"].upper()}


# RunnableLambda로 함수 실행
to_upper = RunnableLambda(to_upper)

# # Lambda식을 사용해도 된다
# to_upper = RunnableLambda(lambda x: {"content": x["content"].upper()})

# chain을 실행
to_upper_chain = to_upper | prompt
print(to_upper_chain.invoke({"content": "yeah!"}))
