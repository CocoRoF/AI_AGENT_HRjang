from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

texts = [
    "OpenAI의 Embeddings API를 사용해서 텍스트를 Embedding하는 샘플 코드입니다.",
    "이 샘플 코드는 LangChain의 FAISS를 사용해서 Embedding 결과를 벡터 DB에 저장합니다",
    "Faiss는 Meta(facebook) 가 만든 벡터 DB로 검색 성능은 양호합니다.",
]

# 텍스트를 Embedding해서 벡터 DB에 저장한다
# 'from_text'는 벡터 DB를 초기화한다. 데이터를 추가할 때는 'add_texts'를 사용한다
# Document Loader를 입력으로 사용할 경우에는 'from_texts'가 아닌 'from_documents'를 사용한다
vectorstore = FAISS.from_texts(texts, OpenAIEmbeddings(model="text-embedding-3-small"))

# 벡터 DB에 저장된 Embedding 결과를 유사도(L2 거리) 기반으로 검색한다
# 유사도가 필요없다면: vectorstore.similarity_search(query)
query = "Faiss는 속도가 빠른가?"
doc_and_scores = vectorstore.similarity_search_with_score(query)
print(doc_and_scores)
