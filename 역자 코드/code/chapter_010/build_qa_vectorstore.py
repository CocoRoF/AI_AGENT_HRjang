

import pandas as pd
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

###### dotenv를 사용하지 않는 경우에는 삭제해 주세요 ######
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    import warnings
    warnings.warn("dotenv not found. Please make sure to set your environment variables manually.", ImportWarning)
################################################

def main():
    # CSV 파일에서 ‘자주 묻는 질문’을 불러온다

    qa_df = pd.read_csv('./data/bearmobile_QA.csv', encoding='euc-kr')# question,answer
    
    # 벡터DB에 저장할 데이터를 생성한다
    qa_texts = []
    for _, row in qa_df.iterrows():
        qa_texts.append(f"question: {row['question']}\nanswer: {row['answer']}")

    # 위 데이터를 벡터DB에 저장한다
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_texts(qa_texts, embeddings)
    db.save_local('./vectorstore/qa_vectorstore')


if __name__ == '__main__':
    main()
    print('done')
