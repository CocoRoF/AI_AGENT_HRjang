import pandas as pd
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


def main():
    # CSV 파일에서 "자주 묻는 질문"을 읽어오기
    qa_df = pd.read_csv("./data/bearmobile_QA.csv")  # question,answer

    # 벡터 DB에 저장할 데이터를 생성
    qa_texts = []
    for _, row in qa_df.iterrows():
        qa_texts.append(f"question: {row['question']}\nanswer: {row['answer']}")

    # 위 데이터를 벡터 DB에 저장
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_texts(qa_texts, embeddings)
    db.save_local("./vectorstore/qa_vectorstore")


if __name__ == "__main__":
    main()
    print("done")
