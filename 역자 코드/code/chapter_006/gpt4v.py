import base64
import streamlit as st
from langchain_openai import ChatOpenAI


###### dotenv를 사용하지 않는 경우에는 삭제해 주세요 ######
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    import warnings
    warnings.warn("dotenv not found. Please make sure to set your environment variables manually.", ImportWarning)
################################################


def init_page():
    st.set_page_config(
        page_title="Image Recognizer",
        page_icon="🤗"
    )
    st.header("Image Recognizer 🤗")
    st.sidebar.title("Options")


def main():
    init_page()

    llm = ChatOpenAI(
        temperature=0,
        model="gpt-4o",
        max_tokens=512
    )

    uploaded_file = st.file_uploader(
        label='Upload your Image here😇',
        # GPT-4V가 처리할 수 있는 이미지 파일만 업로드를 허용
        type=['png', 'jpg', 'webp', 'gif']
    )
    if uploaded_file:
        if user_input := st.chat_input("궁금한 내용을 입력해 주세요!"):
            # 읽어들인 파일은 Base64로 인코딩
            image_base64 = base64.b64encode(uploaded_file.read()).decode()
            image = f"data:image/jpeg;base64,{image_base64}"

            query = [
                (
                    "user",
                    [
                        {
                            "type": "text",
                            "text": user_input
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image,
                                "detail": "auto"
                            },
                        }
                    ]
                )
            ]
            st.markdown("### Question")
            st.write(user_input)     # 사용자의 질문
            st.image(uploaded_file)  # 업로드한 이미지를 표시
            st.markdown("### Answer")
            st.write_stream(llm.stream(query))

    else:
        st.write('먼저 이미지를 업로드해 주세요😇')

if __name__ == '__main__':
    main()
