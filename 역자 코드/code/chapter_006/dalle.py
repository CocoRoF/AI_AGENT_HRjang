import base64
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper

###### dotenv를 사용하지 않는 경우에는 삭제해 주세요 ######
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    import warnings
    warnings.warn("dotenv not found. Please make sure to set your environment variables manually.", ImportWarning)
################################################


GPT4V_PROMPT = """
먼저, 아래의 사용자 요청과 업로드된 이미지를 주의 깊게 살펴보세요.
다음으로, 업로드된 이미지를 바탕으로 이미지를 생성해 달라는 사용자 요청에 따라 DALL·E용 프롬프트를 작성하세요
프롬프트는 반드시 영어로 작성해야 합니다.

사용자 입력: {user_input}

프롬프트에서는 사용자가 업로드한 사진에 무엇이 묘사되어 있는지, 어떻게 구성되어 있는지를 상세히 설명해야 합니다.
사진 속에 무엇이 보이는지가 명확하다면, 장소명이나 인물 이름을 정확하게 기재해 주세요.
또한, 사진의 구도나 줌의 배율도 가능한 한 자세히 설명해 주세요.
사진의 내용을 최대한 정확하게 재현하는 것이 중요합니다.
DALL·E 3 전용 프롬프트는 영어로 작성해 주세요.
"""


def init_page():
    st.set_page_config(
        page_title="Image Converter",
        page_icon="🤗"
    )
    st.header("Image Converter 🤗")


def main():
    init_page()

    llm = ChatOpenAI(
        temperature=0,
        model="gpt-4o",
        max_tokens=512
    )

    dalle3_image_url = None
    uploaded_file = st.file_uploader(
        label='Upload your Image here😇',
        # GPT-4V가 처리할 수 있는 이미지 파일만 업로드를 허용
        type=['png', 'jpg', 'webp', 'gif']
    )

    if uploaded_file:
        user_input = st.chat_input("이미지를 어떻게 편집하고 싶은지 알려주세요")
        if user_input:
            # 읽어들인 파일을 Base64로 인코딩
            image_base64 = base64.b64encode(uploaded_file.read()).decode()
            image = f"data:image/jpeg;base64,{image_base64}"

            query = [
                (
                    "user",
                    [
                        {
                            "type": "text",
                            "text": GPT4V_PROMPT.format(user_input=user_input)
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

            # GPT-4V에게 DALL·E 3용 이미지 생성 프롬프트를 작성하게 한다
            st.markdown("### Image Prompt")
            image_prompt = st.write_stream(llm.stream(query))

            # DALL-E 3에 의한 이미지 생성
            with st.spinner("DALL-E 3 is drawing ..."):
                dalle3 = DallEAPIWrapper(
                    model="dall-e-3",
                    size="1792x1024",   # "1024x1024", "1024x1792" 도 선택 가능
                    quality="standard", # 'hd'로 고화질 이미지 생성도 가능
                    n=1,  # 한 번에 한 장만 생성할 수 있음 (동시에 여러 요청은 가능)
                )
                dalle3_image_url = dalle3.run(image_prompt)

    else:
        st.write('먼저 이미지를 업로드해주세요😇')

    # DALL-E 3의 이미지 표시
    if dalle3_image_url:
        st.markdown("### Question")
        st.write(user_input)
        st.image(
            uploaded_file,
            use_column_width="auto"
        )

        st.markdown("### DALL-E 3 Generated Image")
        st.image(
            dalle3_image_url,
            caption=image_prompt,
            use_column_width="auto"
        )


if __name__ == '__main__':
    main()
