import base64
import streamlit as st
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda

# OpenAI 클라이언트 초기화
client = OpenAI()

IMAGE_PROMPT_TEMPLATE = """
먼저, 아래의 사용자의 요청과 업로드된 이미지를 주의 깊게 읽어주세요.
다음으로, 업로드된 이미지를 기반으로 이미지를 생성해 달라는 사용자의 요청에 따라
이미지 생성용 프롬프트를 작성해 주세요.
프롬프트는 반드시 영어로 작성해야 합니다.

주의: 이미지 속 사람이나 특정 장소, 랜드마크, 상표 등을 식별하지 말아 주세요.
묘사는 사진 속 시각적 요소를 중립적으로 설명하는 방식으로 해주세요.

사용자 입력: {user_input}

프롬프트에서는 사용자가 업로드한 사진에 무엇이 담겨 있는지,
어떻게 구성되어 있는지를 설명해 주세요.
사진의 구도와 줌 정도도 설명해 주세요.
사진의 내용을 재현하는 것이 중요합니다.

이미지 생성용 프롬프트를 영어로 출력해 주세요:
"""


def generate_image(prompt: str) -> str:
    """GPT Image 모델로 이미지를 생성하고 base64 문자열을 반환"""
    response = client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        size="1024x1024",  # "1024x1536", "1536x1024"도 선택 가능
        quality="medium",  # "low", "medium", "high" 중 선택
        n=1,
    )
    return response.data[0].b64_json


# RunnableLambda로 감싸서 LCEL 체인에서 사용 가능하게 만든다
image_generator = RunnableLambda(generate_image)


def init_page():
    st.set_page_config(page_title="Image Converter", page_icon="🎨")
    st.header("Image Converter 🎨")


def main():
    init_page()

    llm = ChatOpenAI(
        temperature=0,
        model="gpt-4o",
    )

    generated_image_base64 = None

    uploaded_file = st.file_uploader(
        label="이미지를 업로드해 주세요 📷",
        type=["png", "jpg", "webp", "gif"],
    )

    if uploaded_file:
        if user_input := st.chat_input("이미지를 어떻게 가공하고 싶은지 알려줘!"):
            # 읽은 파일을 Base64로 인코딩
            image_base64 = base64.b64encode(uploaded_file.read()).decode()
            image = f"data:image/jpeg;base64,{image_base64}"

            query = [
                (
                    "user",
                    [
                        {
                            "type": "text",
                            "text": IMAGE_PROMPT_TEMPLATE.format(user_input=user_input),
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": image, "detail": "auto"},
                        },
                    ],
                )
            ]

            # LLM에게 이미지 생성용 프롬프트를 작성하게 함
            st.markdown("### Image Prompt")
            image_prompt = st.write_stream(llm.stream(query))

            # GPT Image로 이미지 생성
            with st.spinner("GPT Image가 그림을 그리는 중입니다..."):
                generated_image_base64 = image_generator.invoke(image_prompt)
    else:
        st.write("먼저 이미지를 업로드해 주세요 📷")

    # 생성된 이미지 표시
    if generated_image_base64:
        st.markdown("### Question")
        st.write(user_input)
        st.image(uploaded_file, use_container_width=True)
        st.markdown("### Generated Image")
        # base64를 디코딩하여 이미지로 표시
        image_bytes = base64.b64decode(generated_image_base64)
        st.image(image_bytes, caption=image_prompt, use_container_width=True)


if __name__ == "__main__":
    main()
