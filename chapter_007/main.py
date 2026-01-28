import streamlit as st


def init_page():
    st.set_page_config(page_title="Ask My PDF(s)", page_icon="🧐")


def main():
    init_page()
    st.sidebar.success("👆 왼쪽 메뉴에서 진행해 주세요")
    st.markdown(
        """
        ### Ask My PDF(s)에 오신 것을 환영합니다!
        - 이 앱에서는 업로드한 PDF에 대해 질문할 수 있습니다.
        - 먼저 왼쪽 메뉴에서 `📄 PDF 업로드`를 선택해 PDF를 업로드해 주세요.
        - PDF를 업로드한 뒤에는 `🧐 PDF 질의응답`을 선택해 질문해 보세요 😇
    """
    )


if __name__ == "__main__":
    main()
