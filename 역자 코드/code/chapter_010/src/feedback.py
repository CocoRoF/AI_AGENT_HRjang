import os
import streamlit as st
from langsmith import Client
from streamlit_feedback import streamlit_feedback

import langchain
langchain.debug = True  # 또는 langchain.verbose = True

# ✅ LangSmith V2 Tracing 설정 추가
os.environ["LANGCHAIN_PROJECT"] = "default"
os.environ["LANGCHAIN_TRACING_V2"] = "true"  # <- Run이 항상 기록되도록 함

# Client 생성 (인자 없이)
langsmith_client = Client()

def add_feedback():
    run_id = st.session_state.get("run_id")
    print("\U0001F9EA 현재 run_id:", run_id)

    feedback = streamlit_feedback(
        feedback_type="thumbs",
        optional_text_label="[선택 사항] 설명을 입력하세요",
        key=f"feedback_{run_id}",
    )

    scores = {"👍": 1, "👎": 0}

    if feedback:
        score = scores.get(feedback["score"])
        comment = feedback.get("text")

        print("\U0001F9EA 피드백 score:", feedback["score"])
        print("\U0001F9EA 매핑된 점수:", score)
        print("\U0001F9EA 코멘트:", comment)
        print("\U0001F9EA LANGCHAIN_PROJECT:", os.getenv("LANGCHAIN_PROJECT"))
        print("\U0001F9EA LANGCHAIN_API_KEY 존재 여부:", "있음" if os.getenv("LANGCHAIN_API_KEY") else "없음")

        if score is not None:
            feedback_type_str = f"thumbs {feedback['score']}"

            try:
                feedback_record = langsmith_client.create_feedback(
                    run_id,
                    feedback_type_str,
                    score=score,
                    comment=comment,
                )
                st.session_state.feedback = {
                    "feedback_id": str(feedback_record.id),
                    "score": score,
                }
                print("✅ 피드백 생성 완료:", feedback_record)
            except Exception as e:
                print("❌ 피드백 생성 중 오류:", e)
                st.error("피드백 전송 중 오류가 발생했습니다.")
        else:
            print("⚠️ score 값이 None이므로 피드백 생략됨")
            st.warning("유효하지 않은 피드백 점수입니다")
    else:
        print("ℹ️ 사용자가 아직 피드백을 제출하지 않음")


# 디버깅용: 최근 피드백 목록 출력 함수

def debug_list_feedbacks():
    print("\n\U0001F50D 최근 피드백 5개:")
    for fb in langsmith_client.list_feedback(project_name="pr-bold-joke-17", limit=5):
        print("🟡 Feedback ID:", fb.id)
        print("➡️ Run ID:", fb.run_id)
        print("📝 Comment:", fb.comment)
        print("👍 Score:", fb.score)
        print("📅 Time:", fb.created_at)
        print("-----")