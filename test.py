from src.feedback import add_feedback

...


def main():
    ...

    customer_support_agent = create_customer_support_agent()

    if prompt := st.chat_input(placeholder="법인 명의로 계약이 가능한가요?"):
        ...

        with st.chat_message("assistant"):
            with st.spinner("답변 생성 중..."):
                run_id = uuid7()
                result = customer_support_agent.invoke(
                    {"messages": [{"role": "user", "content": prompt}]},
                    {**config, "run_id": run_id},
                )
            response = result["messages"][-1].content
            st.write(response)

    # LangSmith feedback 버튼
    add_feedback()
