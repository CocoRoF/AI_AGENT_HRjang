cache = Cache()
config = {"configurable": {"thread_id": st.session_state["thread_id"]}}

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="법인 명의로 계약이 가능한가요?"):
    st.chat_message("user").write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 첫 번째 질문인 경우 캐시 확인
    if st.session_state["first_question"]:
        if cache_content := cache.search(query=prompt):
            st.chat_message("assistant").write(f"(cache) {cache_content}")
            st.session_state.messages.append(
                {"role": "assistant", "content": cache_content}
            )
            st.stop()  # 캐시 내용을 출력한 경우 실행 종료

    with st.chat_message("assistant"):
        with st.spinner("답변 생성 중..."):
            result = customer_support_agent.invoke(
                {"messages": [{"role": "user", "content": prompt}]}, config
            )
        response = result["messages"][-1].content
        st.write(response)

        if response:
            st.session_state.messages.append({"role": "assistant", "content": response})

    # 첫 번째 질문인 경우 캐시에 저장
    if st.session_state["first_question"] and response:
        cache.save(prompt, response)
