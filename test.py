image_base64 = base64.b64encode(uploaded_file.read()).decode()
image = f"data:image/jpeg;base64,{image_base64}"
query = (
    "user",
    [
        {"type": "text", "text": user_input},
        {
            "type": "image_url",
            "image_url": {"url": image, "detail": "auto"},
        },
    ],
)
st.write_stream(llm.stream(query))
