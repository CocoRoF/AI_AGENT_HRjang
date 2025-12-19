# pip install langfuse

# Initialize Langfuse handler
from langfuse.callback import CallbackHandler

langfuse_handler = CallbackHandler(
    secret_key="sk-lf-...",
    public_key="pk-lf-...",
    host="https://cloud.langfuse.com",  # EU region
    # host="https://us.cloud.langfuse.com", # US region
)

# Your Langchain code

# Add Langfuse handler as callback (classic and LCEL)
chain.invoke({"input": "<user_input>"}, config={"callbacks": [langfuse_handler]})
