import tiktoken

encoding = tiktoken.encoding_for_model("gpt-5")
text = "This is a test for tiktoken."
tokens = encoding.encode(text)
print(len(tokens))
