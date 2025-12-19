import tiktoken
encoding = tiktoken.encoding_for_model('gpt-5')

text = "This is a test for tiktoken."
tokens = encoding.encode(text)
print(len(tokens))	# 토큰 수 : 9
print(tokens)	# [2500, 382, 261, 1746, 395, 260, 8251, 2488, 13]
