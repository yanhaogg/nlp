import tiktoken
# 选择编码器（GPT-4 / GPT-3 .5-turbo 使用cl100k_base）
enc = tiktoken .get_encoding("cl100k_base")
text = "大模型在日常对话中的表现已经相当成熟"
tokens = enc .encode(text)
print(len(tokens))
print(tokens)
