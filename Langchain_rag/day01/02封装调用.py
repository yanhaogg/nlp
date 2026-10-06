#python -m Langchain_rag.day01.02封装调用

from model.client import *

#print(llm.generate_response(prompt="你好"))
# prompt = '''
# 请将下面的中文翻译成英文:
# 人工智能正在改变世界
# '''

prompt = '''
请将下面的中文翻译成英文:
人工智能正在改变世界
'''
print(llm.generate_response(prompt))