from dotenv import load_dotenv
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model="deepseek-chat",api_key=os.getenv('DS_API_KEY'),base_url=os.getenv('DS_API_BASE'))
response = llm.invoke("请用一句话介绍 LangChain")
print(type(response))           # <class 'langchain_core.messages.ai.AIMessage'>
print(response.content)         # 模型生成的主要内容
print(response.response_metadata)  # 包含 finish_reason、model_name 等信息
print(response.usage_metadata)     # token 使用情况
print('-'*50)
#使用解析器

parser = StrOutputParser()
chain = llm | parser              
# 使用 LCEL 组合
result = chain.invoke("请用一句话介绍 LangChain")
print(result)                     
# 直接拿到字符串
print(type(result)) 
#parser如果直接解析只能解析字符串，但是如果放到chain里面就可以解析aimessage，输出str
