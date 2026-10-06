from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# 调用聊天
client = OpenAI(api_key=os.getenv('DS_API_KEY'),base_url=os.getenv('DS_API_BASE'))
response1 = client.chat.completions.create(
    model = os.getenv('DEEPSEEK'),
    messages = [{"role": "user", "content": "Hello"}],
    stream = False
)
#open_ai原生输出
print(response1)
print('-'*50)


from langchain_openai import ChatOpenAI
llm=ChatOpenAI(model=os.getenv('DEEPSEEK'),api_key=os.getenv('DS_API_KEY'),base_url=os.getenv('DS_API_BASE'))
res2=llm.invoke('hello')
print(res2)
print('='*50)