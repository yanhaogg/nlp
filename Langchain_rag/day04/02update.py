from dotenv import load_dotenv
import os

load_dotenv()

api_key=os.getenv('DS_API_KEY')
api_base=os.getenv('DS_API_BASE')

from langchain_openai import ChatOpenAI

llm=ChatOpenAI(model="deepseek-flash",api_key=api_key,base_url=api_base)

res=llm.invoke('hello')
print(res)

# from openai import OpenAI
# # 调用聊天
# client = OpenAI()
# response1 = client.chat.completions.create(
#     model = "deepseek-flash",
#     messages = [{"role": "user", "content": "Hello"}],
#     stream = False
# )
# print(response1.choices[0].message.content)