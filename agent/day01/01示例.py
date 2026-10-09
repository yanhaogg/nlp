# 读取本地文件
# 本质上是通过python 代码,给本地计算机下达指令, 让本地计算机干活
#
def read_file(path):
    with open(path,'r',encoding='utf-8') as f:
        content = f.read()
    return content
# print(read_file('../data/data.txt'))

# 如何让大模型自己读取
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model = "deepseek-flash",api_key=os.getenv('DS_API_KEY'),base_url=os.getenv('DS_API_BASE'))

# 1.直接把函数写在提示词里面
# res = llm.invoke("""
# 请执行下面的代码 :
# with open('../data/data.txt','r',encoding='utf-8') as f:
#     content = f.read()
# """)
# print(res.content)

# 2.直接调用指令
prompt = """
函数 : 
with open('../data/data.txt','r',encoding='utf-8') as f:
    content = f.read()
现在按照以下JSON结构调用函数:
{
"name":"read_file",
"arguments":{
"path":"文件路径"}
}
"""
res = llm.invoke(prompt)
print(res.content)

# 11 : 13
