import json
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
import os

from dotenv import load_dotenv
load_dotenv()

# 1.使用 @ tool 完成工具的封装
@tool
def load_file(filepath):
    '''
    函数用来加载本地文件,参数filepath是具体的文件地址
    '''
    with open(filepath,'r',encoding='utf-8') as f:
        return f.read()

@tool
def save_file(filepath,content):
    '''
    函数用来保存内容到本地文件,参数 filepath 为文件路径,content 为要保存的文本
    '''
    with open(filepath,'w',encoding='utf-8') as f:
        f.write(content)
    return f"已经保存到 {filepath}"

tools = [load_file,save_file]

# 2.配置 Agent
llm = ChatOpenAI(model = "deepseek-flash",api_key=os.getenv('DS_API_KEY'),base_url=os.getenv('DS_API_BASE'))

agent = create_agent(
    model = llm,  # 传入大脑(LLM)
    tools = tools, # 传入工具池
    system_prompt = """
任务流程:    
1. 使用 load_file 工具读取指定文件内容;
2. 将读取的文本 翻译成 英文;
3. 使用 save_file 工具,将处理后的结果保存到 指定路径;
4. 最后输出简洁的任务完成总结

请严格使用工具完成每个步骤
"""
)

# 3.测试
user_input = "帮我读取文件'agent/data/data.txt',翻译成英文,并保存到本地,路径名为 'agent/day01/output2.txt'"

res = agent.invoke({
    "messages":[HumanMessage(content = user_input)]
})

print(res["messages"][-1].content)
#print(res)
