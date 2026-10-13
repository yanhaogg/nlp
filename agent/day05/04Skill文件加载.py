from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from dotenv import load_dotenv
import os
load_dotenv()


def load_skill(path):
    '''
    读取 Skill Markdown 文件
    '''
    with open(path,'r',encoding='utf-8') as f:
        return f.read()

# def build_prompt(user_input,skill_content):
#     '''
#     将 Skill 内容拼接到模型上下文中
#     '''
#     content = f'''
# 角色 : 你是一个翻译助手
# 请严格遵守Skill完成任务:
#
# {skill_content}
#
# 用户任务:
# {user_input}
# '''
#     return content

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

user_input = "文件地址 'agent/data/data.txt'"

skill =  load_skill('agent/skills/Skill.md')
#LLM(model="deepseek-flash", api_key=os.getenv("DS_API_KEY"), base_url=os.getenv("DS_API_BASE"))
agent = create_agent(
    model = ChatOpenAI(model="deepseek-flash", api_key=os.getenv("DS_API_KEY"), base_url=os.getenv("DS_API_BASE")),
    system_prompt = skill,
    tools = tools
)

res = agent.invoke({
    "messages":[HumanMessage(content = user_input)]
})

# print(res["messages"][-1].content)
# print(res)