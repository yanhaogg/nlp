import json
from langchain.tools import tool
from langchain_core.runnables import Runnable
import os



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

class SimpleAgent(Runnable):
    def __init__(self,llm,tools):
        self.llm = llm
        self.tools = {tool.name:tool for tool in tools}

    def invoke(self,content):
        # 模拟决策过程,生成对应指令
        load_cmd = {"tool":"load_file",
                    "args":{
                        "filepath":content["file_path"]
                    }
                    }
        res = self.tools[load_cmd["tool"]].invoke(load_cmd["args"])

        res = self.llm.invoke(f"翻译成英文 : \n {res}").content

        save_cmd = {"tool":"save_file",
                    "args":{
                        "filepath" : "agent/day01/output.txt",
                        "content" : res
                    }
                    }
        self.tools[save_cmd["tool"]].invoke(save_cmd["args"])

        return {"res":"已经完成翻译,请查看具体翻译结果"}

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(model = "deepseek-flash",api_key=os.getenv('DS_API_KEY'),base_url=os.getenv('DS_API_BASE'))
tools = [load_file,save_file]
agent = SimpleAgent(llm,tools)
res = agent.invoke({
    "file_path": "agent/data/data.txt"
})
print(res)