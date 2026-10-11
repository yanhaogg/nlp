import os
import requests
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from dotenv import load_dotenv
load_dotenv()

# 1. 工具池的构建
from model.toolkit import *

tools = [wikipedia_search]

# 2.创建 agent
llm = ChatOpenAI(model = "deepseek-flash",api_key=os.getenv('DS_API_KEY'),base_url=os.getenv('DS_API_BASE'))

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="""你是一个智能投资研究助手，请严格使用 Self-Ask 模式进行复杂专业知识推理。
    【Self-Ask 规则】
    - 每一步先提出一个 Follow-up Question（自问子问题）
    - 需要专业知识时调用 wikipedia_search 工具
    - 根据 Observation 生成下一个子问题
    - 重复循环，直到所有关键问题都得到解答
    以 "Follow-up Question: " 开头，最终用 "Final Answer: " 给出结构化、专业化的分析报告。"""
    ,
    debug = False
)

# 3.运行示例
user_input = "帮我分析贵州茅台的投资价值，包括公司背景、竞争优势、当前风险和未来机会。"

# [system_prompt]  添加一个human_prompt

for chunk in agent.stream(
        {"messages":[HumanMessage(content = user_input)]},
        config = {
        "configurable":{
            "handle_parsing_errors":True,
            "max_iterations":5
                        }
        }
):
    # 直接打印 chunk 中所有消息
    # print(chunk)
    for key,value in chunk.items():
        print(f"{key}")
        print(value["messages"][0].content)
    print("=" * 80)

