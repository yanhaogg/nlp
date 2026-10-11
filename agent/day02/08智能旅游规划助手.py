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

@tool
def get_city_attractions(city):
    '''获取指定城市的主要旅游景点'''
    attractions = {
        "北京": "故宫、天安门、长城、颐和园、鸟巢",
        "上海": "外滩、东方明珠、豫园、南京路、迪士尼",
        "广州": "广州塔、长隆野生动物世界、陈家祠、珠江夜游",
        "深圳": "世界之窗、欢乐谷、莲花山公园、大梅沙海滨",
        "杭州": "西湖、灵隐寺、雷峰塔、千岛湖"
    }
    return attractions.get(city,f"{city} 暂无预设景点, 可参考当地旅游官网")

tools = [get_city_attractions,get_weather,wikipedia_search]

# 2.创建 agent
llm = ChatOpenAI(model = "deepseek-flash",api_key=os.getenv('DS_API_KEY'),base_url=os.getenv('DS_API_BASE'))

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="""你是一个智能助手，请严格使用 Plan-and-Execute 模式完成任务。
    【工作流程】
    1. Plan 阶段：先制定一个完整、详细、编号的执行计划，列出所有必要步骤和每个步骤的目标。
    2. Execute 阶段：严格按照计划，一步一步执行。每完成一步，就记录工具返回的结果（Observation），然后继续
    下一步。
    3. 只有当所有计划步骤都完成后，才给出最终答案。
    请先输出完整的 Plan（以 "Plan:" 开头），然后开始 Execute 过程。
    最终答案请以 "Final Answer: " 开头，用清晰友好的语言回复用户。""",
    debug = False
)

# 3.运行示例
user_input = "帮我规划一个周末从北京去上海的旅行，预算控制在5000元以内，偏好文化体验和美食。"

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

