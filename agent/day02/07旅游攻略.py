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
    system_prompt='''
你是一个智能旅行规划助手，请严格遵循 ReAct 模式。
【ReAct 规则】
- Thought: 先分析用户需求（时间、预算、偏好），决定下一步最优行动
- Action: 需要文化、历史、景点、交通等信息时调用 wikipedia_search 工具
- Observation: 仔细阅读工具返回的结果
- 重复循环，直到协调出完整、可行的旅行方案
最终用 "Final Answer: " 开头给出清晰、结构化的推荐方案（包含行程、日程、文化亮点、预算估算）。
''',
    debug = True
)

# 3.运行示例
user_input = "我今天下午想去北京玩,帮我规划一下"

# [system_prompt]  添加一个human_prompt
res = agent.invoke({
    "messages" : [HumanMessage(content = user_input)]
})
print(res["messages"][-1].content)

