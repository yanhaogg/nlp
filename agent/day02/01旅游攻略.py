import os
import requests
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from dotenv import load_dotenv
load_dotenv()

# 1. 工具池的构建
from model.toolkit import get_weather

print(get_weather.invoke({"city": "北京"}))

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

tools = [get_city_attractions,get_weather]

# 2.创建 agent
llm = ChatOpenAI(model = "deepseek-flash",api_key=os.getenv('DS_API_KEY'),base_url=os.getenv('DS_API_BASE'))

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt='''
角色 : 你是一个专业,细心的一日游规划助手
任务 : 用户输入一个城市后,请严格按照以下顺序执行:
1.调用  get_weather 工具,获取实时天气
2.调用  get_city_attractions 工具,获取该城市主要景点
3.综合以上信息,为用户生成一个简单一日游攻略(包括 : 上午,中午,下午,晚上建议行程,注意事项等)
约束:回复要简洁,实用,友好,最后以'祝你玩得开心'结束
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

