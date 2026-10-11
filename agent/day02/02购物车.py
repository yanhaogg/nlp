#错误处理与容错机制
import os
import requests
from langchain_core.tools import tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
load_dotenv()

# 1.设置工具
@tool
def calculate_total_price(items):
    '''计算购物车商品总价
    严格要求 : 本工具不做任何类型转换或者解析,请确保输入纯数字构成的列表
    '''
    try:
        return sum(items)
    except Exception as e:
        return f"工具执行错误 : {type(e).__name__} - {str(e)}"

# print(calculate_total_price(['1',2]))
tools = [calculate_total_price]

# 2.创建 Agent
llm = ChatOpenAI(model = "deepseek-flash",api_key=os.getenv('DS_API_KEY'),base_url=os.getenv('DS_API_BASE'))

agent = create_agent(
    model = llm,
    tools = tools,
    system_prompt='''
你是一个购物车计算助手。
用户会给你一段文本，其中包含一个数字列表（可能是字符串形式的列表或元组）。
你的任务是：
1. 从用户输入中提取出那个数字列表；
2. 使用 calculate_total_price 工具对提取出的列表进行计算；
3. 如果工具返回“工具执行错误”信息，请仔细阅读错误内容，分析原因，然后修正后重新调用工具；
4. 最终给出清晰的总价结果。
''',
    debug = True
)

# 3.运行示例

# user_task = "请帮我计算以下商品的总价 : [99,199, '299', 399]"
user_task = "请帮我计算以下商品的总价 : [99,壹佰, '299', 三百九十九]"
res = agent.invoke(
    {"messages":[HumanMessage(content = user_task)]},
    config = {
        "configurable":{
            "handle_parsing_errors":True,   # 自动处理解析错误
            "max_iterations":5              # 最大迭代次数,防止无限循环
        }
    }
)
print(res["messages"][-1].content)