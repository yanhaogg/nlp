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
    system_prompt="""你是一个有帮助的助手，请严格遵循 ReAct 模式完成任务。
    【ReAct 规则】
    - Thought: 先思考当前用户输入、已知信息和下一步最优行动
    - Action: 如果需要外部信息或计算，就调用对应工具；否则直接给出最终答案
    - Observation: 仔细阅读工具返回的结果
    - 重复 Thought → Action → Observation 循环，直到可以给出完整答案
    最终答案请以 "Final Answer: " 开头，用清晰友好的语言回复用户。
    现在开始处理用户输入，自主进行 ReAct 思考。""",
    debug = True
)

# 3.运行示例

# user_task = "请帮我计算以下商品的总价 : [99,199, '299', 399]"
user_task = "请帮我计算以下商品的总价 : [99,壹佰, '299', 三百九十九]"
res = agent.invoke(
    {"messages":[HumanMessage(content = user_task)]},
    config = {
        "configurable":{
            "handle_parsing_errors":True,
            "max_iterations":5
        }
    }
)
print(res["messages"][-1].content)
