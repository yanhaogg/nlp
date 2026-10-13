import os
from typing import TypedDict,Literal


from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph,START,END

# 1.定义状态
# 状态本质上就是把整个图里面的数据全部集中在一个字典里面
class CustomerServiceState(TypedDict):
    user_input:str
    intent:str
    order_id:str
    refund_amount:float
    retry_count:int
    max_retry:int
    need_human_review:bool # 是否需要人工审核
    human_approved:bool # 人类是否同意/批准
    answer:str

# 2. 创建 Agent
intent_llm = ChatOpenAI(model = 'deepseek-flash',api_key=os.getenv("DS_API_KEY"),base_url=os.getenv("DS_API_BASE"),temperature = 0)

intent_agent = create_agent(
    model = intent_llm,
    tools = [],
    system_prompt='''
你是一个客服意图识别 Agent。
你的任务是判断用户输入是否属于退款问题。
请只输出以下两个标签之一：
- refund：用户想要退款、退货、取消订单并返还金额
- general：其他普通客服问题

禁止输出任何解释。
'''
)

# 11 : 15
# 3.定义节点
# 3.1 判断用户意图
# Agent与节点的关系,节点的主要功能靠Agent实现,Agent是节点的重要实现工具.但Agent 不等于 节点
# 但凡涉及到判断节点,都要有两个,一个是判断,另一个是route开头的处理
# 判断会被加到Node里面,而route开头的函数,会被添加到conditional_edges里面
def classify_intent(state:CustomerServiceState):
    '''
    使用 intent_agent 判断用户意图
    '''
    res = intent_agent.invoke({
        "messages":[{"role":"user","content":state["user_input"]}]
    })

    intent = res["messages"][-1].content.strip().lower()

    if "refund" in intent:
        intent = "refund"
    else:
        intent = "general"

    return {"intent":intent}

# 处理判断结果的函数
def route_by_intent(state:CustomerServiceState):
    '''根据意图选择流程'''
    if state["intent"] == "refund":
        return "refund"

    return "general"

# 3.2 处理客服问题(回答普通问题的节点)
# 这里回答问题,可以使用llm, Agent,或者Rag-llm,Agentic-RAG
def handle_general_question(state:CustomerServiceState):
    """处理普通客服问题"""
    return {"answer":"您好,您的问题已经收到.我们会为您及时提供帮助"}

# 3.3 处理订单问题
def check_order_info(state:CustomerServiceState):
    '''
    检查订单信息是否完整

    怎么检测订单号并不是关键,关键是如何构建循环.

    如果用户输入的信息,包含 ORD,则认为包含订单号,我们自己定义订单号即可
    如果没有订单号,则要求用户补充
    '''
    user_input = state["user_input"]

    if "ORD" in user_input:
        # 这里模拟订单号和退款金额
        # 实际情况下,一般使用 (RAG,或者MQL封装的tool或者 MCP Server)
        order_id = "ORD-20300101-001"

        if "大额" in user_input or "5000" in user_input:
            refund_amount = 5000
        else:
            refund_amount = 99

        return {"order_id":order_id,
                "refund_amount":refund_amount,
                "answer":"已经识别到订单信息,正在判断退款规则"}

    return {"answer":"为了能够继续帮您处理退款,请您提供正确的订单号和退款金额"}

def route_by_order_info(state:CustomerServiceState):
    # 这里相当于处理完之后,会有三路分支
    if state["order_id"]:
        return "complete"
    if state["retry_count"] >= state["max_retry"]:
        return "failed"

    return "incomplete"

# 分支1 : 循环追问订单信息
def ask_for_order_info(state:CustomerServiceState):
    '''
    循环追问用户补充订单信息

    为了课堂演示,这里使用 input() 模拟用户补充信息
    '''
    print("Agent : 为了继续处理退款,请您提供订单编号")
    new_input = input("用户补充:")

    return {"user_input":state["user_input"] + "\n" + new_input,
            "retry_count":state["retry_count"] + 1}

# 分支2 : 结束分支
def handle_missing_order_info(state : CustomerServiceState):
    '''达到最大追问次数后,结束流程'''
    return {"answer":"由于多次未能获取有效订单号,暂时无法处理退款申请,请稍后重新提交完整订单信息"}

# 3.4 退款问题处理
# 分支3 : 判断退款金额
def judge_refund_amount(state:CustomerServiceState):
    '''判断是否需要人工审核
    5000元需要审核(1000元以上需要人工审核)
    否则自动退款
    '''
    refund_amount = state["refund_amount"]

    if refund_amount >= 1000:
        return {
            "need_human_review":True,
            "answer":f"该订单退款金额过高,需要人工审核"
    }
    return {
        "need_human_review":False,
        "answer":f"该订单退款金额正常,直接进行自动退款流程"
    }

def route_by_refund_amount(state:CustomerServiceState):
    '''根据退款金额,选择自动处理或者人工审核'''
    if state["need_human_review"]:
        return "human"
    return "auto"

# 3.5 自动退款 vs 人工审核
def auto_refund(state:CustomerServiceState):
    """自动退款处理"""
    return {
        "answer":f"订单 {state['order_id']} 已经通过自动退款规则,退款金额 {state['refund_amount']}元,预计1~3个工作日原路返回 "
    }

def human_review(state:CustomerServiceState):
    """
    Human-in-the-Loop : 人工确认

    直接使用input() 模拟人工审核
    """
    print("\n系统提示 : 该退款需要人工审核.")
    print(f"订单号:{state['order_id']}")
    print(f"退款金额:{state['refund_amount']}")

    decision = input("人工审核是否通过? 请输入 yes / no :").strip().lower()

    approved = (decision == "yes")
    return {"human_approved":approved}

def handle_human_result(state:CustomerServiceState):
    if state["human_approved"]:
        return {"answer":f"订单 {state['order_id']} 已经通过人工审核,退款金额 {state['refund_amount']}元,预计1~3个工作日原路返回 "}
    return {"answer":f"订单 {state['order_id']} 的大额退款申请未通过人工审核,建议通过客服了解具体详情"}


# 4.构建 Langgraph
builder = StateGraph(CustomerServiceState)

# 4.1 添加节点
builder.add_node("classify_intent",classify_intent)
builder.add_node("handle_general_question",handle_general_question)
builder.add_node("check_order_info",check_order_info)
builder.add_node("ask_for_order_info",ask_for_order_info)
builder.add_node("handle_missing_order_info",handle_missing_order_info)
builder.add_node("judge_refund_amount",judge_refund_amount)
builder.add_node("auto_refund",auto_refund)
builder.add_node("human_review",human_review)
builder.add_node("handle_human_result",handle_human_result)

# 4.2 添加边
builder.add_edge(START,"classify_intent")

# 4.3 添加条件分支
# 三个参数,分别代表 : 判断节点,处理节点,分支节点(要往哪个节点走)
builder.add_conditional_edges(
    "classify_intent",
    route_by_intent,
    {
        "refund":"check_order_info",
        "general":"handle_general_question"
    }
)

# 4.4 条件分支1: 直接结束
builder.add_edge("handle_general_question",END)

# 4.5 条件分支2 : 检查订单信息
builder.add_conditional_edges(
    "check_order_info", # 判断函数
    route_by_order_info, # 处理函数
    {
        "complete" : "judge_refund_amount",
        "incomplete": "ask_for_order_info",
        "failed": "handle_missing_order_info"
    }
)

# 4.6 添加循环重试 : 当用户补充完信息后,再次检查订单信息
builder.add_edge("ask_for_order_info","check_order_info")

builder.add_edge("handle_missing_order_info",END)

builder.add_conditional_edges(
    "judge_refund_amount",
    route_by_refund_amount,
    {
        "auto" : "auto_refund",
        "human" : "human_review"
    }
)

builder.add_edge("auto_refund",END)

builder.add_edge("human_review","handle_human_result")
builder.add_edge("handle_human_result",END)

# 5.编译
graph = builder.compile()

if __name__ == '__main__':
    res = graph.invoke({
        "user_input": "我要退款,什么流程",
        "intent": "",
        "order_id": "",
        "refund_amount": 0.0,
        "retry_count": 0,
        "max_retry": 5,
        "need_human_review": False,
        "human_approved": False,
        "answer": ""
    })

    print('\n最终回复:')
    print(res["answer"])