# 用户输入  ->  识别意图  ->  根据意图选择处理节点  ->  输出结果

from typing import TypedDict

from langgraph.graph import StateGraph,START,END

# 1.定义对应的处理函数和状态字典
class AgentState(TypedDict):
    user_input : str
    intent : str
    answer : str

def classify_intent(state):
    user_input = state["user_input"]

    if "退款" in user_input:
        intent = "refund"
    else:
        intent = "general"

    return {"intent" : intent}

def handle_refund(state: AgentState):
    return {
        "answer":"您这在咨询退款问题, 请提供订单号, 我将继续为您服务"
    }

def handle_general(state:AgentState):
    return {
        "answer" : "您好, 我已经收到您的问题, 将为您提供帮助"
    }

def route_by_intent(state: AgentState):
    if state["intent"] == "refund":
        return "refund"

    return "general"

# 2. 创建完整图
# 2.1 创建图
builder = StateGraph(AgentState) # 状态字典全局通用

# 2.2 往图中添加节点
builder.add_node("classify_intent",classify_intent)
builder.add_node("handle_refund",handle_refund)
builder.add_node("handle_general",handle_general)

# 2.3 往图中添加边
# 边的添加逻辑是把节点 Node 链接起来
builder.add_edge(START,"classify_intent")

# 2.4 添加条件边
builder.add_conditional_edges(
    "classify_intent", # 出发点
    route_by_intent, # 判断逻辑
    {
        "refund" :"handle_refund", # 判断逻辑中的第一个结果
        "general" : "handle_general" # 顺次的结果
    }
)

builder.add_edge("handle_refund",END)
builder.add_edge("handle_general",END)

# 2.5 编译图
graph = builder.compile()

# 2.6 执行图
res = graph.invoke({
    "user_input" : "我想申请退款",
    "intent" : "",
    "answer" : ""
})

print(res)
print(res["answer"])
