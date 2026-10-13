# langgraph : 四个核心概念 State,Node,Edge,Graph
# 1.State : 状态, 整个图运行过程中共享的数据(数据库).每个节点都可以读取 State,也可以向 State 写入新的内容
from typing import TypedDict

# 在 State 里面,会去存储跟用户问题相关的内容,以及大模型生成的相关内容
class AgentState(TypedDict):
    user_input : str
    intent : str
    answer : str

# 2. Node : 节点, 代表图中的一个处理步骤, 一般是指一个函数, 接收一个data ,返回一个更新的 data
# 比如某节点 : int函数 , '9' -> int('9') -> 9
def classify_intent(state):
    user_input = state["user_input"]

    if "退款" in user_input:
        intent = "refund"
    else:
        intent = "general"

    return {"intent" : intent}

# 3. Edge : 边,节点之间的链接关系,执行完一个节点之后,下一步执行哪个节点

# 4. Conditional Edge : 条件边,可以根据 State 动态选择下一步

# 5. Graph : 图 ,指的是完整的流程,由 State , Node 和 Edge 组合而成
