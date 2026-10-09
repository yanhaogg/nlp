from langchain.tools import tool
import tiktoken
from langchain_core.messages import HumanMessage
import os
# 在原函数前面添加一个 @tool 即完成封装
# 装饰器意思就是给原函数,添加或者修改一些功能

# 1.工具封装
# 把不可调用的函数,封装成支持 Function Call 调用的函数
@tool
def count_tokens(text):
    '''
    函数说明 : 精确统计一段文本的 token 数量

    参数:
        text : 需要统计 token 数量的文本
    返回:
        该文本对应的 token 数量(整数)
    '''
    enc = tiktoken.get_encoding('cl100k_base')
    totens = enc.encode(text)
    return len(totens)
print(count_tokens.name)
print(count_tokens.description)
# 这里的描述(description)是给LLM看的,把函数的功能介绍给LLM,方便大模型发出准确指令.

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model = "deepseek-flash",api_key=os.getenv('DS_API_KEY'),base_url=os.getenv('DS_API_BASE'))

# 2.绑定的是工具池
# 把 tools 的信息,告诉LLM,让LLM知道现在有哪些函数(工具)
tools = [count_tokens]
llm_with_tools = llm.bind_tools(tools) # 让模型知道自己拥有 工具池里面的 工具

# 用户输入提示词
user_input = "统计下面文本的 token 数 : 人工智能的发展正在深刻改变我们的生活方式和生产方式。"

messages = [HumanMessage(content = user_input)]

# tools与llm绑定,相当于告诉LLM,只需要发送对应Function Call指令即可
response = llm_with_tools.invoke(messages)

print(response.tool_calls)


# 3.系统执行工具
if response.tool_calls:
    tool_call = response.tool_calls[0]
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]

    # 解析 Function Call指令,并指定对应的tool完成具体的工作
    if tool_name == "count_tokens":
        # 前面已经通过 @tool,将原函数,转变成对应的langchain格式的工具,因此支持Runnable协议,
        # 就支持invoke,batch,stream三种方法
        res = count_tokens.invoke(tool_args)
        print(f"tool 执行结果:{res}")