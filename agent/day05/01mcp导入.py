# pip install mcp
from mcp.server.fastmcp import FastMCP#服务端
from mcp.client.stdio import stdio_client#客户端
from mcp import ClientSession,StdioServerParameters#客户对话，标准io参数

# 服务目标:
# 1.标准化 : 提供的是统一的工具描述, 调用和返回格式
# 2.可发现性 : Agent 可以自动发现并加载可用的工具
# 3.安全性和可控性 : 支持与认证,权限控制,隔离运行和审计机制结合使用

