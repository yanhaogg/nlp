# 客户端
# 1. 查看服务端有哪些工具
# 2. 结合 Agent 的需求,适当调用服务器上面的工具
import asyncio

from mcp import ClientSession,StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    # 配置  MCP Server的启动参数
    server_params = StdioServerParameters(
        command = "python",#发送的是python命令,然后让出cpu的控制权,等待服务端返回结果
        args = ["agent/day05/02finance_server.py"]#指定服务端的脚本文件
    )

    # 通过 stdio 启动并连接 MCP Sever
    async with stdio_client(server_params) as (read,write):#启动客户端,并连接服务端
        # 创建一个会话,用于与服务端进行交互
        async with ClientSession(read,write) as session:
            # 1. 初始化会话
            await session.initialize() # 发起请求,然后让出cpu的控制权

            # 2. 动态发现所有可用工具
            tools_res = await session.list_tools()

            print("可用工具列表: ")

            for tool in tools_res.tools:
                print(f":- {tool.name} : {tool.description}")

            # 3. 调用工具
            res = await session.call_tool(
                "get_stock_price",
                arguments = {"symbol":"SH.000001"}
            )

            print("\n工具调用结果:")
            print(res.content)

if __name__ == '__main__':
    asyncio.run(main()) # 异步程序的启动入口
# 15 : 03
