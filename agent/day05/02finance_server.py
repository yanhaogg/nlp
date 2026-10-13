from mcp.server.fastmcp import FastMCP

# 创建 MCP Sever
mcp = FastMCP('finance_tools')

@mcp.tool()
def get_stock_price(symbol):
    '''
    获取股票实时价格
    args :
        symbol: 股票代码,比如SH.600001
    return:
        包含股票代码,价格和时间戳的字典
    '''
    return {
        "symbol":symbol.upper(),
        "price":17.56,
        "timestamp":"2026-05-01 09:31:00"
    }
if __name__ == '__main__':
    mcp.run()
#运行服务端