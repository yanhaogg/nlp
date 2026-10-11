import baostock as bs
from langchain_core.tools import tool

@tool
def get_stock_history(code,start_date,end_date,frequency = "d"):
    '''
    用来查询 A 股股票历史 K 线数据(日线/周线/月线)
    code : 股票编码,示例 sh.600519(贵州茅台),sz.000001(平安银行)
    frequency: d=日线 , w=周线, m=月线
    start_date : 开始日期, 格式 xxxx-xx-xx
    end_date : 结束日期, 格式 xxxx-xx-xx
    '''
    # 1.先进行登录
    bs.login()

    rs = bs.query_history_k_data_plus(
        code=code,
        # 要获取的信息, 日期,编码,开盘价,最高价,最低价,收盘价,成交量
        fields="date,code,open,high,low,close,volume,amount",
        start_date=start_date,
        end_date=end_date,
        frequency=frequency,  # d 代表日K, w 代表周 K, m 代表月 K
        adjustflag="3"  # 3 = 后复权
    )

    bs.logout()

    # 2.转化为df
    df = rs.get_data()
    print(f"共获取 {len(df)} 条记录")
    print(f"最近10个交易日的情况\n{df.tail(10).to_string()}")
    return df