import baostock as bs
from langchain_core.tools import tool

# 1.先进行登录
bs.login()

rs = bs.query_history_k_data_plus(
    code = "sh.600519",
    # 要获取的信息, 日期,编码,开盘价,最高价,最低价,收盘价,成交量,成交金额
    fields = "date,code,open,high,low,close,volume,volume,amount",
    start_date = "2025-07-01",
    end_date = "2025-12-01",
    frequency = "d", # d 代表日K, w 代表周 K, m 代表月 K
    adjustflag = "3" # 3 = 后复权
)

bs.logout()

# 2.转化为df
df = rs.get_data()
summary = f"共获取 {len(df)} 条记录"
print(df.tail(10).to_string())
print(df)