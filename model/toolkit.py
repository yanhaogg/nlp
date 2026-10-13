import os
import requests
import baostock as bs
import wikipedia

from langchain_core.tools import tool
from dotenv import load_dotenv
load_dotenv()

@tool
def get_weather(city):
    '''
    获取指定城市的实时天气预报

    参数:
        city: 城市名称,如"北京"或者"beijing"
    返回:
        格式化天气信息字符串,包含地点,天气,温度和更新时间
        如果查询失败,直接返回"没有找到"
    '''
    weather_api_key = os.getenv("weather_api_key")
    city = city

    url = f"https://api.seniverse.com/v3/weather/now.json?key={weather_api_key}&location={city}&language=zh-Hans&unit=c&fields=location,update_time"
    try:
        res = requests.get(url)  # 最重要的一句话
        data = res.json()
        weather = data["results"][0]["now"]["text"]
        temperature = data["results"][0]["now"]["temperature"]
        location = data["results"][0]["location"]["name"]
        update_time = data["results"][0]["last_update"]
        return f"地点 : {location},天气 : {weather},温度 : {temperature},更新时间 : {update_time}"
    except:
        return f"没有检索到对应消息"

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
    try:
        login_result = bs.login()
        if login_result.error_code != '0':
            return f"登录失败: {login_result.error_msg}"

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
    except Exception as e:
        return f"查询股票历史数据失败 : {type(e).__name__} - {str(e)}"

@tool
def wikipedia_search(query):
    '''使用维基百科API搜索信息(支持中文)'''
    try:
        # 设置语言为中文
        wikipedia.set_lang("zh")
        page = wikipedia.page(query,auto_suggest= False)
        return f"标题 : {page.title}\n摘要 : {page.summary}"
    # 捕捉到的错误,第一,防止程序崩溃,会继续运行
    # 第二,捕捉到的报错信息,会帮助 Agent 排查错误
    except wikipedia.DisambiguationError as e:
        options = e.options[:3]
        return f"模糊匹配 : 请从以下选项中选择: {','.join(options)}"
    except wikipedia.PageError:
        return "未找到相关信息"
    except Exception as e:
        return f"搜索错误 : {str(e)}"