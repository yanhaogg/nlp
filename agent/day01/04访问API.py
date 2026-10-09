import os
import requests

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model = "deepseek-flash",api_key=os.getenv('DS_API_KEY'),base_url=os.getenv('DS_API_BASE'))

# 1.原调用代码
weather_api_key = os.getenv("weather_api_key")
city = "北京"

url = f"https://api.seniverse.com/v3/weather/now.json?key={weather_api_key}&location={city}&language=zh-Hans&unit=c&fields=location,update_time"
res = requests.get(url)  # 最重要的一句话
data = res.json()
print(data)
weather = data["results"][0]["now"]["text"]
temperature = data["results"][0]["now"]["temperature"]
location = data["results"][0]["location"]["name"]
update_time = data["results"][0]["last_update"]

print(location,temperature,weather,update_time)

# print(llm.invoke("现在北京天气怎么样"))

# 2.封装成函数
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
