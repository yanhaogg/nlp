from langchain_core.prompts import PromptTemplate

# template='''
# 是一个天气查询助手。请根据以下格式返回天气信息：
# {{
#     "city": "{city}",
#     "date": "{date}",
#     "weather": 
# }}
# '''

# prompt = PromptTemplate(input_variables=["city", "date"], template=template)
# print(prompt.format(city="北京", date="2025-6-18"))





# 将JSON作为变量整体注入
fixed_template = """
你是一个天气查询助手。请根据用户提供的城市和日期
城市：{city}
日期：{date}
并根据以下格式返回天气信息：
格式：
{params}
"""

prompt = PromptTemplate(
    input_variables=["city", "date", "params"],
    template=fixed_template
)
# 生成时传入已转义的JSON字符串
final_prompt = prompt.format(
    city="北京",
    date="2025-06-18",
    params='''
    {
        "city": "北京",
        "date": "2025-06-18",
        "weather": 
    }'''
)
print(final_prompt)

from model.client import *
res=llm.generate_response(final_prompt)
print(type(res))
print(res)

#转dict
import json
tem=json.loads(res)
print(type(tem))
print(tem)


#使用JsonOutputParser进行输出解析
from langchain_core.output_parsers import JsonOutputParser
parser=JsonOutputParser()
tem=parser.parse(res)
print(type(tem))
print(tem)



