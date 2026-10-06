from langchain_core.prompts import PromptTemplate

template = """
你是一个天气查询助手。请根据用户提供的城市和日期，返回天气信息。
城市：{city}
日期：{date}
"""

prompt = PromptTemplate(
    input_variables=["date", "city"], 
    template=template,
    # 方法1：封装时预设默认值
    partial_variables={"date": "明天"}
)
# 即使不传city，也会使用默认值

#generated_prompt = prompt.format(city="北京", date="明天")

generated_prompt1=prompt.format(city='北京')

print(generated_prompt1)