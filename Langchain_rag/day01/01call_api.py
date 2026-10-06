# Please install OpenAI SDK first: `pip3 install openai`
import os
from openai import OpenAI
from model.api_key import api_key

client = OpenAI(
    api_key=api_key,                                                                #导入api
    base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-flash",                                                         #选择调用的模型
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "你好"},
    ],
    #stream=False,
    #reasoning_effort="high",
    #extra_body={"thinking": {"type": "enabled"}}
)

print(response.choices[0].message.content)