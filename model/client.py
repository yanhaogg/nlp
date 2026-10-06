import os
from dotenv import load_dotenv
load_dotenv()
a=os.getenv("DS_API_KEY")

api_key = os.getenv("DS_API_KEY")
base_url="https://api.deepseek.com"

from openai import OpenAI
import time
'''
client = OpenAI(
    api_key=api_key,                                                                #导入api
    base_url="https://api.deepseek.com")

response = client.chat.completions.create(                                          #completions:对话补全
    model="deepseek-flash",                                                         #选择调用的模型
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},               #给模型赋予的角色身份
        {"role": "user", "content": "你好，你叫什么名字"},                            #用户输入的内容query
    ],
    extra_body={"thinking": {"type": "disabled"}}                                   #非思考模式
)

print(response.choices[0].message.content)
'''


def convert(messages):
    # 批量转换任意消息类型为OpenAi兼容模式
    mapping = {
        "system": "system",
        "human": "user",
        "ai": "assistant"
    }
    result = [
        {"role": mapping.get(message.type), "content": message.content}#, message.type
        for message in messages
    ] 
    return result


#封装一下
class Model:
    def __init__(self, model_name, api_key=api_key, base_url=base_url):
        self.model_name = model_name
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

    def generate_response(self, prompt, temperature=1.0, top_p=0.95, max_tokens=1024, stop=None,system_role="You are a helpful assistant"):
        time0=time.time()

        if not isinstance(prompt, list):
            prompt=[
                {"role": "system", "content": system_role},               
                {"role": "user", "content": prompt},                            
            ]

        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            stop=stop,
            stream=False,
            extra_body={"thinking": {"type": "disabled"}} 
        )
        time1=time.time()
        print(f"Response time: {time1-time0:.2f}s")
        return response.choices[0].message.content

    def generate_streaming_response(self, prompt, temperature=1.0, top_p=0.95, max_tokens=1024, stop=None,system_role="You are a helpful assistant"):
            time0=time.time()

            if not isinstance(prompt, list):
                prompt=[
                    {"role": "system", "content": system_role},               
                    {"role": "user", "content": prompt},                            
                ],

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stop=stop,
                stream=True,
                extra_body={"thinking": {"type": "disabled"}} 
            )
            time1=time.time()

            full_response = ""

            for chunk in response:
                content = chunk.choices[0].delta.content

                if content is not None:
                    print(content, end="", flush=True)
                    full_response += content

            print(f"Response time: {time1-time0:.2f}s")
            return full_response


llm=Model(model_name="deepseek-flash")

if __name__ == "__main__":
    response = llm.generate_response(prompt="你好")
    print(response)