from model.client import *

user_input = "今天天气不错，________"

prompt = f'''
task: 句子续写
context: {user_input}
constraints: 不超过50个字

'''

#Temperature=0.0: 生成的文本更确定，重复性更高，适合需要精确输出的任务。
print("Temperature=0.0:")
for i in range(3):
    response = llm.generate_response(prompt=prompt, temperature=0.0)
    print(response)


print("\nTemperature=1.0:")
for i in range(3):
    response = llm.generate_response(prompt=prompt, temperature=1.0)
    print(response)


print("\nTemperature=2.0:")
for i in range(3):
    response = llm.generate_response(prompt=prompt, temperature=2.0)
    print(response)

