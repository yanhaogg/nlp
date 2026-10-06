from langchain_core.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from model.client import *

#定义模板
system_template = SystemMessagePromptTemplate.from_template("You are a helpful assistant.")
human_template = HumanMessagePromptTemplate.from_template(
    template="{user_question}",
    )


template = ChatPromptTemplate.from_messages([
    system_template,
    human_template
])

#填充变量生成messages
messages = template.format_messages(user_question="What is the capital of France?" )

print(messages)


#使用模板填充内容，会变成对应的messages对象
# 和promptTemplate有什么不同

#promptTemplate构造的是一个字符串，
#而ChatPromptTemplate构造的是一个messages对象列表

#openai是原生包，langchain_openai是封装了openai的包，提供了更多的功能和接口

for message in messages:
    print(message.type, message.content) 

#print(llm.generate_response(messages, temperature=1.0, top_p=0.95, max_tokens=1024, stop=None))

messages=convert(messages)
print(llm.generate_response(messages, temperature=1.0, top_p=0.95, max_tokens=1024, stop=None))