from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from model.client import *
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is the capital of France?"),
    AIMessage(content="The capital of France is Paris.")
]

print(messages)


for message in messages:
    print(message.type, message.content)
print('='*30)
messages=convert(messages)
for message in messages:
    print(message.get("role"), message.get("content"))