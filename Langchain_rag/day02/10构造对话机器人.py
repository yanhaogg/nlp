from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate, AIMessagePromptTemplate
from model.client import *
system_template = SystemMessagePromptTemplate.from_template(
    template="你是我的一个带记忆的聊天机器人"
)

user_template = HumanMessagePromptTemplate.from_template(
    template="{user_input}"
)

ai_template = AIMessagePromptTemplate.from_template(template="{ai_response}")

chat_prompt = ChatPromptTemplate.from_messages(
    [system_template]
)

messages = system_template.format_messages()

# msg=user_template.format_messages(user_input=input("User: "))
# msg=convert(msg)
# print(msg)
# print(llm.generate_response(msg, temperature=1.0, top_p=0.95, max_tokens=1024, stop=None))


from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
def main(memory=6,quiting='\\exit'):
    while True:
        user_input = input("User: ")
        if user_input.lower() == quiting:
            break

        if len(messages) >= memory:
            messages.pop(1)
            messages.pop(1)

       # print("msg:",user_template.format_messages(user_input=user_input))
        
        message = user_template.format_messages(user_input=user_input)[0]

        #print(f"msg[0]: {message}")
        #message = HumanMessage(content=user_input)
        messages.append(message)
  

        response = llm.generate_response(convert(messages), temperature=1.0, top_p=0.95, max_tokens=1024, stop=None)
        ai_message = ai_template.format_messages(ai_response=response)[0]
        
        #ai_message = AIMessage(content=response)
        messages.append(ai_message)
        #print(messages)
        print(f"AI: {response}")
        print("-" * 50)

main(memory=6,quiting='\\exit')