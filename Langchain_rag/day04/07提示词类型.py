from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
# 创建一个聊天提示模板
chat_prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="你是一位{role}专家"),
    HumanMessage(content="请用一句话解释{concept}")
])
# 通过 invoke 生成 PromptValue
prompt_value = chat_prompt.invoke({
    "role": "量子物理",
    "concept": "量子纠缠"
})
print(type(prompt_value))
print(prompt_value.to_string())      # 转为字符串
print(prompt_value.to_messages())    # 转为消息列表 


