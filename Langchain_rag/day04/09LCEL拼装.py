from dotenv import load_dotenv
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate,SystemMessagePromptTemplate,HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

llm = ChatOpenAI(model="deepseek-chat",api_key=os.getenv('DS_API_KEY'),base_url=os.getenv('DS_API_BASE'))
chat_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "你是一位{role}专家"
    ),
    HumanMessagePromptTemplate.from_template(
        "请用一句话解释{concept}"
    )
])
parser = StrOutputParser()


# 4. 使用 LCEL 组合成链
chain = chat_prompt | llm | parser
# 5. 调用
result = chain.invoke({"role":"量子物理","concept": "量子纠缠"})
print(result)

# print(
#     chat_prompt.invoke({
#         "role": "量子物理",
#         "concept": "量子纠缠"
#     })
# )