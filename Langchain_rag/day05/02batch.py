from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
load_dotenv()

prompt = ChatPromptTemplate.from_template("请用一句话解释：{concept}")

llm = ChatOpenAI(model="deepseek-chat",api_key=os.getenv('DS_API_KEY'),base_url=os.getenv('DS_API_BASE'))
parser = StrOutputParser()
chain = prompt | llm | parser

concepts = ["量子纠缠", "机器学习", "大语言模型"]
# for concept in concepts:
#     result = chain.invoke({"concept": concept})
#     print(result)

# result = chain.invoke({
# "concept": "量子纠缠"
# })
# print(result)

# results = chain.batch([
#     {"concept": "量子纠缠"},
#     {"concept": "机器学习"},
#     {"concept": "大语言模型"}
# ])

q=[{"concept": co} for co in concepts]
results = chain.batch(q)
for result in results: 
    print(result)