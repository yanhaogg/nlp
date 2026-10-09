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

for chunk in chain.stream({"concept": "量子纠缠"}):
    print(chunk, end="")