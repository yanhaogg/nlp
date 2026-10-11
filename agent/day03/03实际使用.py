from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
import os

from dotenv import load_dotenv
load_dotenv()

# 1.定义 prompt
# 输入dict , 输出 PromptValue
prompt = ChatPromptTemplate.from_template("请使用一句话介绍 {topic} ")

# 2.定义 模型
# 输入PromptValue/str , 输出 AIMessage
llm = ChatOpenAI(model = "deepseek-flash",api_key=os.getenv('DS_API_KEY'),base_url=os.getenv('DS_API_BASE'))

# 3.定义输出解析器
# 输入 AIMessage , 输出 str
parser= StrOutputParser()

# 4.定义 RunnableLambda
# 输入 str  , 输出 str
def add_love(text):
    return text + "\n我爱大语言模型."
add_love_runnable = RunnableLambda(add_love)

# 5.使用LCEL链组合完成
chain = prompt | llm | parser | add_love_runnable

# 6.实际调用
res = chain.invoke({"topic":"量子力学"})
print(res)