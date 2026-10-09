from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
load_dotenv()

# 1.定义查询改写提示词模板
prompt = ChatPromptTemplate.from_messages([
    ("system","""
角色:你是一个专业的查询改写助手。
任务:先准确判断用户的意图,然后将用户的原始问题改写成更适合向量检索的版本。
约束：
- 使问题更加清晰、完整、具体
- 补充必要的上下文或明确指代
- 去除口语化表达，使其更正式
输出格式:只输出改写后的查询，不要输出任何解释或额外内容
"""),
    ("human","原始问题:{query}\n\n请改写成更适合检索的问题:")
])

# 2.初始化LLM
llm = ChatOpenAI(model="deepseek-chat",api_key=os.getenv('DS_API_KEY'),base_url=os.getenv('DS_API_BASE'))

# 3.构建 chain
chain = prompt | llm | StrOutputParser()

# 4.执行查询改写
query = '''马某下载使用某公司开发的词典APP时，系统提示需阅读隐私政策，隐私政策中载明需要收集电话号码等个人信息。若用户未实际阅读即点击手机屏幕其他位置，提示内容即消失并自动勾选“已阅读并同意隐私政策” ，且勾选后没有撤回同意的途径。若用户点击拒绝，该APP即自动退出，不提供任何查词服务。马某认为该APP强迫或变相强迫接受隐私政策，收集手机号等属于过度收集个人信息。'''

rewritten_query = chain.invoke({"query":query})

print(f"原始问题:{query}")
print(f"改写的问题:{rewritten_query}")
