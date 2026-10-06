from langchain_core.output_parsers import StrOutputParser,CommaSeparatedListOutputParser
from langchain_core.prompts import PromptTemplate

from model.client import *
# 1. 最简单的 PromptTemplate
prompt_template = PromptTemplate(#from_template
    input_variables=["topic"],
    template="""
任务(Task):列出5个与'{topic}'相关的热门关键词，用英文逗号分隔。
内容(Context):主题：{topic}
"""
)
# 2. 格式化提示词
prompt = prompt_template.format(topic="大模型")
print("发送给模型的提示词：\n", prompt)
# 3. 真实调用你的模型
llm_output = llm.generate_response(prompt)          # ← 直接用你的 llm
print("\n模型原始输出:", llm_output)

print("逗号split:",llm_output.split(','))
# 4. 解析
output_parser = CommaSeparatedListOutputParser()#逗号分隔，同时去掉首尾空格
result = output_parser.parse(llm_output)
print("\n解析后结果类型:", type(result))
print("解析后结果(Python 列表):", result)