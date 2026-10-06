from model.client import *

prompt = """
任务(Task):生成一篇科幻文章
内容(Context):主人公名叫Alice
约束(Constraint):文章长度约200 字
"""

res = llm.generate_response(prompt=prompt, temperature=1.0, top_p=0.95, max_tokens=300, stop=None)#"\n\n"
#res = llm.generate_streaming_response(prompt=prompt, temperature=1.0, top_p=0.95, max_tokens=300, stop=None)#"\n\n"
print(res)