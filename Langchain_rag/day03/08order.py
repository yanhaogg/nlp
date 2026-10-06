import json
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import SimpleJsonOutputParser
from model.client import *


# ====================== 1. 使用 Loader 加载数据库 ======================
loader = TextLoader("Langchain_rag/data/database.json", encoding="utf-8")
docs = loader.load()
# 把加载后的文本转成 Python 字典（实际项目中可根据需要优化）
database = json.loads(docs[0].page_content)


# ====================== 2. 提取订单编号（核心业务） ======================
extract_prompt = ChatPromptTemplate.from_messages([
("system", """你是一个订单信息提取助手。
请从用户输入中提取订单号，并严格按以下 JSON 格式输出：
{{"order_id": "11位数字订单号"}}
如果无法提取到有效订单号，请输出：{{"error": "未识别到有效订单号"}}"""),
("human", "{user_input}")
])
user_input = input("\n请输入订单相关问题:")
messages = extract_prompt.format_messages(user_input=user_input)
messages=convert(messages)
raw_output = llm.generate_response(messages)


parser = SimpleJsonOutputParser()
parsed = parser.parse(raw_output)
print("\n【模型提取结果】")
print(parsed)
# ====================== 3. 根据订单号查询数据库 ======================
if "error" in parsed:
    print("\n【查询结果】未识别到有效订单号。")
else:
    order_id = parsed["order_id"]
    order_info = database.get(order_id)
    print(f"\n【数据库查询结果】")
    if order_info:
        print(f"订单号：{order_id}")
        print(f"商品名称：{order_info.get('商品名称', '未知')}")
        print(f"下单时间：{order_info.get('下单时间', '未知')}")
        print(f"订单状态：{order_info.get('订单状态', '未知')}")
        # ====================== 4. RAG 生成回答 ======================
        # 把查询到的订单信息作为 context 喂给 LLM
        context = f"""订单号：{order_id}
        商品名称：{order_info.get('商品名称', '未知')}
        下单时间：{order_info.get('下单时间', '未知')}
        订单状态：{order_info.get('订单状态', '未知')}"""
        generate_prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个专业的订单客服助手。请根据用户的问题和提供的订单信息，用中文给出准确、友好的回答。如果信息不足，请直接说明。"),
        ("human", f"""用户问题：{user_input}
                    订单信息：{context}
                    请回答用户的问题。""")
        ])
        gen_messages = generate_prompt.format_messages(user_input=user_input,context=context)
        gen_messages=convert(gen_messages)
        #print(gen_messages)
        response = llm.generate_response(gen_messages)
        print("\n【RAG 生成回答】")
        print(response.content if hasattr(response, "content") else response)
    else:
        print(f"未找到订单号 {order_id} 的信息。")#20250615001是什么商品