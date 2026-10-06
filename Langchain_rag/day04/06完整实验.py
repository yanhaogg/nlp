import os
from langchain_community.vectorstores import  Chroma 
from openai import OpenAI
import pickle
import chromadb
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
# 1.数据处理
# 1.1 加载数据
with open('Langchain_rag/data/法律条文切片.pkl','rb') as f:
    texts = pickle.load(f)
# 1.2 创建 Document 类
docs = [Document(page_content=text,
    metadata = {'source':'法律条文'})
    for idx,text in enumerate(texts)]
# 2.数据库操作
# 2.1 创建数据库(这里我们更换一下路径)
client = chromadb.PersistentClient(path="Langchain_rag/final_db")

embeddings=OpenAIEmbeddings(
    model=os.getenv('QIANWEN'),
    api_key=os.getenv('BAIDU_QIANFAN_API_KEY'),
    base_url=os.getenv('BAIDU_QIANFAN_BASE_URL'),
    check_embedding_ctx_length=False,#关闭 LangChain 的长度检查/token 化行为，让它直接传字符串
    chunk_size=20,
)
db = Chroma.from_documents(
    documents= docs,                    
    embedding=embeddings,               
    persist_directory="./final_db",      
    collection_name="law_knowledge",
    ids=[f"law_chunk_{i}" for i in range(len(docs))] # 设置ids
)
# 2.2 数据库增删查
# 2.2.1 查看数据库
collection = db.get()
print(collection["documents"][0])
print(f'文档的数量：{len(collection["documents"])}')
new_docs = [
Document(
    page_content="第十四条 个人信息处理者应当对其处理的个人信息负责...",
    metadata={"chapter": "第二章", "article": "第14条", "source": "个人信息保护法"}
    ),
    Document(
    page_content="第十五条 处理个人信息应当取得个人同意...",
    metadata={"chapter": "第二章", "article": "第15条", "source": "个人信息保护法"}
    )
]
db.add_documents(new_docs,ids=['law_chunk_101','law_chunk_102'])
collection = db.get()
print(collection['ids'])
print(f'文档的数量：{len(collection["documents"])}')
# 2.2.3 删除数据
db.delete(ids=['law_chunk_1','law_chunk_2'])
collection = db.get()
print(collection['ids'])
print(f'文档的数量：{len(collection["documents"])}')
db.delete(
where={"chapter": "第二章"}          
# 删除所有第二章的文档
)
collection = db.get()
print(collection['ids'])
print(f'文档的数量：{len(collection["documents"])}')


#3 数据库检索
#3.1 相似度检索similarity_search() 基础相似度检索
# results = db.similarity_search(
#     query="信息保护",  # 查询文本
#     k=3,  # 返回数量
#     #filter={"source": "法律条文"}  # 元数据过滤
# )

# print(results)

#3.2 带相似度得分的相似度检索
# docs_scores=db.similarity_search_with_relevance_scores(
#     query='信息保护',#'个人责任',
#     k=2
# )
# for doc, score in docs_scores:
#     print(f"相似度: {score:.4f}\n内容: {doc.page_content[:50]}...")


#3.3做一个相似度过滤，只保留相似度高的
def retrieve_docs(query,top_k=3,threshold=0.1):
    res = db.similarity_search_with_relevance_scores(query, k=top_k)
    res = [r for r in res if r[1] > threshold]
    if res:
        final_res = ''.join([r[0].page_content for r in res])
    else:
        final_res = ''
    return final_res



#4. 增强与生成
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate
)
from model.client import *
system_template = """
你是一个专业的法律AI助理，专长是根据《个人信息保护法》等法律条文进行知识解答。
请严格遵守以下要求：
1. 回答必须完全基于提供的文档内容，不得臆造信息
2. 直接回应问题，避免无关内容
"""
human_template = """根据以下文档内容，回答问题：
{context}
问题：{query}
答案：
"""
chat_prompt = ChatPromptTemplate.from_messages(
    [SystemMessagePromptTemplate.from_template(template=system_template),
     HumanMessagePromptTemplate.from_template(template=human_template)]
)
query = input("请输入问题：")
context = retrieve_docs(query)
print(f"context:{context}")
messages = chat_prompt.format_messages(context=context, query=query)
print(messages)
# messages = convert(messages)
# print(llm.invoke(messages))



from dotenv import load_dotenv
import os

load_dotenv()

api_key=os.getenv('DS_API_KEY')
api_base=os.getenv('DS_API_BASE')

from langchain_openai import ChatOpenAI

llm=ChatOpenAI(model="deepseek-flash",api_key=api_key,base_url=api_base)
print(llm.invoke(messages))


#朴素rag，纯粹使用向量检索，不去做任何的优化改进。 

#马某下载某公司开发的APP的时候，系统提示需阅读隐私政策，隐私政策中载明需要收集电话号码等个人信息。 若用户未实际阅读及点击手机屏幕其他位置，
# 提示内容消失，并自动勾选已阅读并同意隐私政策。且勾选后没有撤回同意的途径。 若用户点击拒绝，app 自动退出，不提供任何查词服务。马某认为该app过度手机个人信息