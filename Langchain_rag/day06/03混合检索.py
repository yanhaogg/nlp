# ===========   导入模型  =============
from langchain_chroma import Chroma
import chromadb
import json
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser

from langchain_openai import OpenAIEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()


embeddings=OpenAIEmbeddings(
    model=os.getenv('QIANWEN'),
    api_key=os.getenv('BAIDU_QIANFAN_API_KEY'),
    base_url=os.getenv('BAIDU_QIANFAN_BASE_URL'),
    check_embedding_ctx_length=False#关闭 LangChain 的长度检查/token 化行为，让它直接传字符串
)


#  ==========  配置参数   ===========
persist_dir = "Langchain_rag/db"
collection_name = "nlp_papers"

vector_weight = 0.7
bm25_weight = 0.3

# 1.混合检索
# 1.1 加载数据库
client = chromadb.PersistentClient(path = persist_dir)

db = Chroma(
    client = client,
    collection_name=collection_name,
    embedding_function=embeddings
)

# print(len(db.get()["ids"]))

# 1.2 创建两个检索器
# 5 * 2 * 4 = 40
vector_retriever = db.as_retriever(
    search_type = "similarity",
    search_kwargs = {"k": 5}
)

all_docs = db.get()
documents = [Document(page_content = content,metadata = meta)
             for content,meta in zip(all_docs["documents"],all_docs["metadatas"])]

# BM25Retriever.from_documents会把所有的page_content转换为与TF-IDF类似的表,即每一行(一段话)代表一个稀疏句向量
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 5

# 1.3 混合检索器
ensemble_retriever = EnsembleRetriever(
    retrievers = [bm25_retriever,vector_retriever],
    weights = [bm25_weight,vector_weight] # BM25 权重 0.3, 向量权重 0.7
)

# 1.4 测试检索效果
# query = "什么是机器翻译测试集中的冗余比（redundancy ratio）？请提供定义和计算方法。"
#
# res = ensemble_retriever.invoke(query)
#
# for i,doc in enumerate(res):
#     score = doc.metadata.get("relevance_score","N/A")
#     title = doc.metadata.get("title","N/A")[:100]
#     print(f" {i+1}. {title}...")

# 2. 查询改写
# 多查询改写目的 : 通过增加查询问题的数量,以增加与关键切片(包含正确答案的切片)的匹配概率
# 2.1 查询改写函数
from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_core.prompts import ChatPromptTemplate

def multi_query_rewrite(query,num_queries = 4):
    prompt = ChatPromptTemplate.from_template('''
角色:你是一个 arXiv 学术论文检索专家。
任务:用户提出一个中文查询，请生成{num_queries} 条 不同的英文查询。
约束:
1. 每条查询都要使用不同的学术表述方式和专业术语
2. 适当扩展相关关键词和同义表达
3. 保持原意不变，长度适中
4. 输出为列表形式，每条查询用英文逗号分隔，不要加其他解释
用户查询：{query}
''')
    rewritten_llm = ChatOpenAI(model="deepseek-flash",api_key=os.getenv('DS_API_KEY'),base_url=os.getenv('DS_API_BASE'))
    parser = CommaSeparatedListOutputParser()
    chain = prompt | rewritten_llm | parser
    res = chain.invoke({"num_queries" : num_queries,"query":query})
    return res
query = "什么是机器翻译测试集中的冗余比？"
# queries = multi_query_rewrite(query)
# for i,q in enumerate(queries):
#     print(f"{i+1}. {q}")

# 2.2 多查询混合检索
def multi_query_retrieval(query,k=5):
    '''
    多查询改写 + 混合检索 + 结果合并去重
    '''
    # 1. 生成多个改写查询
    rewritten_queries = multi_query_rewrite(query,num_queries=4)

    # 2.分别对每条查询进行检索
    res = [] # 最多 4 * 5 * 2 = 40 个切片
    seen_chunk_ids = set()
    for query in rewritten_queries:
        docs = ensemble_retriever.invoke(query)

        for doc in docs:
            chunk_id = doc.metadata.get("chunk_id")
            if chunk_id and chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(chunk_id)
                res.append(doc)

    # 3.显示结果
    print(f"一共返回 {len(res)} 条唯一chunk")
    return res

# res = multi_query_retrieval(query)
# print(res)

# 3.重排序
# 3.1 概念
# listwise 进行打分时,对整个列表中的所有内容,进行一块打分,相当于 batch
# pairwise 进行打分时,会进行逐个配对,然后分别进行打分
# 假设表格中一共有 6 个 chunk,pariwise 会进行多少次调用大模型 ? 15 Cn2;list wise直接进行一次到位
# 所以,再rerank的时候,把chunk与query匹配计算得分,listwise更加方便.

# 3.2 重排序函数
def rerank_with_llm(query,chunks,top_k = 10):
    if len(chunks) <= top_k:
        return chunks

    # 1. 构造提示词, 让LLM对所有候选切片进行排序
    res = []
    # 把 chunk里面的关键信息提取成字符串
    for i,doc in enumerate(chunks):
        title = doc.metadata.get("title","unknown")
        content = doc.page_content
        summary = doc.metadata.get("summary","unknown")
        res.append(f"""
[候选 {i}]\n论文标题 : {title}\n内容 : {content}\n摘要 : {summary}
""")
    context = '\n'.join(res)

    prompt = ChatPromptTemplate.from_template('''
角色 : 你是一个严格的学术论文检索排序专家。
用户查询 : {query}
上下文 : 里面是所有的候选切片 {context}
任务 : 你重新对上下文中的所有候选,进行精细排序,只保留最相关、最能直接回答用户查询的 {top_k} 条。
输出格式 : 请严格按照以下JSON格式输出(只输出候选的数字序号,不要添加任何解释):
{{
"ranked_indices":[0,3,1,7,19,...]  // 排序后的原始序号列表(保留前{top_k} 个)
}}
    ''')

    rerank_llm = ChatOpenAI(model = "deepseek-flash",api_key=os.getenv('DS_API_KEY'),base_url=os.getenv('DS_API_BASE'),temperature = 0.2,top_p = 0.6)
    parser = JsonOutputParser()

    chain = prompt | rerank_llm | parser
    res = chain.invoke({"context":context,"top_k":top_k,"query":query})
    print(res)
    return res

# 3.3 整理前面的函数
def full_retrival_with_rerank(query,top_k = 5):
    candidates = multi_query_retrieval(query)
    # 最终获取了一个索引列表
    docs_index = rerank_with_llm(query,candidates,top_k=top_k)
    final_docs = [candidates[i] for i in docs_index["ranked_indices"]]
    return final_docs

# 4.最终生成
ADVANCED_SYSTEM_PROMPT = """
你是一位严谨、专业、可信的**学术文献助手**，专门帮助用户基于 arXiv 论文上下文回答问题。
【任务要求】
1. 严格仅使用提供的上下文信息回答问题，**禁止添加任何外部知识或推测**。
2. 如果上下文无法充分回答，请明确说明“根据提供的论文内容，无法找到完整答案”。
3. 必须进行 Chain-of-Thought 思考：先分析上下文 → 判断是否存在冲突 → 组织逻辑答案。
4. 答案必须专业、逻辑清晰、结构分明，并提供可溯源的出处。
5. 如果检测到知识冲突，必须进行对比分析，不能简单罗列。

用户问题：
{query}
上下文（已按相关性排序）：
{context}


【输出格式要求】（必须严格遵守，返回纯 JSON，不要添加任何额外文字）
{{
"answer": "最终答案（自然流畅的中文，200-400字左右）",
"sources": [
{{
"title": "论文标题",
"arxiv_id": "arXiv:xxxx.xxxxx",
"chunk_index": 1
}}
],
"confidence": "high / medium / low"
}}
"""

def answer_question(query,top_k):
    """
    完整 RAG 流程：
    多查询改写 → 混合检索 → 重排序 → 生成答案
    """
    # # 1. 多查询改写 + 混合检索（召回较多候选）
    # candidates = multi_query_retrieval(query)

    # # 2. 重排序
    # docs_index = rerank_with_llm(query, candidates, top_k=top_k)
    # final_docs = [candidates[i] for i in docs_index["ranked_indices"]]
    final_docs = full_retrival_with_rerank(query,top_k=top_k)

    # 3. 构建上下文
    res = []
    # 把 chunk里面的关键信息提取成字符串
    for i, doc in enumerate(final_docs):
        title = doc.metadata.get("title", "unknown")
        content = doc.page_content
        summary = doc.metadata.get("summary", "unknown")
        res.append(f"""
    [候选 {i}]\n论文标题 : {title}\n内容 : {content}\n摘要 : {summary}
    """)
    context = "\n\n".join(res)
    # 5. 生成答案（使用高级提示词）
    prompt = ChatPromptTemplate.from_template(template = ADVANCED_SYSTEM_PROMPT)
    generate_llm = ChatOpenAI(model="deepseek-flash", api_key=os.getenv('DS_API_KEY'), base_url=os.getenv('DS_API_BASE'), temperature=0.2, top_p=0.6)

    parser = JsonOutputParser()

    chain = prompt | generate_llm | parser

    res = chain.invoke({"query":query,"context" : context})
    return res
res = answer_question(query,10)
with open('Langchain_rag/day06/answer.json','w',encoding='utf-8') as f:
    json.dump(res,f,ensure_ascii=False,indent=4)
