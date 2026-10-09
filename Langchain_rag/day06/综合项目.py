# 导入区
import chromadb
from langchain_chroma import Chroma
from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter
)
import numpy as np
import json
from pathlib import Path
import time
from openai import RateLimitError

# 导入 embeddings 模型
import os
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()


embeddings=OpenAIEmbeddings(
    model=os.getenv('QIANWEN'),
    api_key=os.getenv('BAIDU_QIANFAN_API_KEY'),
    base_url=os.getenv('BAIDU_QIANFAN_BASE_URL'),
    check_embedding_ctx_length=False#关闭 LangChain 的长度检查/token 化行为，让它直接传字符串
)

# 一.离线索引阶段
# 1. 数据处理
# 1.1 加载数据
loader_path = "Langchain_rag/pdf_data/A Persona-based Multi-turn Conversation Model in an Adversarial Learning Framework.pdf"

loader = PyMuPDF4LLMLoader(file_path=loader_path)
res = loader.load()
#res=res[0].page_content
# print(res)

# 1.2 数据清洗
import re
import hashlib


def basic_clean(text):
    # 1.去除多余空格
    text = re.sub(r" +", " ", text)
    # 2.去除多余换行
    text = re.sub(r"\n+", "\n", text)
    # 3.去除常见特殊字符
    text = re.sub(r'[■□]+', "", text)
    return text


def standard_clean(text):
    # 1.去除页码
    text = re.sub(r'(第\s*\d+\s*页|Page\s*\d+)', "", text)

    # 2.去除页眉页脚
    text = re.sub(r"(XX公司内部培训资料\s*|第\s*\d+\s*页|版权所有 © 2026\s*)", "", text)

    # 3.换行切断修复
    text = re.sub(r'(\w+)-\s*\n*\s*(\w+)', r'\1\2', text)

    # 4.哈希去重
    paragraphs = [p.strip() for p in text.split("\n")]
    seen = {}  # 查重字典
    cleaned = []  # 去重后的干净列表

    for para in paragraphs:
        # 特征提取 : MD5哈希 + 前150字符
        key = hashlib.md5(para[:150].encode('utf-8')).hexdigest()
        if key not in seen:
            seen[key] = para
            cleaned.append(para)
    return "\n".join(cleaned)

# 1.3 文本切片
def split_markdown_document(text,chunk_size= 1500,overlap = 200):
    '''技术文档专用切片器'''
    # 1. 按照 H2/H3 标题划分
    headers_to_split_on = [
        ("##", "章节"),
        ("###", "小节")
    ]
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=True  # 是否过滤掉标题内容
    )
    chunks = splitter.split_text(text)

    # 2.递归切分过长章节
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = overlap,
        separators = ["\n\n","\n","。","！", "？","；","，"])
    chunks = text_splitter.split_documents(chunks)

    return chunks

# 2.JSON 文件处理
# 2.1 定义加载函数
def load_metadata(path):
    if not path.exists():
        return {}
    with open(path,"r",encoding='utf-8') as f:
        return json.load(f)

# 2.2 合并 metadata 数据
def merge_metadata(chunks,json_metadata):
    '''
    chunks : 一篇 PDF 对应的所有切片
    json_metadata : 加载的同名 JSON 文件
    '''
    for i,chunk in enumerate(chunks):
        arxiv_id = json_metadata.get("arxiv_id","unknown")
        chunk_id = f"{arxiv_id}_chunk_{i:03d}"

        # merged_meta 是为了整理json里面的内容,防止有些json没有对应的字段
        merged_meta = {
            "chunk_id": chunk_id,
            "arxiv_id": json_metadata.get("arxiv_id"),
            "title": json_metadata.get("title"),
            "authors": json_metadata.get("authors", []),
            "published": json_metadata.get("published"),
            "updated": json_metadata.get("updated"),
            "categories": json_metadata.get("categories", []),
            "source": chunk.metadata.get("source"),
        }
        # 对原本的metadata进行更新,如果已经存在则覆盖
        chunk.metadata.update(merged_meta)
    return chunks


# 3.入库准备
# 3.1 入库文件扫描 : (1)检测pdf是否有配对的Json,(2)是否已经完成入库
# 查看 01扫描库里文件.py
 
# 3.2 正式入库配置
# ===========   配置    ===========
paper_dir = Path('Langchain_rag/pdf_data')
db_dir = 'Langchain_rag/db'
collection_name = "nlp_papers"
batch_size = 20
scan_path = 'Langchain_rag/day06/scan.json' # 处理完一篇论文后,把论文前缀增加到 2 的文件夹里面

# 3.3 数据库加载
client = chromadb.PersistentClient(path = db_dir)

db = Chroma(
    client = client,
    collection_name=collection_name,
    embedding_function=embeddings
)

# 3.4 扫描文件读取
with open(scan_path,'r',encoding='utf-8') as f:
    data = json.load(f)
pdf_list = data["1"]


# 4.正式入库
success_count = 0
for name in pdf_list[:]:
    #  path = 文件夹所在目录 + 前缀名
    path = paper_dir / Path(name)
    # 给  path/json  添加后缀
    pdf_path = path.with_suffix('.pdf')
    json_path = path.with_suffix('.json')

    # 4.1 读取 + 清洗 + 切片
    loader = PyMuPDF4LLMLoader(pdf_path)
    doc = loader.load()
    raw_text = '\n'.join([p.page_content for p in doc])
    text = basic_clean(raw_text)
    cleaned_text = standard_clean(text)
    chunks = split_markdown_document(cleaned_text)
    if not chunks:
        continue

    # 4.2 读取 metadata
    json_data = load_metadata(json_path)

    # 4.3 把metadata与切片融合
    # 一个Document 包含 page_content,metadata ,是Document自己包含的数据信息
    # 入库的时候,一定要指定 ids, 因为它是该数据(切片)在数据库中唯一标识符
    merged_chunks = merge_metadata(chunks,json_data)
    arxiv_id = json_data.get("arxiv_id","unknown")
    if arxiv_id != "unknown":
        ids = [f"{arxiv_id}_chunk_{idx:03d}" for idx in range(len(merged_chunks))]
    else:
        ids = [f"{name}_chunk_{idx:03d}" for idx in range(len(merged_chunks))]

    # 4.4 向量化入库
    before = len(db.get()['ids']) # 入库之前看一下有多少切片
    #
    # for i in range(0,len(merged_chunks),batch_size):
    #     # batch = [Document(切片1),Document(切片2),...] 一共20个
    #     batch = merged_chunks[i:i + batch_size]
    #     # batched_ids = [..chunk_1,..chunk_2,...] 一共20个
    #     batched_ids = ids[i:i+batch_size]
    #     db.add_documents(batch,ids=batched_ids)
    #     #time.sleep(1)
    for i in range(0, len(merged_chunks), batch_size):
        batch = merged_chunks[i:i + batch_size]
        batched_ids = ids[i:i + batch_size]

        for retry in range(5):
            try:
                db.add_documents(
                    batch,
                    ids=batched_ids
                )
                break

            except RateLimitError:
                wait_time = 2 ** retry

                print(
                    f"触发 TPM 限流，"
                    f"{wait_time}s 后重试..."
                )

                time.sleep(wait_time)
    after = len(db.get()["ids"])
    added = after-before
    print()#换行
    print(f"{name} 成功写入 {added} 个chunk(累计{after}个)")
    if added:
        data['1'].pop(0)
        data['2'].append(name)
        success_count += 1

    with open(scan_path,'w',encoding='utf-8') as f:
        json.dump(data,f,ensure_ascii=False)
print(f"成功处理论文数 : {success_count} / {len(pdf_list)}")
print(f"还未处理论文数 : {len(data['1'])}")





