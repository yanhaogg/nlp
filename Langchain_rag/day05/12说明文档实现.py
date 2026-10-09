from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter
)
import numpy as np

def split_markdown_document(text,chunk_size= 512,overlap = 80):
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

with open('Langchain_rag/data/家用电视技术说明文档.md','r',encoding='utf-8') as f:
    docs = f.read()

chunks = split_markdown_document(docs)

chunk_lengths = [len(chunk.page_content) for chunk in chunks]

# 基本统计量
print(f"切片长度:{len(chunks)}")
print(f"最小长度:{np.min(chunk_lengths)} 字符")
print(f"最大长度:{np.max(chunk_lengths)} 字符")
print(f"平均长度:{np.mean(chunk_lengths)} 字符")
print(f"第一个切片的内容: \n{chunks[0].page_content}")

# 17 : 18